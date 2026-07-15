import asyncio
import os
from datetime import datetime

import uvicorn
from starlette.middleware.cors import CORSMiddleware
from starlette.requests import Request
from starlette.responses import StreamingResponse
from starlette.types import ASGIApp, Receive, Scope, Send

from mcp.server.fastmcp import FastMCP, Context
from mcp.server.elicitation import AcceptedElicitation, DeclinedElicitation, CancelledElicitation
from mcp.server.transport_security import TransportSecuritySettings
from libonvif.devices.camera import get_camera_by_ip
from pydantic import BaseModel

# DNS-rebinding protection stays enabled, but we explicitly allow the
# llama.cpp web UI's origin (10.1.1.87) alongside the usual localhost
# entries FastMCP would otherwise add automatically. Supplying our own
# transport_security here means FastMCP's auto-default (localhost-only)
# is skipped in favor of this one.
mcp = FastMCP(
    "sse-example",
    transport_security=TransportSecuritySettings(
        enable_dns_rebinding_protection=True,
        allowed_hosts=["127.0.0.1:*", "localhost:*", "[::1]:*", "10.1.1.87:*"],
        allowed_origins=[
            "http://127.0.0.1:*",
            "http://localhost:*",
            "http://[::1]:*",
            "http://10.1.1.87:*",
        ],
    ),
)

class TripTypeResponse(BaseModel):
    value: str

@mcp.tool()
async def add(a: int, b: int) -> int:
    """Add two numbers together."""
    return a + b

@mcp.tool()
async def example_elicit_tool(context: Context) -> str:
    """
    Example tool that asks the user a question via MCP elicitation, to
    test whether a given client (e.g. llama.cpp's web UI) implements the
    client side of the elicitation flow - Claude Desktop returned
    "Method not found" when this was tried there.
    """
    result = await context.elicit(
        message="What type of trip are you planning? Options: business, leisure, family, adventure",
        schema=TripTypeResponse,
    )
    if isinstance(result, AcceptedElicitation):
        return result.data.value
    elif isinstance(result, DeclinedElicitation):
        return "DECLINED"
    elif isinstance(result, CancelledElicitation):
        return "CANCELLED"
    return "INVALID RESPONSE"

@mcp.tool()
async def get_camera(ip_address: str) -> str:
    """
    Query a camera by IP address and return its full state as a JSON
    string. Credentials come from the CAMERA_USERNAME/CAMERA_PASSWORD
    environment variables - the same pattern used by the camera MCP
    server. Added here as a test of whether real camera data (a large,
    deeply nested JSON payload) flows correctly through the
    streamable-http transport and this server's CORS/session setup, not
    just the trivial add() tool above.

    Args:
        ip_address: The IP address of the camera to query.

    Returns:
        The camera's JSON representation, as produced by Camera.to_json().
    """
    camera = get_camera_by_ip(ip_address, os.environ.get("CAMERA_USERNAME", ""), os.environ.get("CAMERA_PASSWORD", ""))
    return camera.to_json()


class PrivateNetworkAccessMiddleware:
    """
    Adds the Access-Control-Allow-Private-Network header some Chromium
    browsers require (in addition to normal CORS) before allowing a page
    served from a non-loopback origin to fetch a loopback address like
    127.0.0.1. Without this, the browser can reject the request before
    it ever reaches this server, showing up client-side as a generic
    "Failed to fetch" with no server-side log at all.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        async def send_wrapper(message):
            if message["type"] == "http.response.start":
                headers = message.setdefault("headers", [])
                headers.append((b"access-control-allow-private-network", b"true"))
            await send(message)

        await self.app(scope, receive, send_wrapper)


async def event_stream(request: Request) -> StreamingResponse:
    """
    Plain Server-Sent Events endpoint, independent of the MCP protocol -
    just a raw text/event-stream that emits one tick every 5 seconds.
    Built to test/observe the SSE mechanism itself directly (e.g. via
    curl -N http://127.0.0.1:8000/events, or a browser EventSource),
    separate from anything MCP-specific like tool calls or sessions.
    """

    async def generator():
        count = 0
        try:
            while True:
                await asyncio.sleep(5)
                count += 1
                yield f"data: tick {count} at {datetime.now().isoformat()}\n\n"
        except asyncio.CancelledError:
            pass

    return StreamingResponse(generator(), media_type="text/event-stream")


def main():
    app = mcp.streamable_http_app()
    app.add_route("/events", event_stream, methods=["GET"])
    app.add_middleware(PrivateNetworkAccessMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
        # The streamable-http transport returns a session ID in a custom
        # response header on the initialize call, and expects it echoed
        # back on every subsequent request. Browsers hide custom response
        # headers from JS by default unless the server explicitly exposes
        # them via CORS - without this, the client never sees the session
        # ID and every follow-up request gets rejected as missing one.
        expose_headers=["mcp-session-id"],
    )
    # Defaults: host=127.0.0.1, port=8000, path=/mcp
    # -> server listens at http://127.0.0.1:8000/mcp
    uvicorn.run(app, host="127.0.0.1", port=8000)

if __name__ == "__main__":
    main()
