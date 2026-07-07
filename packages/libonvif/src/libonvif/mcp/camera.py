import logging
from libonvif.utils.adapters import find_adapters
from libonvif.devices.camera import Camera, discover, get_camera_by_ip
from mcp.server.fastmcp import FastMCP
import os
import sys
import webbrowser

logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)
logger = logging.getLogger(__name__)

mcp = FastMCP("camera")

USER_AGENT = "camera-app/1.0"

def get_camera_credentials(camera: Camera) -> None:
    camera.username = os.environ.get("CAMERA_USERNAME", "")
    camera.password = os.environ.get("CAMERA_PASSWORD", "")

def on_error(xaddr: str, ex: Exception) -> None:
    logger.debug(f"error: {xaddr} - {ex}")

def camera_filled(camera: Camera) -> None:
    logger.debug(f"Camera Filled: {camera.name} : {camera.device_information.serial_number}")

@mcp.tool()
async def stream_camera(camera: Camera) -> str:
    #http://10.1.1.76:8889/AMC014641NE6L35AT8/MediaProfile000
    url = f"http://10.1.1.76:8889/{camera.device_information.serial_number}/"

@mcp.tool()
async def get_snapshot(url: str) -> str:
    """
    Open a URL in the user's default web browser.

    Args:
        url: The full URL to open, e.g. "https://example.com"

    Returns:
        A confirmation message.
    """
    if not (url.startswith("http://") or url.startswith("https://")):
        return f"Refused to open '{url}': must start with http:// or https://"

    curl = f"{url[:7]}{os.environ.get("CAMERA_USERNAME", "")}:{os.environ.get("CAMERA_PASSWORD", "")}@{url[7:]}"
    opened = webbrowser.open(curl)
    if opened:
        return f"Opened {url} in default browser."
    else:
        return f"Failed to open {url}."
    
@mcp.tool()
async def get_camera(ip_address: str) -> str:
    camera = get_camera_by_ip(ip_address, os.environ.get("CAMERA_USERNAME", ""), os.environ.get("CAMERA_PASSWORD", ""))
    return f"{camera}"

@mcp.tool()
async def get_cameras() -> str:
    """Get cameras on the local network."""

    ip_address = "0.0.0.0"
    if sys.platform == "win32":
        ips = find_adapters()
        if len(ips):
            ip_address = ips[0]
            logger.debug(f"host ip addresses: {ips}")

    cameras = discover(ip_address,
                       get_camera_credentials,
                       on_error=on_error,
                       camera_filled=camera_filled,
                       use_threads=True)
    
    logger.debug(f"Found {len(cameras)} {"camera" if len(cameras) == 1 else "cameras"}")

    names = []
    for camera in cameras:
        names.append(f"{camera.name} : {camera.xaddr} : {camera.device_information.serial_number}")
    
    return "\n--\n".join(names)

def main():
    logger.debug("Server starting...")
    mcp.run(transport="stdio")

if __name__ == "__main__":
    main()