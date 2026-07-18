import os
import time
from pathlib import Path

import niquests
from niquests.auth import HTTPDigestAuth

from libonvif.devices.camera import get_camera_by_ip
from libonvif.utils.server import EventServer
from libonvif.utils.subscriber import SubscriptionManager
import logging
from importlib.metadata import version as get_installed_version
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
CAMERA_IP = os.environ.get("CAMERA_IP", "10.1.1.78")
CAMERA_USERNAME = os.environ.get("CAMERA_USERNAME", "")
CAMERA_PASSWORD = os.environ.get("CAMERA_PASSWORD", "")
EVENT_SERVER_PORT = int(os.environ.get("EVENT_SERVER_PORT", "8856"))
SNAPSHOT_DIR = Path(__file__).parent / "snapshots"
OPENCLAW_HOOK_URL = os.environ.get("OPENCLAW_HOOK_URL", "http://127.0.0.1:18789/hooks/agent")
OPENCLAW_HOOK_TOKEN = os.environ.get("OPENCLAW_HOOK_TOKEN", "")

_camera = None  # set in main(), used by on_camera_events to fetch a snapshot


def fetch_snapshot() -> Path | None:
    """
    Download the camera's current snapshot as a JPEG file, saved with a
    timestamped filename. Returns the saved path, or None on failure.

    Uses camera.profiles[0].snapshot_uri (populated by get_camera_by_ip)
    rather than hardcoding a URL, and the same HTTP Digest auth pattern
    used elsewhere for snapshot access.
    """
    snapshot_uri = _camera.profiles[0].snapshot_uri
    try:
        response = niquests.get(
            snapshot_uri,
            auth=HTTPDigestAuth(CAMERA_USERNAME, CAMERA_PASSWORD),
            timeout=10,
        )
        response.raise_for_status()
    except Exception as e:
        print(f"Failed to fetch snapshot: {e}")
        return None

    SNAPSHOT_DIR.mkdir(exist_ok=True)
    filename = SNAPSHOT_DIR / f"snapshot_{int(time.time())}.jpg"
    filename.write_bytes(response.content)
    print(f"Saved snapshot to {filename}")
    return filename


def notify_openclaw(alarm_summary: str) -> None:
    """
    POST to OpenClaw's /hooks/agent endpoint, telling the agent a camera
    event occurred. Deliberately does NOT send image data itself - the
    OpenClaw agent already has this same camera MCP server configured
    (see mcp.servers.camera in openclaw.json), so it can fetch and view
    a fresh snapshot directly via its own tools rather than us pushing
    image bytes through a webhook's plain-text message field.
    """
    payload = {
        "message": (
            f"Motion event received from the camera at {CAMERA_IP}. "
            f"({alarm_summary}) "
            "Please use the camera tools to fetch a current snapshot and "
            "briefly describe what you see."
        ),
        "name": "Camera Motion",
        "wakeMode": "now",
    }
    try:
        response = niquests.post(
            OPENCLAW_HOOK_URL,
            json=payload,
            headers={"Authorization": f"Bearer {OPENCLAW_HOOK_TOKEN}"},
            timeout=10,
        )
        response.raise_for_status()
        print(f"Notified OpenClaw: {response.json()}")
    except Exception as e:
        print(f"Failed to notify OpenClaw: {e}")


def on_camera_events(alarms: list[dict]) -> None:
    """
    Phase 3: on any event (unfiltered - the subscription-confirmation
    event that fires immediately on subscribe is still being used as our
    trigger for now), save a local snapshot for our own record, then
    notify OpenClaw via its /hooks/agent webhook so its own agent (which
    has vision + camera MCP tools + real delivery channels) picks up
    from there.
    """
    summary_parts = []
    for alarm in alarms:
        print("-" * 40)
        for key, value in alarm.items():
            print(f"{key}: {value}")
        summary_parts.append(str(alarm.get("topic", "unknown topic")))
    print()

    fetch_snapshot()  # kept for our own local audit trail
    notify_openclaw(", ".join(summary_parts) or "event received")


def main():

    try:
        libonvif_version = get_installed_version("libonvif")
    except Exception as e:
        logger.error(f"Failed to get libonvif version: {e}")
        libonvif_version = None

    print(f"LIBONVIF VERSION: {libonvif_version}")


    camera = get_camera_by_ip(CAMERA_IP, CAMERA_USERNAME, CAMERA_PASSWORD)
    global _camera
    _camera = camera

    print("Available events:")
    for event in camera.event_properties.topic_set:
        print(f"  {event}")
    print()

    event_server = EventServer("0.0.0.0", EVENT_SERVER_PORT, on_camera_events)
    event_server.start()

    subscription_manager = SubscriptionManager(camera)
    # event=None subscribes to everything - phase 1 wants all events,
    # unfiltered, to confirm the pipeline before narrowing to motion only.
    subscription_manager.subscribe_push_event(camera, "0.0.0.0", EVENT_SERVER_PORT, None)

    print(f"Subscribed to all events on {CAMERA_IP}, listening on port {EVENT_SERVER_PORT}.")

    try:
        input("Press enter key to quit\n")
    except KeyboardInterrupt:
        pass
    finally:
        subscription_manager.unsubscribe_events(camera)
        event_server.stop()


if __name__ == "__main__":
    main()
