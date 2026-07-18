import os
import time
from datetime import datetime
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
CAMERA_IP = os.environ.get("CAMERA_IP", "10.1.1.77")
CAMERA_USERNAME = os.environ.get("CAMERA_USERNAME", "")
CAMERA_PASSWORD = os.environ.get("CAMERA_PASSWORD", "")
EVENT_SERVER_PORT = int(os.environ.get("EVENT_SERVER_PORT", "8856"))
SNAPSHOT_DIR = Path(__file__).parent / "snapshots"
OPENCLAW_HOOK_URL = os.environ.get("OPENCLAW_HOOK_URL", "http://127.0.0.1:18789/hooks/agent")
OPENCLAW_HOOK_TOKEN = os.environ.get("OPENCLAW_HOOK_TOKEN", "")

# Reserved subdirectory inside OpenClaw's own workspace where the agent
# should save its own copy of the snapshot. $WORKSPACE_DIR is substituted
# by OpenClaw itself when a tool call uses it - not something we resolve
# here. This is separate from SNAPSHOT_DIR above, which is our own local
# copy on this machine.
OPENCLAW_SNAPSHOT_SUBDIR = "camera-events"

_camera = None  # set in main(), used by on_camera_events to fetch a snapshot


def build_snapshot_filename(camera_ip: str, event_type: str) -> str:
    """
    Shared naming scheme for every snapshot/marker file this watcher
    produces (both our own local copies and the filename we instruct
    OpenClaw to use for its own copy), so files can be found by camera,
    event type, and time without needing to open them:

        {camera_ip with dashes instead of dots}_{event_type}_{timestamp}.jpg

    e.g. "10-1-1-77_motion_true_20260718T215035.jpg"

    event_type is expected to be "motion_true" or "motion_false" for now,
    but is a free string so other event types can reuse this later.
    """
    safe_ip = camera_ip.replace(".", "-")
    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    return f"{safe_ip}_{event_type}_{timestamp}.jpg"


def fetch_snapshot(filename: str) -> Path | None:
    """
    Download the camera's current snapshot as a JPEG file, saved under
    the given filename (see build_snapshot_filename). Returns the saved
    path, or None on failure.

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
    path = SNAPSHOT_DIR / filename
    path.write_bytes(response.content)
    print(f"Saved snapshot to {path}")
    return path


def save_empty_marker(filename: str) -> Path:
    """
    Save a 0-byte marker file under the given filename, recording that a
    motion-ended (State: false) event occurred without fetching or
    storing an actual image for it - keeps a complete record of every
    motion state transition without the storage/noise cost of a real
    snapshot for events we don't otherwise act on.
    """
    SNAPSHOT_DIR.mkdir(exist_ok=True)
    path = SNAPSHOT_DIR / filename
    path.touch()
    print(f"Recorded empty marker at {path}")
    return path


def notify_openclaw(filename: str) -> None:
    """
    POST to OpenClaw's /hooks/agent endpoint, telling the agent exactly
    what to do and where to save its own copy of the snapshot - this
    replaces an earlier, vaguer message ("please describe what you see")
    that left the agent to rediscover the correct tool sequence through
    several failed attempts (get_snapshot_image_base64_encoded and the
    browser tool both proved unusable to it) before eventually landing on
    download_snapshot_to_file + read. Naming the exact tools and path up
    front skips that trial and error on every single motion event.
    """
    file_path = f"$WORKSPACE_DIR/{OPENCLAW_SNAPSHOT_SUBDIR}/{filename}"
    payload = {
        "message": (
            f"Motion detected on the camera at {CAMERA_IP}. Do the following:\n"
            f"1. Call camera__download_snapshot_to_file with url set to that "
            f"camera's snapshot_uri (from camera__get_camera) and file_path "
            f"set to exactly \"{file_path}\".\n"
            f"2. Call read on that same path to view the image.\n"
            f"3. Briefly describe what you see.\n"
            "Do not use get_snapshot_image_base64_encoded or the browser tool "
            "for this - go directly to download_snapshot_to_file, then read."
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
    Phase 4: filters on the VideoSource/MotionAlarm State field itself
    (topic-level filtering was already narrowed to just this topic
    earlier). State: "true" (real motion) saves a real local snapshot and
    notifies OpenClaw with explicit instructions. State: "false" (motion
    ended) only records a 0-byte local marker file - no OpenClaw
    notification at all, since there's nothing useful for an agent to do
    with a "motion stopped" event, and spending a run on it would just
    reintroduce the noise the topic filter was meant to cut.
    """
    for alarm in alarms:
        print("-" * 40)
        for key, value in alarm.items():
            print(f"{key}: {value}")
    print()

    for alarm in alarms:
        if alarm.get("topic") != "VideoSource/MotionAlarm":
            continue

        is_motion = str(alarm.get("data", {}).get("State", "")).lower() == "true"
        event_type = "motion_true" if is_motion else "motion_false"
        filename = build_snapshot_filename(CAMERA_IP, event_type)

        if is_motion:
            fetch_snapshot(filename)
            notify_openclaw(filename)
        else:
            save_empty_marker(filename)


def main():

    if not OPENCLAW_HOOK_TOKEN:
        raise SystemExit(
            "ERROR: OPENCLAW_HOOK_TOKEN is not set.\n"
            "This must match the 'hooks.token' value in openclaw.json, or "
            "every notify_openclaw() call will silently fail later with a "
            "confusing 'Illegal header value' error instead of this clear "
            "one. Set it before running, e.g.:\n\n"
            "    OPENCLAW_HOOK_TOKEN=<your-token> uv run src/sse/motion_watcher.py\n"
        )

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
    # Filtering to VideoSource/MotionAlarm only - subscribing to everything
    # (event=None) was sending an excessive volume of unrelated alarms to
    # OpenClaw. This is a real server-side ONVIF topic filter (embedded in
    # the Subscribe request itself), so the camera only pushes matching
    # events - it isn't just discarding unwanted ones on our end after
    # they've already arrived.
    subscription_manager.subscribe_push_event(camera, "0.0.0.0", EVENT_SERVER_PORT, "VideoSource/MotionAlarm")

    print(f"Subscribed to VideoSource/MotionAlarm on {CAMERA_IP}, listening on port {EVENT_SERVER_PORT}.")

    try:
        input("Press enter key to quit\n")
    except KeyboardInterrupt:
        pass
    finally:
        subscription_manager.unsubscribe_events(camera)
        event_server.stop()


if __name__ == "__main__":
    main()
