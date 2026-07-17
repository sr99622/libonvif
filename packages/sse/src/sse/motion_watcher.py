import base64
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
LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "http://10.1.1.87:8080")

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


def describe_image(image_path: Path) -> str | None:
    """
    Send a JPEG image to the local vision-capable model (llama.cpp's
    OpenAI-compatible chat completions endpoint) and return its
    description. Returns None on failure.

    llama.cpp is not running in router mode here (a single model was
    loaded directly), so the "model" field below is effectively ignored -
    it's required by the API shape but there's nothing to route between.
    """
    image_bytes = image_path.read_bytes()
    image_b64 = base64.b64encode(image_bytes).decode("utf-8")

    payload = {
        "model": "local-model",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe what you see in this security camera snapshot."},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
                ],
            }
        ],
    }

    try:
        response = niquests.post(f"{LLM_BASE_URL}/v1/chat/completions", json=payload, timeout=60)
        response.raise_for_status()
    except Exception as e:
        print(f"Failed to get description from model: {e}")
        return None

    data = response.json()
    return data["choices"][0]["message"]["content"]


def on_camera_events(alarms: list[dict]) -> None:
    """
    Phase 2: on any event (unfiltered - the subscription-confirmation
    event that fires immediately on subscribe is being used as our
    trigger for now, ahead of adding real topic filtering), fetch and
    save a snapshot as a JPEG file. Sending it to the model comes next.
    """
    for alarm in alarms:
        print("-" * 40)
        for key, value in alarm.items():
            print(f"{key}: {value}")
    print()

    if filename := fetch_snapshot():
        description = describe_image(filename)
        if description:
            print(f"Description: {description}\n")


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
