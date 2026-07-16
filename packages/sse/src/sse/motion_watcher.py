import os

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


def on_camera_events(alarms: list[dict]) -> None:
    """
    Phase 1: print every event received, completely unfiltered, to
    confirm the push subscription + event server pipeline works
    end-to-end before adding topic filtering or snapshot/model logic.
    """
    for alarm in alarms:
        print("-" * 40)
        for key, value in alarm.items():
            print(f"{key}: {value}")
    print()


def main():

    try:
        libonvif_version = get_installed_version("libonvif")
    except Exception as e:
        logger.error(f"Failed to get libonvif version: {e}")
        libonvif_version = None

    print(f"LIBONVIF VERSION: {libonvif_version}")


    camera = get_camera_by_ip(CAMERA_IP, CAMERA_USERNAME, CAMERA_PASSWORD)

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
