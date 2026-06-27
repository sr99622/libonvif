from libonvif.devices.camera import get_camera_by_ip
from argparse import ArgumentParser
from libonvif.utils.server import EventServer
from libonvif.utils.subscriber import SubscriptionManager

def on_camera_events(alarms: list[dict[str, str]]) -> None:
    for alarm in alarms:
        for key, value in alarm.items():
            print(f"{key}: {value}")
    print("\nPress enter key to quit\n")

parser = ArgumentParser()
parser.add_argument("-i", "--ip_address", help="Camera IP address")
parser.add_argument("-t", "--host", default="0.0.0.0", help="Host IP address")
parser.add_argument("-u", "--username", help="User name")
parser.add_argument("-p", "--password", help="Password")
parser.add_argument("-e", "--event", help="Event Filters")
args = parser.parse_args()

camera = None
subscription_manager = None
event_server = None

try:
    camera = get_camera_by_ip(args.ip_address, args.username, args.password)
    print("Available Events")
    for event in camera.event_properties.topic_set:
        print(event)
    print()

    port = 8856
    event_server = EventServer(args.host, port, on_camera_events)
    event_server.start()
    subscription_manager = SubscriptionManager(camera)

    if args.event:
        topics = args.event.split(",")
        for topic in topics:
            if topic in camera.event_properties.topic_set:
                subscription_manager.subscribe_push_event(camera, args.host, port, topic)
            else:
                print(f"** WARNING ** Did not find matching event for {topic}")
    else:
        subscription_manager.subscribe_push_event(camera, args.host, port, None)

    key = input("Press enter key to quit\n")

except Exception as ex:
    print(f"error: {ex}")

finally:
    if camera and subscription_manager:
        subscription_manager.unsubscribe_events(camera)
    if event_server:
        event_server.stop()
