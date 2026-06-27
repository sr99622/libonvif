from time import sleep
from datetime import datetime
import threading
from libonvif.devices.camera import Camera, get_camera_by_ip, pull_messages
from libonvif.datastructures.event import parse_notify
from argparse import ArgumentParser
from libonvif.utils.subscriber import SubscriptionManager
from concurrent.futures import ThreadPoolExecutor
from itertools import cycle

user_input = None
print_lock = threading.Lock()

def display_output(msg: str, end_char: str = None) -> None:
    with print_lock:
        print(msg, end=end_char, flush=True)

def get_input():
    global user_input
    while True:
        user_input = input()

def show_notifications(camera: Camera, xaddr: str) -> None:
    xml = pull_messages(camera, xaddr)
    if alarms := parse_notify(args.ip_address, xml):
        display_output("\r ", end_char="")
        display_output("")
    for alarm in alarms:
        for key, value in alarm.items():
            display_output(f"{key}: {value}")
    
parser = ArgumentParser()
parser.add_argument("-i", "--ip_address", help="Camera IP address")
parser.add_argument("-u", "--username", help="User name")
parser.add_argument("-p", "--password", help="Password")
parser.add_argument("-e", "--event", help="Add Event")
args = parser.parse_args()

executor = ThreadPoolExecutor(max_workers=8)
camera = None
subscription_manager = None
spinner = cycle("|/-\\")

try:
    camera = get_camera_by_ip(args.ip_address, args.username, args.password)
    print("Available Events")
    for event in camera.event_properties.topic_set:
        print(event)
    print("\npress enter key to quit\n")

    subscription_manager = SubscriptionManager(camera)
    active = set()

    if args.event:
        topics = args.event.split(",")
        for topic in topics:
            if topic in camera.event_properties.topic_set:
                subscription_manager.subscribe_pull_event(camera, topic)
            else:
                print(f"** WARNING ** Did not find matching event for {topic}")
    else:
        subscription_manager.subscribe_pull_event(camera, None)

    input_thread = threading.Thread(target=get_input, daemon=True)
    input_thread.start()

    while True:
        if user_input is not None:
            break

        sleep(1)
        display_output(f"\r{next(spinner)}", end_char="")

        # avoid overloading camera with pull requests
        active = {f for f in active if not f.done()}
        for reference in camera.subscription_references:
            if len(active) < 8:
                active.add(executor.submit(show_notifications, camera, reference.xaddr))

except Exception as ex:
    print(f"error: {ex}")

finally:
    executor.shutdown(wait=False, cancel_futures=True)
    if camera and subscription_manager:
        subscription_manager.unsubscribe_events(camera)
