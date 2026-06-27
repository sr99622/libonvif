<h2>libonvif</h2>

libonvif is a pure python library implementing the ONVIF client protocol used for controlling IP cameras. The repository includes a Terminal User Interface program, onvif-tui, that demonstrates libonvif programming and can be used to query and control cameras.

The library and applications were built using [uv](https://docs.astral.sh/uv/getting-started/installation/) for python requirements, and is recommended for development.

<h3>Legacy Version</h3>

libonvif was originally built in C. That code can be found in the legacy folder of the repository. Onvif GUI was a Graphical User Interface application that has been replaced with [Cayenue](https://github.com/sr99622/Cayenue).

<h2>onvif-tui</h2>

<image src=assets/onvif-tui.gif>

&nbsp;

onvif-tui is a Terminal User Interface application with many features.

* Automatically find cameras
* Control video and audio settings
* Pan Tilt Zoom motion
* Receive or Pull camera events
* Toggle relays 

&nbsp;

onvif-tui can be installed using [pipx](https://pipx.pypa.io/stable/how-to/install-pipx/)

```
pipx install onvif-tui
```

The application will run without any command line arguments. The username and password will be required for camera authentication. The -i argument is optional for situations where there are multiple network interfaces on the host computer. The -m argument is used to connect with a camera without using discovery.

```
-u username for camera authentication
-p passwword for camera authentication
-m camera ip address for manual camera discovery
-i host local ip address for binding discovery broadcast 
```

The following command will work in most cases where the cameras reside on the same subnet as the host computer and there are no firewall issues. If the cameras are remote to the host, or firewall issues cannot be overcome, use -m option to address the camera directly by IP address.

```
onvif-tui -u <username> -p <password>
```

<details><summary>Firewall Issues</summary>

&nbsp;

```
Requested Ports:

port 3702/UDP is for WS-Discovery (Web Services Dynamic Discovery) protocols.

port 8856/TCP is for HTTP server used to receive camera events
```

<i>Events can be pulled by the host if unable to receive events.</i>

&nbsp;

</details>

&nbsp;

<h2>libonvif Programming Examples</h2>

* <h3>Simple Camera Query</h3>

The following program shows a simple camera query using libonvif. You will need to supply the camera ip address, user name and password, the output is a full printout of the camera data.

```
from libonvif.devices.camera import get_camera_by_ip
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("-i", "--ip_address", help="Camera IP address")
parser.add_argument("-u", "--username", help="User name")
parser.add_argument("-p", "--password", help="Password")
args = parser.parse_args()

try:
    camera = get_camera_by_ip(args.ip_address, args.username, args.password)
    print(camera)
except Exception as ex:
    print(f"error: {ex}")

```

&nbsp;

* <h3>Receive Camera Events</h3>

This program will receive event notifications from the camera. At a minimum, you will need to supply the camera ip address, user name and password. If the -e flag is not used, all events from the camera will be recieved. Received events are processed by a callback function, in this case `on_camera_events`. Event filters can be added to show only flagged events. Multiple events can be entered using comma delimiter. Available event filters are printed out by the program at launch. Please refer to the pull_events.py example program for an alternate event mechanism if the host is unable to receive traffic on tcp port.

```
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
```

&nbsp;

* <h3>Discovery</h3>

Cameras can be found using the discovery program below. The program will prompt for a user name and password for each camera. If you would like to find all cameras using the same password, adjust get_camera_credentials to fixed values and change the use_threads flag of the discover function call to True. The -i flag can be used to set the network interface through which discovery is performed.

```
from getpass import getpass
from libonvif.utils.adapters import find_adapters
from libonvif.devices.camera import Camera, discover
from argparse import ArgumentParser
import sys

def get_camera_credentials(camera: Camera) -> None:
    print(f"\nEnter credentials for {camera.name} : {camera.xaddr}")
    camera.username = input("Username: ").strip()
    camera.password = getpass("Password: ")

def on_error(xaddr: str, ex:Exception) -> None:
    print(f"error: {xaddr} - {ex}")

def camera_filled(camera: Camera) -> None:
    print(f"Camera Filled: {camera.name} : {camera.device_information.serial_number}")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-i", "--ip_address", default="0.0.0.0", help="Local Network Interface IP address")
    args = parser.parse_args()
    ip_address = args.ip_address
    if sys.platform == "win32" and ip_address == "0.0.0.0":
        ips = find_adapters()
        if len(ips):
            ip_address = ips[0]
            print(f"host ip addresses: {ips}")

    cameras = discover(ip_address, 
                       get_camera_credentials, 
                       on_error=on_error, 
                       camera_filled=camera_filled, 
                       use_threads=False)

    print(f"Found {len(cameras)} {"camera" if len(cameras) == 1 else "cameras"}")
```

&nbsp;

<h2>Dependencies</h2>

XML processing is handled with [lxml](https://lxml.de/)

HTTP client uses [niquests](https://niquests.readthedocs.io/en/latest/)

The onvif-tui application is built using [textual](https://textual.textualize.io/)

&nbsp;

<h2>License</h2>

Copyright (c) 2026  Stephen Rhodes

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
