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