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

