from libonvif.devices.camera import Camera, find_camera_manually
from argparse import ArgumentParser

class Manual():
    def __init__(self, args:ArgumentParser):
        self.ip_address = args.ip_address
        self.username = args.username
        self.password = args.password

    def get_camera_credentials(self, camera: Camera) -> None:
        camera.username = self.username
        camera.password = self.password

    def on_error(self, xaddr: str, ex:Exception) -> None:
        print(f"An error has occurred at {xaddr}")
        print(ex)

    def camera_filled(self, camera: Camera) -> None:
        print(f"camera serial number: {camera.device_information.serial_number}")

    def __call__(self) -> None:
        cameras = find_camera_manually(self.ip_address, self.get_camera_credentials, self.on_error, self.camera_filled)
        print("finished")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-i", "--ip_address", help="Camera IP address")
    parser.add_argument("-u", "--username", help="User name")
    parser.add_argument("-p", "--password", help="Password")
    args = parser.parse_args()
    manual = Manual(args)
    manual()
