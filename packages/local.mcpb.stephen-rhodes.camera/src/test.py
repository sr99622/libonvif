import os
import sys
import asyncio
import logging
from libonvif.devices.camera import Camera, discover
from libonvif.utils.adapters import find_adapters
import niquests as requests
from camera import get_camera_mcp_version, set_camera_profile_resolution

from niquests.auth import HTTPDigestAuth

logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)
logger = logging.getLogger(__name__)

def get_camera_credentials(camera: Camera) -> None:
    camera.username = os.environ.get("CAMERA_USERNAME", "")
    camera.password = os.environ.get("CAMERA_PASSWORD", "")

def on_error(xaddr: str, ex: Exception) -> None:
    logger.debug(f"error: {xaddr} - {ex}")

def camera_filled(camera: Camera) -> None:
    logger.debug(f"Camera Filled: {camera.hostname.name} : {camera.device_information.serial_number}")

def iterate_cameras_and_get_snapshots():
    logger.debug("Starting test...")
    adapters = find_adapters()
    for adapter in adapters:
        logger.debug(f"Found adapter: {adapter}")

    ip_address = adapters[0] if adapters else "0.0.0.0"
    cameras = discover(ip_address,
                       get_camera_credentials,
                       on_error=on_error,
                       camera_filled=camera_filled,
                       use_threads=True)
    
    for camera in cameras:
        uri = camera.profiles[0].snapshot_uri if camera.profiles else "No profiles available"
        logger.debug(f"Discovered camera: {camera.hostname.name} - {uri}")

        if camera.hostname.name:
            logger.debug(f"Found camera {camera.hostname.name}")
            response = requests.get(uri, auth=HTTPDigestAuth(camera.username, camera.password), timeout=5)
            logger.debug(f"Response status code: {response.status_code}")
            if response.status_code == 200:
                logger.debug("Successfully accessed the camera snapshot.")
                with open(f"snapshot_{camera.hostname.name}.jpg", "wb") as f:
                    f.write(response.content)
                    logger.debug(f"Snapshot saved to snapshot_{camera.hostname.name}.jpg")
            else:
                logger.error(f"Failed to access the camera snapshot. Status code: {response.status_code}")

def set_camera_resolution():
    logger.debug("Setting camera resolution...")
    # Example values for testing
    ip_address = "10.1.1.78"
    profile_token = "MediaProfile000"
    return set_camera_profile_resolution(ip_address, profile_token, 1280, 720)

async def main() -> None:
    logger.debug("Starting main function...")
    version = await get_camera_mcp_version()
    logger.debug(f"Camera MCP Version: {version}")

    result = await set_camera_resolution()
    logger.debug(f"Set camera resolution result: {result}")

if __name__ == "__main__":
    asyncio.run(main())