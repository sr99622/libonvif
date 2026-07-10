import base64
import logging
from libonvif.utils.adapters import find_adapters
from libonvif.devices.camera import Camera, discover, get_camera_by_ip, set_hostname
from mcp.server.fastmcp import FastMCP
import os
import sys
import webbrowser
import niquests as requests
from niquests.auth import HTTPDigestAuth

logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)
logger = logging.getLogger(__name__)

mcp = FastMCP("camera")

USER_AGENT = "camera-app/1.0"
VERSION = "1"

def get_camera_credentials(camera: Camera) -> None:
    camera.username = os.environ.get("CAMERA_USERNAME", "")
    camera.password = os.environ.get("CAMERA_PASSWORD", "")

def on_error(xaddr: str, ex: Exception) -> None:
    logger.debug(f"error: {xaddr} - {ex}")

def camera_filled(camera: Camera) -> None:
    logger.debug(f"Camera Filled: {camera.hostname} : {camera.device_information.serial_number}")

@mcp.tool()
async def get_camera_mcp_version() -> str:
    """
    Get the version of the camera application.

    Returns:
        The version string.
    """
    return VERSION

@mcp.tool()
async def change_camera_hostname(ip_address: str, new_hostname: str) -> str:
    """
    Change the hostname of a camera.

    Args:
        ip_address: The IP address of the camera to change.
        new_hostname: The new hostname to set.

    Returns:
        A message indicating success or failure
    """
    camera = get_camera_by_ip(ip_address, os.environ.get("CAMERA_USERNAME", ""), os.environ.get("CAMERA_PASSWORD", ""))
    if not camera:
        return f"Camera with IP {ip_address} not found."

    try:
        camera.hostname.name = new_hostname
        set_hostname(camera)
        return f"Successfully changed hostname of camera at {ip_address} to {new_hostname}."
    except Exception as e:
        logger.error(f"Failed to change hostname for camera at {ip_address}: {e}")
        return f"Failed to change hostname for camera at {ip_address}: {e}"

@mcp.tool()
async def check_camera_mcp_environment() -> str:
    """
    Collect information about the environment under which camera server is running
    
    Args:
        None

    Returns:
        A delimited string containing environment variable settings

    """

    output = []
    output.append(os.environ.get("CAMERA_USERNAME", "Empty $env:CAMERA_USERNAME"))
    output.append(os.environ.get("CAMERA_PASSWORD", "Empty $env:CAMERA_PASSWORD"))
    output.append(os.environ.get("STREAM_SERVER_IP", "Empty $env:STREAM_SERVER_IP"))
    output.append(os.environ.get("PATH", "Empty $env:PATH"))

    return "\n--\n".join(output)

@mcp.tool()
async def stream_camera(camera_device_information_serial_number: str, camera_media_profile_token: str) -> str:
    """
    Open a camera live stream in the user's default web browser.

    Args:
        camera_device_information_serial_number: The camera serial number found in the ONVIF data of the camera
                                                 that is stored in the device_information topic group.

        camera_media_profile_token: The media profile token found the ONVIF data topic profiles. The default choice
                                    should be the first profile.

    Returns:
        A message indicating success or failure
    """
    #http://10.1.1.76:8889/AMC014641NE6L35AT8/MediaProfile000
    url = f"http://{os.environ.get("STREAM_SERVER_IP")}:8889/{camera_device_information_serial_number}/{camera_media_profile_token}"
    opened = webbrowser.open(url)
    if opened:
        return f"Opened {url} in default browser."
    else:
        return f"Failed to open {url}."
    
@mcp.tool()
async def get_snapshot_image_base64_encoded(url: str) -> str:
    """
    Get a snapshot image from a camera as a base64-encoded string.

    Args:
        url: The full URL to the snapshot, e.g. "https://example.com/snapshot.jpg"

    Returns:
        The snapshot image as a base64-encoded string.
    """
    if not (url.startswith("http://") or url.startswith("https://")):
        raise ValueError(f"Refused to get snapshot from '{url}': must start with http:// or https://")

    try:
        response = requests.get(url, auth=HTTPDigestAuth(os.environ.get("CAMERA_USERNAME", ""), os.environ.get("CAMERA_PASSWORD", "")), timeout=5)
        response.raise_for_status()
        return base64.b64encode(response.content).decode('utf-8')
    except Exception as e:
        logger.error(f"Failed to get snapshot from {url}: {e}")
        return None

@mcp.tool()
async def download_snapshot_to_file(url: str, file_path: str) -> str:
    """
    Download a snapshot from a camera to a specified file path.

    Args:
        url: The full URL to the snapshot, e.g. "https://example.com/snapshot.jpg"
        file_path: The local file path where the snapshot will be saved.

    Returns:
        A message indicating success or failure.
    """
    if not (url.startswith("http://") or url.startswith("https://")):
        return f"Refused to download '{url}': must start with http:// or https://"

    try:
        response = requests.get(url, auth=HTTPDigestAuth(os.environ.get("CAMERA_USERNAME", ""), os.environ.get("CAMERA_PASSWORD", "")), timeout=5)
        response.raise_for_status()
        with open(file_path, 'wb') as f:
            f.write(response.content)
        return f"Snapshot downloaded successfully to {file_path}."
    except Exception as e:
        logger.error(f"Failed to download snapshot from {url}: {e}")
        return f"Failed to download snapshot from {url}: {e}"

@mcp.tool()
async def show_snapshot_in_browser(url: str) -> str:
    """
    Open a snapshot URL in the user's default web browser.

    Args:
        url: The full URL to open, e.g. "https://example.com"

    Returns:
        A confirmation message.
    """
    if not (url.startswith("http://") or url.startswith("https://")):
        return f"Refused to open '{url}': must start with http:// or https://"

    curl = f"{url[:7]}{os.environ.get("CAMERA_USERNAME", "")}:{os.environ.get("CAMERA_PASSWORD", "")}@{url[7:]}"
    opened = webbrowser.open(curl)
    if opened:
        return f"Opened {url} in default browser."
    else:
        return f"Failed to open {url}."
    
@mcp.tool()
async def get_camera(ip_address: str) -> str:
    """
    Get information about a camera at the specified IP address.

    Args:
        ip_address: The IP address of the camera to retrieve.

    Returns:
        A string representation of the camera's information.
    """
    camera = get_camera_by_ip(ip_address, os.environ.get("CAMERA_USERNAME", ""), os.environ.get("CAMERA_PASSWORD", ""))
    return f"{camera}"

@mcp.tool()
async def get_cameras() -> str:
    """
    Get cameras on the local network.
    
    Args:
        None

    Returns:
        A delimited string containing camera information
    """

    ip_address = "0.0.0.0"
    if sys.platform == "win32":
        ips = find_adapters()
        if len(ips):
            ip_address = ips[0]
            logger.debug(f"host ip addresses: {ips}")

    cameras = discover(ip_address,
                       get_camera_credentials,
                       on_error=on_error,
                       camera_filled=camera_filled,
                       use_threads=True)
    
    logger.debug(f"Found {len(cameras)} {"camera" if len(cameras) == 1 else "cameras"}")

    names = []
    for camera in cameras:
        names.append(f"{camera.hostname.name} : {camera.xaddr} : {camera.device_information.serial_number}")
    
    return "\n--\n".join(names)

def main():
    logger.debug("Server starting...")
    mcp.run(transport="stdio")

if __name__ == "__main__":
    main()