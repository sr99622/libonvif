import base64
import json
import logging
from importlib.metadata import version as get_installed_version
from pathlib import Path
from libonvif.utils.adapters import find_adapters
from libonvif.devices.camera import Camera, discover, get_camera_by_ip, set_hostname, \
        set_video_encoder_configuration, set_audio_encoder_configuration, camera_from_json, refresh_camera, \
        goto_preset, continuous_move, move_stop, get_local_date_and_time, set_system_date_and_time, \
        get_time_offset, set_preset, get_presets, remove_preset, create_preset_tour, modify_preset_tour, \
        remove_preset_tour, operate_preset_tour, get_preset_tours
from mcp.server.fastmcp import FastMCP, Context
from mcp.server.elicitation import AcceptedElicitation, DeclinedElicitation, CancelledElicitation
from pydantic import BaseModel
import os
import sys
import webbrowser
import niquests as requests
from niquests.auth import HTTPDigestAuth
import re


logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)
logger = logging.getLogger(__name__)

mcp = FastMCP("camera")

USER_AGENT = "camera-app/1.0"

class TripTypeResponse(BaseModel):
    value: str

def get_camera_credentials(camera: Camera) -> None:
    camera.username = os.environ.get("CAMERA_USERNAME", "")
    camera.password = os.environ.get("CAMERA_PASSWORD", "")

def on_error(xaddr: str, ex: Exception) -> None:
    logger.debug(f"error: {xaddr} - {ex}")

def camera_filled(camera: Camera) -> None:
    logger.debug(f"Camera Filled: {camera.hostname} : {camera.device_information.serial_number}")

def list_files(directory):
    """Recursively list all files in a directory."""
    for root, _, files in os.walk(directory):
        for file in files:
            yield os.path.join(root, file)

@mcp.tool()
def grep_search(pattern, directory, fileExtension=None):
    """Search for a regex pattern in files under a directory."""
    results = []

    # Validate directory
    if not os.path.isdir(directory):
        return {"error": f"Directory not found: {directory}"}

    try:
        regex = re.compile(pattern, re.IGNORECASE)
    except re.error as e:
        return {"error": f"Invalid regex: {e}"}

    try:
        for file_path in list_files(directory):
            if fileExtension and not file_path.endswith(fileExtension):
                continue

            try:
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    for line_num, line in enumerate(f, start=1):
                        if regex.search(line):
                            results.append({
                                "file": file_path,
                                "lineNum": line_num,
                                "line": line.strip()
                            })
            except (OSError, UnicodeDecodeError):
                # Skip unreadable files
                continue

    except Exception as e:
        return {"error": f"Search failed: {e}"}

    return {"matches": results}

@mcp.tool()
async def example_async_tool(context: Context) -> str:
    """
    Example async tool that asks the user a question via MCP elicitation,
    to experiment with the elicitation flow (server -> client -> user ->
    client -> server) as a building block for eventually responding to
    camera events interactively.
    """
    result = await context.elicit(
        message="What type of trip are you planning? Options: business, leisure, family, adventure",
        schema=TripTypeResponse,
    )
    if isinstance(result, AcceptedElicitation):
        return result.data.value
    elif isinstance(result, DeclinedElicitation):
        return "DECLINED"
    elif isinstance(result, CancelledElicitation):
        return "CANCELLED"
    return "INVALID RESPONSE"

@mcp.tool()
async def get_camera_mcp_version() -> str:
    """
    Get the version of the camera application, along with the version of the
    installed libonvif package it depends on.

    Returns:
        A JSON string with two fields:
            camera_mcp_version: version derived from the pyproject.toml file.
            libonvif_version: version of the installed libonvif package,
                               read via importlib.metadata.
    """

    camera_mcp_version = None
    current_file = Path(__file__)
    filename = Path(current_file.parent.parent) / "pyproject.toml"
    with open(filename, "r") as f:
        for line in f:
            if line.startswith("version"):
                camera_mcp_version = line.split("=")[1].strip().strip('"')
                logger.debug(f"Found camera_mcp version: {camera_mcp_version}")
                break

    try:
        libonvif_version = get_installed_version("libonvif")
    except Exception as e:
        logger.error(f"Failed to get libonvif version: {e}")
        libonvif_version = None

    return json.dumps({
        "camera_mcp_version": camera_mcp_version,
        "libonvif_version": libonvif_version,
    }, indent=4)

@mcp.tool()
async def set_camera_video_encoder(json_string: str, profile_token: str) -> str:
    """
    Push the video_encoder configuration for one media profile to a camera.

    Unlike a function with individual arguments per setting, this tool works
    directly on the camera's JSON representation (as returned by get_camera
    or get_cameras): edit whichever fields you want to change inside
    profiles[profile_token].video_encoder in that JSON, then pass the
    edited JSON string back in here. Every field currently set under that
    profile's video_encoder is pushed to the camera in a single ONVIF call -
    there is no need for a separate tool per field.

    Only the video_encoder node of the matching profile is read and pushed.
    Edits made anywhere else in the JSON (device_information, hostname,
    network_interfaces, other profiles, etc.) are ignored by this tool.

    Editable fields under profiles[profile_token].video_encoder, and how to
    choose a valid value for each:

        encoding
            The codec name, e.g. "H264". Must be one of the codecs the
            camera actually offers - check which of jpeg / mpeg4 / h264 are
            non-null under this same profile's video_encoder_options.

        resolution
            A string in the exact format f"{width} x {height}" (e.g.
            "1920 x 1080"). The (width, height) pair must be one of the
            entries in video_encoder_options.<codec>.resolutions_available
            for this profile's encoding (e.g. video_encoder_options.h264.
            resolutions_available when encoding is "H264"). Do not invent a
            resolution not present in that list.

        rate_control.frame_rate_limit
            Integer frames per second. Must fall within
            video_encoder_options.<codec>.frame_rate_range (min/max).

        multicast.ip_address
            A multicast IPv4 address (224.0.0.0-239.255.255.255). Leave as
            the camera's existing value unless you specifically need to
            change the multicast group.

        multicast.port
            Integer UDP port for the multicast stream.

        multicast.ttl
            Integer time-to-live (hop count) for multicast packets.

        session_timeout
            An ISO 8601 duration string, e.g. "PT60S" for 60 seconds.

        gov_length
            Integer GOP length (frames between keyframes). Must fall within
            video_encoder_options.<codec>.gov_length_range (min/max).

        profile
            The H.264 profile name, e.g. "Baseline", "Main", or "High".
            Must be one of the entries in
            video_encoder_options.h264.profiles_supported.

    Args:
        json_string: The JSON string representation of the camera, as
                     returned by get_camera or get_cameras, with the desired
                     changes already made under
                     profiles[profile_token].video_encoder.
        profile_token: The media profile token whose video_encoder should be
                       pushed to the camera.

    Returns:
        A message indicating success or failure
    """
    try:
        camera = camera_from_json(json_string)
    except Exception as e:
        logger.error(f"Failed to parse camera JSON: {e}")
        return f"Failed to parse camera JSON: {e}"

    try:
        camera.errors = None

        for profile in camera.profiles:
            if profile.token == profile_token:
                set_video_encoder_configuration(camera, profile.video_encoder)
                if camera.errors:
                    raise Exception(f"Camera returned errors: {camera.errors}")
                return f"Successfully set video encoder configuration for camera at {camera.xaddr}, profile {profile_token}."

        return f"Profile {profile_token} not found on camera at {camera.xaddr}."

    except Exception as e:
        logger.error(f"Failed to set video encoder configuration for camera at {camera.xaddr}: {e}")
        return f"Failed to set video encoder configuration for camera at {camera.xaddr}: {e}"

@mcp.tool()
async def set_camera_audio_encoder(json_string: str, profile_token: str) -> str:
    """
    Push the audio_encoder configuration for one media profile to a camera.

    Like set_camera_video_encoder, this tool works directly on the camera's
    JSON representation (as returned by get_camera or get_cameras): edit
    whichever fields you want to change inside
    profiles[profile_token].audio_encoder in that JSON, then pass the
    edited JSON string back in here. Every field currently set under that
    profile's audio_encoder is pushed to the camera in a single ONVIF call.

    Only the audio_encoder node of the matching profile is read and pushed.
    Edits made anywhere else in the JSON (device_information, hostname,
    video_encoder, other profiles, etc.) are ignored by this tool.

    Editable fields under profiles[profile_token].audio_encoder, and how to
    choose a valid value for each:

        encoding
            The codec name, e.g. "G711" or "AAC". Must match one of the
            entries in this same profile's audio_encoder_options list (each
            entry there has its own .encoding).

        bitrate
            Integer bitrate. Must be one of the values in
            audio_encoder_options[i].bitrate_list for the entry whose
            .encoding matches this encoder's encoding.

            Note: on at least some hardware (observed on an Amcrest G711
            implementation), bitrate and sample_rate appear to be coupled -
            changing bitrate alone was silently ignored by the camera, while
            changing sample_rate caused bitrate to change along with it.
            If you need a specific bitrate, try setting sample_rate to the
            value that pairs with it and verify both fields with a fresh
            get_camera call afterward, since a "success" response does not
            guarantee every field you set was actually applied.

        sample_rate
            Integer sample rate. Must be one of the values in
            audio_encoder_options[i].sample_rate_list for the entry whose
            .encoding matches this encoder's encoding.

            See the note under bitrate above - on some hardware this is the
            field that actually drives the change, with bitrate following
            it rather than being independently settable.

        multicast.ip_address
            A multicast IPv4 address (224.0.0.0-239.255.255.255). Leave as
            the camera's existing value unless you specifically need to
            change the multicast group.

        multicast.port
            Integer UDP port for the multicast stream.

        multicast.ttl
            Integer time-to-live (hop count) for multicast packets.

        session_timeout
            An ISO 8601 duration string, e.g. "PT30S" for 30 seconds.

    Args:
        json_string: The JSON string representation of the camera, as
                     returned by get_camera or get_cameras, with the desired
                     changes already made under
                     profiles[profile_token].audio_encoder.
        profile_token: The media profile token whose audio_encoder should be
                       pushed to the camera.

    Returns:
        A message indicating success or failure
    """
    try:
        camera = camera_from_json(json_string)
    except Exception as e:
        logger.error(f"Failed to parse camera JSON: {e}")
        return f"Failed to parse camera JSON: {e}"

    try:
        camera.errors = None

        for profile in camera.profiles:
            if profile.token == profile_token:
                set_audio_encoder_configuration(camera, profile.audio_encoder)
                if camera.errors:
                    raise Exception(f"Camera returned errors: {camera.errors}")
                return f"Successfully set audio encoder configuration for camera at {camera.xaddr}, profile {profile_token}."

        return f"Profile {profile_token} not found on camera at {camera.xaddr}."

    except Exception as e:
        logger.error(f"Failed to set audio encoder configuration for camera at {camera.xaddr}: {e}")
        return f"Failed to set audio encoder configuration for camera at {camera.xaddr}: {e}"

@mcp.tool()
async def goto_camera_preset(json_string: str, profile_token: str, preset_token: str) -> str:
    """
    Move a PTZ camera to one of its stored presets.

    Presets are found at camera.ptz.presets in the camera's JSON
    representation (as returned by get_camera/get_cameras) - a list of
    PTZPreset entries, each with a token and a (often blank) name. Find
    the preset you want by matching its token or name in that list, then
    pass its token here as preset_token.

    profile_token should almost always be the camera's main media profile
    token - typically profiles[0].token, e.g. "MediaProfile000" - since
    PTZ presets are defined per-profile and the main profile is where
    they're normally stored.

    This tool only sends the move command; it does not wait for the
    camera to finish moving or confirm it arrived. To check on that,
    call get_camera again afterward and look at ptz.status.pan_tilt_status
    and ptz.status.zoom_status ("IDLE" once the move has completed) and
    ptz.status.position for the camera's current position.

    Args:
        json_string: The JSON string representation of the camera, as
                     returned by get_camera or get_cameras.
        profile_token: The media profile token to command (see above).
        preset_token: The token of the preset to move to, from
                      camera.ptz.presets in the same JSON.

    Returns:
        A message indicating success or failure
    """
    try:
        camera = camera_from_json(json_string)
    except Exception as e:
        logger.error(f"Failed to parse camera JSON: {e}")
        return f"Failed to parse camera JSON: {e}"

    if not camera.ptz or not camera.ptz.presets:
        return f"Camera at {camera.xaddr} has no PTZ presets available."

    preset = None
    for candidate in camera.ptz.presets:
        if candidate.token == preset_token:
            preset = candidate
            break

    if not preset:
        return f"Preset {preset_token} not found on camera at {camera.xaddr}."

    try:
        camera.errors = None
        goto_preset(camera, profile_token, preset)
        if camera.errors:
            raise Exception(f"Camera returned errors: {camera.errors}")
        return f"Successfully moved camera at {camera.xaddr} to preset {preset_token}."
    except Exception as e:
        logger.error(f"Failed to move camera at {camera.xaddr} to preset {preset_token}: {e}")
        return f"Failed to move camera at {camera.xaddr} to preset {preset_token}: {e}"

@mcp.tool()
async def set_camera_preset(json_string: str, profile_token: str, preset_token: str = None, preset_name: str = None) -> str:
    """
    Create a new PTZ preset, or overwrite an existing one with the camera's
    current position.

    Two modes, based on whether preset_token is supplied:

    - preset_token omitted (create mode): the camera creates a brand new
      preset at its current position. Cameras support a limited number of
      presets - check how many already exist in camera.ptz.presets before
      creating another, in case the camera silently rejects it once full.
      If preset_name is given, the new preset is created first, then
      renamed in a second call - the underlying ONVIF operation can't
      assign a name to a preset that doesn't have a token yet, so this
      tool creates it unnamed, determines the token the camera just
      assigned, then renames it. The camera doesn't move between these
      two calls, so the rename call safely re-saves the same position.

    - preset_token supplied (overwrite mode): the preset matching that
      token in camera.ptz.presets has its position overwritten to the
      camera's CURRENT position - not restored to wherever it used to
      point. If you only want to rename an existing preset without moving
      it, first call goto_camera_preset to move the camera back to that
      preset's own position, THEN call this tool - otherwise the preset's
      saved position will be silently replaced with wherever the camera
      happens to be sitting right now. Pass preset_name to also update the
      preset's stored name at the same time.

    Args:
        json_string: The JSON string representation of the camera, as
                     returned by get_camera or get_cameras.
        profile_token: The media profile token to command (almost always
                       the main profile, e.g. profiles[0].token).
        preset_token: Token of an existing preset to overwrite (from
                      camera.ptz.presets). Omit to create a new preset
                      instead.
        preset_name: Optional name to assign to the preset (new or
                     existing).

    Returns:
        A message indicating success or failure. On successful creation,
        includes the newly assigned preset token.
    """
    try:
        camera = camera_from_json(json_string)
    except Exception as e:
        logger.error(f"Failed to parse camera JSON: {e}")
        return f"Failed to parse camera JSON: {e}"

    try:
        camera.errors = None

        if preset_token:
            preset = None
            for candidate in (camera.ptz.presets if camera.ptz else []):
                if candidate.token == preset_token:
                    preset = candidate
                    break
            if not preset:
                return f"Preset {preset_token} not found on camera at {camera.xaddr}."
            if preset_name is not None:
                preset.name = preset_name
            set_preset(camera, profile_token, preset)
            if camera.errors:
                raise Exception(f"Camera returned errors: {camera.errors}")
            return f"Successfully overwrote preset {preset_token} on camera at {camera.xaddr} with its current position."

        # create mode
        existing_tokens = {p.token for p in (camera.ptz.presets if camera.ptz else [])}
        set_preset(camera, profile_token)
        if camera.errors:
            raise Exception(f"Camera returned errors: {camera.errors}")

        get_presets(camera, profile_token)
        if camera.errors:
            raise Exception(f"Camera returned errors while refreshing presets: {camera.errors}")

        new_tokens = [p.token for p in camera.ptz.presets if p.token not in existing_tokens]
        if not new_tokens:
            return f"Preset created on camera at {camera.xaddr}, but could not determine its new token from the refreshed preset list."
        new_token = new_tokens[0]

        if preset_name is not None:
            new_preset = None
            for candidate in camera.ptz.presets:
                if candidate.token == new_token:
                    new_preset = candidate
                    break
            new_preset.name = preset_name
            set_preset(camera, profile_token, new_preset)
            if camera.errors:
                raise Exception(f"Preset {new_token} created, but failed to set its name: {camera.errors}")

        name_note = f" named '{preset_name}'" if preset_name else ""
        return f"Successfully created new preset {new_token}{name_note} on camera at {camera.xaddr}."

    except Exception as e:
        logger.error(f"Failed to set preset for camera at {camera.xaddr}: {e}")
        return f"Failed to set preset for camera at {camera.xaddr}: {e}"

@mcp.tool()
async def remove_camera_preset(json_string: str, profile_token: str, preset_token: str) -> str:
    """
    Permanently delete a PTZ preset from a camera.

    This removes the preset entirely - it is not the same as clearing or
    resetting a preset's position, and it cannot be undone from this
    tool. If you want to reuse a preset's token/slot for a different
    position instead of deleting it outright, use set_camera_preset in
    overwrite mode instead.

    Args:
        json_string: The JSON string representation of the camera, as
                     returned by get_camera or get_cameras.
        profile_token: The media profile token to command (almost always
                       the main profile, e.g. profiles[0].token).
        preset_token: Token of the preset to remove, from
                      camera.ptz.presets in the same JSON.

    Returns:
        A message indicating success or failure
    """
    try:
        camera = camera_from_json(json_string)
    except Exception as e:
        logger.error(f"Failed to parse camera JSON: {e}")
        return f"Failed to parse camera JSON: {e}"

    preset = None
    for candidate in (camera.ptz.presets if camera.ptz else []):
        if candidate.token == preset_token:
            preset = candidate
            break
    if not preset:
        return f"Preset {preset_token} not found on camera at {camera.xaddr}."

    try:
        camera.errors = None
        remove_preset(camera, profile_token, preset)
        if camera.errors:
            raise Exception(f"Camera returned errors: {camera.errors}")
        return f"Successfully removed preset {preset_token} from camera at {camera.xaddr}."
    except Exception as e:
        logger.error(f"Failed to remove preset {preset_token} from camera at {camera.xaddr}: {e}")
        return f"Failed to remove preset {preset_token} from camera at {camera.xaddr}: {e}"

@mcp.tool()
async def create_camera_preset_tour(json_string: str, profile_token: str, tour_name: str = None) -> str:
    """
    Create a new, empty PTZ preset tour on a camera.

    The underlying ONVIF CreatePresetTour operation has no name field, so
    if tour_name is given, this tool creates the tour first, determines
    the token the camera just assigned (by diffing camera.ptz.tours
    before/after), then applies the name in a follow-up call - the tour
    has no spots yet either way, so this is a safe two-step sequence, the
    same pattern used by set_camera_preset for naming a newly-created
    preset.

    Once created, use set_camera_preset_tour to populate it with spots
    (preset token + stay time pairs), then start_camera_preset_tour to
    run it.

    Args:
        json_string: The JSON string representation of the camera, as
                     returned by get_camera or get_cameras.
        profile_token: The media profile token to command (almost always
                       the main profile, e.g. profiles[0].token).
        tour_name: Optional name to assign to the new tour.

    Returns:
        A message indicating success or failure. On success, includes the
        newly assigned tour token.
    """
    try:
        camera = camera_from_json(json_string)
    except Exception as e:
        logger.error(f"Failed to parse camera JSON: {e}")
        return f"Failed to parse camera JSON: {e}"

    try:
        camera.errors = None
        existing_tokens = {t.token for t in (camera.ptz.tours if camera.ptz else [])}

        create_preset_tour(camera, profile_token)
        if camera.errors:
            raise Exception(f"Camera returned errors: {camera.errors}")

        get_preset_tours(camera, profile_token)
        if camera.errors:
            raise Exception(f"Camera returned errors while refreshing tours: {camera.errors}")

        new_tokens = [t.token for t in camera.ptz.tours if t.token not in existing_tokens]
        if not new_tokens:
            return f"Tour created on camera at {camera.xaddr}, but could not determine its new token from the refreshed tour list."
        new_token = new_tokens[0]

        if tour_name is not None:
            new_tour = None
            for candidate in camera.ptz.tours:
                if candidate.token == new_token:
                    new_tour = candidate
                    break
            new_tour.name = tour_name
            modify_preset_tour(camera, profile_token, new_tour)
            if camera.errors:
                raise Exception(f"Tour {new_token} created, but failed to set its name: {camera.errors}")

        name_note = f" named '{tour_name}'" if tour_name else ""
        return f"Successfully created new preset tour {new_token}{name_note} on camera at {camera.xaddr}."

    except Exception as e:
        logger.error(f"Failed to create preset tour for camera at {camera.xaddr}: {e}")
        return f"Failed to create preset tour for camera at {camera.xaddr}: {e}"

@mcp.tool()
async def set_camera_preset_tour(json_string: str, profile_token: str, tour_token: str) -> str:
    """
    Push a PTZ preset tour's full configuration - name, auto_start, and
    spots - to a camera.

    Like set_camera_video_encoder, this tool works directly on the
    camera's JSON representation (as returned by get_camera or
    get_cameras): edit whichever fields you want to change inside
    ptz.tours[tour_token] in that JSON, then pass the edited JSON string
    back in here. Every field currently set under that tour is pushed to
    the camera in a single ONVIF call - this replaces the tour's entire
    spot list with whatever is currently there, rather than adding to or
    removing from it incrementally, so to add or remove a spot, edit the
    full list to the desired end result before calling this.

    Editable fields under ptz.tours[tour_token]:

        name
            Display name for the tour.

        auto_start
            Boolean. Whether the tour starts automatically under the
            camera's own configured starting condition (see
            ptz.tour_options.starting_condition), rather than needing to
            be started manually via start_camera_preset_tour.

        spots
            A list of {preset_token, stay_time} pairs, in the order the
            tour should visit them. preset_token must match a real
            preset in camera.ptz.presets. stay_time is an ISO 8601
            duration string (e.g. "PT5S" for 5 seconds) - check
            ptz.tour_options.tour_spot.stay_time for the camera's allowed
            min/max range, and ptz.tour_options.tour_spot.preset_tokens
            for which presets are eligible to be used in a tour at all
            (not every stored preset may qualify).

    Args:
        json_string: The JSON string representation of the camera, as
                     returned by get_camera or get_cameras, with the
                     desired changes already made under
                     ptz.tours[tour_token].
        profile_token: The media profile token to command (almost always
                       the main profile, e.g. profiles[0].token).
        tour_token: The token of the tour to update, from ptz.tours in
                    the same JSON.

    Returns:
        A message indicating success or failure
    """
    try:
        camera = camera_from_json(json_string)
    except Exception as e:
        logger.error(f"Failed to parse camera JSON: {e}")
        return f"Failed to parse camera JSON: {e}"

    tour = None
    for candidate in (camera.ptz.tours if camera.ptz else []):
        if candidate.token == tour_token:
            tour = candidate
            break
    if not tour:
        return f"Tour {tour_token} not found on camera at {camera.xaddr}."

    try:
        camera.errors = None
        modify_preset_tour(camera, profile_token, tour)
        if camera.errors:
            raise Exception(f"Camera returned errors: {camera.errors}")
        return f"Successfully updated preset tour {tour_token} on camera at {camera.xaddr}."
    except Exception as e:
        logger.error(f"Failed to update preset tour {tour_token} on camera at {camera.xaddr}: {e}")
        return f"Failed to update preset tour {tour_token} on camera at {camera.xaddr}: {e}"

@mcp.tool()
async def remove_camera_preset_tour(json_string: str, profile_token: str, tour_token: str) -> str:
    """
    Permanently delete a PTZ preset tour from a camera.

    This removes the tour entirely - it does not affect the individual
    presets used in its spots, only the tour itself - and cannot be
    undone from this tool.

    Args:
        json_string: The JSON string representation of the camera, as
                     returned by get_camera or get_cameras.
        profile_token: The media profile token to command (almost always
                       the main profile, e.g. profiles[0].token).
        tour_token: Token of the tour to remove, from ptz.tours in the
                    same JSON.

    Returns:
        A message indicating success or failure
    """
    try:
        camera = camera_from_json(json_string)
    except Exception as e:
        logger.error(f"Failed to parse camera JSON: {e}")
        return f"Failed to parse camera JSON: {e}"

    tour = None
    for candidate in (camera.ptz.tours if camera.ptz else []):
        if candidate.token == tour_token:
            tour = candidate
            break
    if not tour:
        return f"Tour {tour_token} not found on camera at {camera.xaddr}."

    try:
        camera.errors = None
        remove_preset_tour(camera, profile_token, tour)
        if camera.errors:
            raise Exception(f"Camera returned errors: {camera.errors}")
        return f"Successfully removed preset tour {tour_token} from camera at {camera.xaddr}."
    except Exception as e:
        logger.error(f"Failed to remove preset tour {tour_token} from camera at {camera.xaddr}: {e}")
        return f"Failed to remove preset tour {tour_token} from camera at {camera.xaddr}: {e}"

@mcp.tool()
async def start_camera_preset_tour(json_string: str, profile_token: str, tour_token: str) -> str:
    """
    Start running a PTZ preset tour on a camera.

    The camera begins moving through the tour's spots in order, pausing
    at each for its configured stay_time, looping continuously until
    stop_camera_preset_tour is called. This does not wait for the tour to
    complete (it never does, on its own) or confirm it started - check
    ptz.tours[tour_token].status.state via a fresh get_camera call to see
    its reported state (e.g. "Idle" vs actively touring).

    Args:
        json_string: The JSON string representation of the camera, as
                     returned by get_camera or get_cameras.
        profile_token: The media profile token to command (almost always
                       the main profile, e.g. profiles[0].token).
        tour_token: Token of the tour to start, from ptz.tours in the
                    same JSON.

    Returns:
        A message indicating success or failure
    """
    try:
        camera = camera_from_json(json_string)
    except Exception as e:
        logger.error(f"Failed to parse camera JSON: {e}")
        return f"Failed to parse camera JSON: {e}"

    tour = None
    for candidate in (camera.ptz.tours if camera.ptz else []):
        if candidate.token == tour_token:
            tour = candidate
            break
    if not tour:
        return f"Tour {tour_token} not found on camera at {camera.xaddr}."

    try:
        camera.errors = None
        operate_preset_tour(camera, profile_token, tour, "Start")
        if camera.errors:
            raise Exception(f"Camera returned errors: {camera.errors}")
        return f"Successfully started preset tour {tour_token} on camera at {camera.xaddr}."
    except Exception as e:
        logger.error(f"Failed to start preset tour {tour_token} on camera at {camera.xaddr}: {e}")
        return f"Failed to start preset tour {tour_token} on camera at {camera.xaddr}: {e}"

@mcp.tool()
async def stop_camera_preset_tour(json_string: str, profile_token: str, tour_token: str) -> str:
    """
    Stop a running PTZ preset tour on a camera.

    Args:
        json_string: The JSON string representation of the camera, as
                     returned by get_camera or get_cameras.
        profile_token: The media profile token to command (almost always
                       the main profile, e.g. profiles[0].token).
        tour_token: Token of the tour to stop, from ptz.tours in the
                    same JSON.

    Returns:
        A message indicating success or failure
    """
    try:
        camera = camera_from_json(json_string)
    except Exception as e:
        logger.error(f"Failed to parse camera JSON: {e}")
        return f"Failed to parse camera JSON: {e}"

    tour = None
    for candidate in (camera.ptz.tours if camera.ptz else []):
        if candidate.token == tour_token:
            tour = candidate
            break
    if not tour:
        return f"Tour {tour_token} not found on camera at {camera.xaddr}."

    try:
        camera.errors = None
        operate_preset_tour(camera, profile_token, tour, "Stop")
        if camera.errors:
            raise Exception(f"Camera returned errors: {camera.errors}")
        return f"Successfully stopped preset tour {tour_token} on camera at {camera.xaddr}."
    except Exception as e:
        logger.error(f"Failed to stop preset tour {tour_token} on camera at {camera.xaddr}: {e}")
        return f"Failed to stop preset tour {tour_token} on camera at {camera.xaddr}: {e}"

@mcp.tool()
async def pan_tilt_camera(json_string: str, profile_token: str, x: float, y: float) -> str:
    """
    Start a continuous pan/tilt move on a PTZ camera.

    x and y are normalized velocities in the range -1.0 to 1.0 (0.0 means
    no motion on that axis): positive x pans right, negative x pans left;
    positive y tilts up, negative y tilts down. These are velocities, not
    positions - the camera keeps moving in that direction at that speed
    until stop_camera_pan_tilt is called.

    This does not stop on its own except at the camera's physical pan/tilt
    limits - most PTZ hardware halts at its mechanical range ends, so
    forgetting to stop is not unsafe, but the camera will simply drift to
    whichever limit it's heading toward and park there rather than stopping
    at a precise point. Call stop_camera_pan_tilt to halt motion exactly
    where you want it, or check ptz.status.position via get_camera to see
    where it ended up.

    This is pan/tilt only - it has no effect on zoom. Use zoom_camera
    separately for zoom; a camera can only perform one of pan/tilt or zoom
    at a time.

    Args:
        json_string: The JSON string representation of the camera, as
                     returned by get_camera or get_cameras.
        profile_token: The media profile token to command (almost always
                       the main profile, e.g. profiles[0].token).
        x: Pan velocity, -1.0 (left) to 1.0 (right). 0.0 for no pan.
        y: Tilt velocity, -1.0 (down) to 1.0 (up). 0.0 for no tilt.

    Returns:
        A message indicating success or failure
    """
    try:
        camera = camera_from_json(json_string)
    except Exception as e:
        logger.error(f"Failed to parse camera JSON: {e}")
        return f"Failed to parse camera JSON: {e}"

    try:
        camera.errors = None
        continuous_move(camera, profile_token, x, y, 0)
        if camera.errors:
            raise Exception(f"Camera returned errors: {camera.errors}")
        return f"Successfully started pan/tilt move on camera at {camera.xaddr} (x={x}, y={y})."
    except Exception as e:
        logger.error(f"Failed to start pan/tilt move on camera at {camera.xaddr}: {e}")
        return f"Failed to start pan/tilt move on camera at {camera.xaddr}: {e}"

@mcp.tool()
async def zoom_camera(json_string: str, profile_token: str, z: float) -> str:
    """
    Start a continuous zoom move on a PTZ camera.

    z is a normalized velocity in the range -1.0 to 1.0, excluding 0.0:
    positive zooms in (telephoto), negative zooms out (wide). This is a
    velocity, not a position - the camera keeps zooming at that speed
    until stop_camera_zoom is called. z=0.0 is rejected here rather than
    silently doing nothing; use stop_camera_zoom if you want to halt an
    in-progress zoom.

    This does not stop on its own except at the camera's physical zoom
    limits (fully wide or fully telephoto) - most PTZ hardware halts
    there, so forgetting to stop is not unsafe, but the camera will simply
    zoom to whichever limit it's heading toward and stop there rather than
    at a precise point. Call stop_camera_zoom to halt zoom exactly where
    you want it, or check ptz.status.position.zoom via get_camera to see
    where it ended up.

    This is zoom only - it has no effect on pan/tilt. Use pan_tilt_camera
    separately for pan/tilt; a camera can only perform one of pan/tilt or
    zoom at a time.

    Args:
        json_string: The JSON string representation of the camera, as
                     returned by get_camera or get_cameras.
        profile_token: The media profile token to command (almost always
                       the main profile, e.g. profiles[0].token).
        z: Zoom velocity, -1.0 (zoom out) to 1.0 (zoom in). Must not be 0.0.

    Returns:
        A message indicating success or failure
    """
    if z == 0:
        return "z must not be 0.0 - to stop an in-progress zoom, call stop_camera_zoom instead."

    try:
        camera = camera_from_json(json_string)
    except Exception as e:
        logger.error(f"Failed to parse camera JSON: {e}")
        return f"Failed to parse camera JSON: {e}"

    try:
        camera.errors = None
        continuous_move(camera, profile_token, 0, 0, z)
        if camera.errors:
            raise Exception(f"Camera returned errors: {camera.errors}")
        return f"Successfully started zoom move on camera at {camera.xaddr} (z={z})."
    except Exception as e:
        logger.error(f"Failed to start zoom move on camera at {camera.xaddr}: {e}")
        return f"Failed to start zoom move on camera at {camera.xaddr}: {e}"

@mcp.tool()
async def stop_camera_pan_tilt(json_string: str, profile_token: str) -> str:
    """
    Stop an in-progress continuous pan/tilt move started by pan_tilt_camera.

    Has no effect on zoom - use stop_camera_zoom to stop a zoom move. If no
    pan/tilt move is currently in progress, this is a harmless no-op on
    most cameras.

    Args:
        json_string: The JSON string representation of the camera, as
                     returned by get_camera or get_cameras.
        profile_token: The media profile token to command (should match
                       whatever was used in the pan_tilt_camera call).

    Returns:
        A message indicating success or failure
    """
    try:
        camera = camera_from_json(json_string)
    except Exception as e:
        logger.error(f"Failed to parse camera JSON: {e}")
        return f"Failed to parse camera JSON: {e}"

    try:
        camera.errors = None
        move_stop(camera, profile_token, is_zoom=False)
        if camera.errors:
            raise Exception(f"Camera returned errors: {camera.errors}")
        return f"Successfully stopped pan/tilt move on camera at {camera.xaddr}."
    except Exception as e:
        logger.error(f"Failed to stop pan/tilt move on camera at {camera.xaddr}: {e}")
        return f"Failed to stop pan/tilt move on camera at {camera.xaddr}: {e}"

@mcp.tool()
async def stop_camera_zoom(json_string: str, profile_token: str) -> str:
    """
    Stop an in-progress continuous zoom move started by zoom_camera.

    Has no effect on pan/tilt - use stop_camera_pan_tilt to stop a pan/tilt
    move. If no zoom move is currently in progress, this is a harmless
    no-op on most cameras.

    Args:
        json_string: The JSON string representation of the camera, as
                     returned by get_camera or get_cameras.
        profile_token: The media profile token to command (should match
                       whatever was used in the zoom_camera call).

    Returns:
        A message indicating success or failure
    """
    try:
        camera = camera_from_json(json_string)
    except Exception as e:
        logger.error(f"Failed to parse camera JSON: {e}")
        return f"Failed to parse camera JSON: {e}"

    try:
        camera.errors = None
        move_stop(camera, profile_token, is_zoom=True)
        if camera.errors:
            raise Exception(f"Camera returned errors: {camera.errors}")
        return f"Successfully stopped zoom move on camera at {camera.xaddr}."
    except Exception as e:
        logger.error(f"Failed to stop zoom move on camera at {camera.xaddr}: {e}")
        return f"Failed to stop zoom move on camera at {camera.xaddr}: {e}"

@mcp.tool()
async def change_camera_hostname(json_string: str, new_hostname: str) -> str:
    """
    Change the hostname of a camera.

    Args:
        json_string: The JSON string representation of the camera, as returned by get_camera or get_cameras.
        new_hostname: The new hostname to set.

    Returns:
        A message indicating success or failure
    """
    try:
        camera = camera_from_json(json_string)
    except Exception as e:
        logger.error(f"Failed to parse camera JSON: {e}")
        return f"Failed to parse camera JSON: {e}"

    try:
        camera.hostname.name = new_hostname
        camera.errors = None
        set_hostname(camera)
        if camera.errors:
            raise Exception(f"Camera returned errors: {camera.errors}")
        return f"Successfully changed hostname of camera at {camera.xaddr} to {new_hostname}."
    except Exception as e:
        logger.error(f"Failed to change hostname for camera at {camera.xaddr}: {e}")
        return f"Failed to change hostname for camera at {camera.xaddr}: {e}"

@mcp.tool()
async def sync_camera_time(json_string: str) -> str:
    """
    Synchronize a camera's clock to this machine's current local time.

    Useful for correcting a camera whose internal clock has drifted or
    reset (e.g. after a power loss reverting it to an epoch default like
    2000-01-01), which otherwise produces confusing timestamps on
    snapshots and event data.

    Builds a SystemDateAndTime from this machine's current local and UTC
    time (matching the timezone-offset format ONVIF expects), pushes it
    to the camera, then re-queries the camera's own reported time to
    recalculate time_offset - the difference in seconds between the
    camera's clock and this machine's, which should end up close to zero
    once synced.

    Args:
        json_string: The JSON string representation of the camera, as
                     returned by get_camera or get_cameras.

    Returns:
        A message indicating success or failure, including the resulting
        time_offset in seconds if successful.
    """
    try:
        camera = camera_from_json(json_string)
    except Exception as e:
        logger.error(f"Failed to parse camera JSON: {e}")
        return f"Failed to parse camera JSON: {e}"

    try:
        camera.errors = None
        sdt = get_local_date_and_time()
        set_system_date_and_time(camera, sdt)
        if camera.errors:
            raise Exception(f"Camera returned errors: {camera.errors}")
        get_time_offset(camera)
        return f"Successfully synchronized time for camera at {camera.xaddr}. time_offset is now {camera.time_offset} seconds."
    except Exception as e:
        logger.error(f"Failed to sync time for camera at {camera.xaddr}: {e}")
        return f"Failed to sync time for camera at {camera.xaddr}: {e}"

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
    return camera.to_json()

@mcp.tool()
async def update_camera_data(json_string: str) -> str:
    """
    Re-query a camera fresh, using the xaddr and credentials currently set
    in the given camera JSON.

    Use this after editing username or password in the JSON returned by
    get_camera/get_cameras - for example, to try different credentials
    against a camera that failed authorization the first time. The edited
    credentials are what get used for the fresh query, not whatever was
    originally used. Any other edits made elsewhere in the JSON are
    ignored, since this re-runs the full query from scratch rather than
    patching the existing data - the returned camera reflects the device's
    actual current state, not your edits (aside from username/password,
    which control how the query is authorized).

    Do not edit xaddr. It is the camera's own self-reported device service
    address, discovered without authorization, and functions as the
    camera's network identity rather than a configurable setting. Changing
    it points this tool at a different device entirely rather than
    re-querying the same camera.

    Args:
        json_string: The JSON string representation of the camera, as
                     returned by get_camera or get_cameras, with the
                     desired username/password already edited.

    Returns:
        The freshly queried camera as a JSON string, or an error message
        if the JSON could not be parsed or the query itself failed (e.g.
        the credentials are still not authorized).
    """
    try:
        camera = camera_from_json(json_string)
    except Exception as e:
        logger.error(f"Failed to parse camera JSON: {e}")
        return f"Failed to parse camera JSON: {e}"

    try:
        refreshed = refresh_camera(camera)
        return refreshed.to_json()
    except Exception as e:
        logger.error(f"Failed to refresh camera at {camera.xaddr}: {e}")
        return f"Failed to refresh camera at {camera.xaddr}: {e}"

@mcp.tool()
async def get_cameras() -> str:
    """
    Get cameras on the local network.
    
    Args:
        None

    Returns:
        A delimited string containing full camera information in json format 
        for each camera found on the local network. Each camera's information 
        is separated by "\n--\n".
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
        names.append(camera.to_json())

    return "\n--\n".join(names)

def main():
    logger.debug("Server starting...")
    mcp.run(transport="stdio")

if __name__ == "__main__":
    main()