from typing import Any, get_args, get_origin, Union, get_type_hints
import types
import re

_INDEX_RE = re.compile(r"^\[(\d+)\]$")

EDITABLE_FIELDS = [
    "network_gateway", 
    "hostname.from_dhcp", 
    "hostname.name",
    "dns.from_dhcp", 
    "dns.dns_manual",
    "ntp.from_dhcp",
    "ntp.ntp_manual",
    "network_interfaces.[*].ipv4.dhcp",
    "network_interfaces.[*].ipv4.manual",
    "profiles.[*].imaging_settings.brightness",
    "profiles.[*].imaging_settings.color_saturation",
    "profiles.[*].imaging_settings.contrast",
    "profiles.[*].imaging_settings.sharpness",
    "profiles.[*].imaging_settings.ir_cut_filter",
    "profiles.[*].audio_encoder.encoding",
    "profiles.[*].audio_encoder.bitrate",
    "profiles.[*].audio_encoder.sample_rate",
    "profiles.[*].audio_encoder.session_timeout",
    "profiles.[*].audio_encoder.multicast.port",
    "profiles.[*].audio_encoder.multicast.ttl",
    "profiles.[*].audio_encoder.multicast.ip_address",
    "profiles.[*].video_encoder.resolution",
    "profiles.[*].video_encoder.session_timeout",
    "profiles.[*].video_encoder.encoding",
    "profiles.[*].video_encoder.profile",
    "profiles.[*].video_encoder.gov_length",
    "profiles.[*].video_encoder.quality",
    "profiles.[*].video_encoder.multicast.port",
    "profiles.[*].video_encoder.multicast.ttl",
    "profiles.[*].video_encoder.multicast.ip_address",
    "profiles.[*].video_encoder.rate_control.frame_rate_limit",
    "profiles.[*].video_encoder.rate_control.encoding_interval",
    "profiles.[*].video_encoder.rate_control.bitrate_limit",
    "ptz.presets.[*].name",
    "ptz.tours.[*].name",
    "ptz.tours.[*].auto_start",
    "ptz.tours.[*].spots.[*].stay_time",
    "ptz.tours.[*].spots.[*].preset_token",
    "relay_outputs.[*].properties.mode",
    "relay_outputs.[*].properties.delay_time",
    "relay_outputs.[*].properties.idle_state",
    "system_date_and_time.date_time_type",
    "system_date_and_time.daylight_savings",
    "system_date_and_time.time_zone.tz",
]

UNUSED_FIELDS = [
    "audio_decoder", 
    "audio_decoder_options", 
    "audio_outputs",
    "capabilities.events.service_capabilities.persistent_notification_storage",
    "capabilities.events.service_capabilities.event_broker_protocols",
    "capabilities.events.service_capabilities.max_event_brokers",
    "capabilities.events.service_capabilities.metadata_over_mqtt",
    "event_properties.fixed_topic_set",
    "event_properties.producer_properties_filter_dialect",
    "event_properties.topic_expression_dialect",
    "capabilities.telex",
    "relay_outputs",
    "ptz",
    "audio_out",
]

HIDDEN_FIELDS = [
    "subscription_references",
    "username",
    "password",
    "on_error",
]

def normalize_fqn(fqn: str) -> str:
    return re.sub(r"\[\d+\]", "[*]", fqn)

def is_editable_field(fqn: str) -> bool:
    return normalize_fqn(fqn) in EDITABLE_FIELDS

def join_fqn(parent_fqn: str | None, field_name: str) -> str:
    if parent_fqn:
        return f"{parent_fqn}.{field_name}"
    return field_name

def unwrap_optional(field_type: Any) -> Any:
    origin = get_origin(field_type)

    if origin is Union or origin is types.UnionType:
        args = [arg for arg in get_args(field_type) if arg is not type(None)]
        if len(args) == 1:
            return args[0]

    return field_type

def parse_ip_string_list(text: str) -> list[str]:
    text = text.strip()

    if not text:
        return []

    if text.startswith("[") and text.endswith("]"):
        text = text[1:-1]

    result: list[str] = []

    for raw in text.split(","):
        item = raw.strip()
        item = item.strip('"').strip("'").strip()

        if not item:
            continue

        result.append(item)

    return result

def convert_string_value(value: str, field_type: Any) -> Any:
    field_type = unwrap_optional(field_type)

    origin = get_origin(field_type)
    args = get_args(field_type)

    if origin is list:
        item_type = args[0] if args else str

        if item_type is str:
            return parse_ip_string_list(value)

        # generic fallback for other list types
        return [
            convert_string_value(item.strip(), item_type)
            for item in value.split(",")
            if item.strip()
        ]

    if field_type is str:
        return value

    if field_type is int:
        return int(value)

    if field_type is float:
        return float(value)

    if field_type is bool:
        value = value.strip().lower()

        if value in {"true", "1", "yes", "y", "on"}:
            return True

        if value in {"false", "0", "no", "n", "off"}:
            return False

        raise ValueError(f"Invalid bool value: {value}")

    return value


def resolve_fqn_owner(root: object, fqn: str) -> tuple[object, str, Any, list[int]]:
    parts = fqn.split(".")
    owner = root
    indices: list[int] = []
    for part in parts[:-1]:
        match = _INDEX_RE.match(part)
        if match:
            index = int(match.group(1))
            indices.append(index)
            owner = owner[index]
        else:
            owner = getattr(owner, part)
    field_name = parts[-1]
    type_hints = get_type_hints(type(owner))
    field_type = type_hints.get(field_name)
    return owner, field_name, field_type, indices

def analyze_field_type(field_type: Any) -> tuple[Any, bool, bool]:
    is_optional = False

    origin = get_origin(field_type)
    args = get_args(field_type)

    # Optional[T] / T | None
    if origin in (Union, types.UnionType):
        non_none = [arg for arg in args if arg is not type(None)]

        if len(non_none) == 1:
            is_optional = True
            field_type = non_none[0]
            origin = get_origin(field_type)
            args = get_args(field_type)

    # list[T] or typing.List[T]
    if origin is list:
        item_type = args[0] if args else Any
        return item_type, is_optional, True

    if field_type is list:
        return Any, is_optional, True

    return field_type, is_optional, False

main_screen_text = """
Data is structured according to ONVIF standard. 
Open tree branches using the enter key to find
camera field settings. Editable fields are marked 
with a green pencil icon '✎'. 

Most fields can be set by using the F2 key and 
pressing enter to commit the change. Some fields
may require the additional step of navigating to 
the branch parent node, then using the 'w' key 
to write the data to the camera, please refer to
the individual field instructions for details.

Multicast streaming can be started and stopped 
from the profiles branch. The first profile is 
generally the main profile and the subsequent 
profiles show sub streams. 
"""

field_descriptions = {
    "network_gateway": 
"""
This value sets the access point for the camera
to reach other networks, including the internet.

This can be set by DHCP or manually. The value
of the field must be a valid ip address on the
local subnet.

""",

    "hostname":
"""
A value that may be assigned by DHCP or set 
manually that identifies the camera on the
network.

Depending on settings, the name may be set 
by DHCP or manually.

""",

    "hostname.from_dhcp":
"""
This operation controls whether the hostname
is set manually or retrieved via DHCP.
""",

    "hostname.name":
"""
This operation sets the hostname on a device.
It shall be possible to set the device hostname 
configurations through the SetHostname command.

A device shall accept string formated according
to RFC 1123 section 2.1 or alternatively to 
RFC 952, other string shall be considered as 
invalid strings.
""",

    "dns":
"""
A value that may be assigned by DHCP or set
manually that specifies the Domain Name
Server to be used by the camera
""",

    "dns.from_dhcp":
"""
Indicate if the DNS address is to be set 
automatically using DHCP. If this value is
set to False, there should be at least one
value set in the dns_manual list to identify 
DNS servers
""",

    "ptz":
"""
Control camera position using the commands

i - info
w - up
s - down
a - left
d - right
z - zoom in
x - zoom out
c - stop
""",

    "ptz.presets":
"""
Presets are used to assign camera position
to a field. Add a new preset at the current
position by typing the 'n' key.

Open the branch to see the presets. Actions
can be taken on the presets individually
when the preset is highlighted.
""",

    "ptz.presets.[*]":
"""

The preset can modified from this screen.
Position the camera at the desired settings
then use the 's' key to set.

g - goto preset position
s - set preset to current position
d - delete
""",

    "ptz.presets.[*].token":
"""
The token is a read only field assigned by
the camera to identify the preset.
""",

    "ptz.presets.[*].name":
"""
The name is a user-editable field that can
be used to identify the preset.

The name is saved when the position is set
using the 's' key from the preset branch.
Note that the position of the preset will
also be set in that case, so make sure that 
the camera is in the desired position already.
""",

    "ptz.presets.[*].ptz_position":
"""
The ptz_position field is designed to be a 
read only field to hold the coordinates for the 
position, but is rarely used in practice.
""",

    "ptz.tours":
"""
Tours are a sequence of preset gotos. Build
a tour by adding spots which are a preset and 
a stay time.

Add a new tour using the 'n' key. Actions can 
be taken on individual tours by opening the 
tours branch and highlighting the tour.
""",

    "ptz.tours.[*]":
"""
Tours are built by adding spots to the tour.
Navigate to the spots branch to add spots. 
Spots are edited individually once they have 
been added.

Please consult the tour_options branch to 
find allowed settings for tours.

s - start tour 
t - stop tour 
d - delete tour
w - write tour to camera (after spot edit) 
""",

    "ptz.tours.[*].spots":
"""
Add spots to the tour using the 'n' key.
Once the spots have been added, open the 
branch and navigate to the spot to edit 
the preset and stay_time.

The tour main branch will show modified.
From there, use the 'w' key to write the 
spots data to the camera.

n - add new spot
""",

    "ptz.tours.[*].spots.[*]":
"""
Open the spot leaves to edit the preset and
stay_time. Consult the tour_options branch
to view allowed entries.

To delete a spot, use the 'd' key.
""",

    "ptz.tours.[*].spots.[*].preset_token":
"""
Use the F2 key to activate the editor and 
type in a preset token.


Allowed values are shown in the tour_options 
branch. After editing, the tour will show 
as (* modified). Use the 'w' from the tour 
main branch to write the tour to the camera.
""",

    "ptz.tours.[*].spots.[*].stay_time":
"""
Use the F2 key to activate the editor and 
type in a stay time.


Allowed values are shown in the tour_options 
branch. After editing, the tour will show 
as (* modified). Use the 'w' from the tour 
main branch to write the tour to the camera.
""",

    "system_date_and_time":
"""
Camera time shown in the video feed is 
local_date_time. Authentication is hashed
using the utc_date_time. time_zone and 
daylight_savings are used to calculate the
local_date_time derived from utc_date_time.

Note that many cameras do not properly 
implement time_zone and daylight_savings, 
so experimentation may be required to find a 
suitable setting.

Editable fields can be changed using the F2
key at which point the system_date_and_time
branch node will display a (* modified) tag.
use the 'w' key to commit the changes.

s - Synch camera time to computer without 
    using daylight_savings

u - Set camera to 'Local as UTC' which ignores
    time_zone and daylight_savings settings

t - Show current time and settings

w - Write user edited changes to camera
""",

    "system_date_and_time.date_time_type":
"""
This can can be set to either 'Manual' or
'NTP'. If set to 'NTP', consult the ntp branch
node for server settings.

NTP stands for Network Time Protocol. There 
are many free servers on the internet that can
provide this service. The camera will need 
internet access if this setting is used with
an internet server.
""",

    "system_date_and_time.daylight_savings":
"""
Setting this flag may adjust the local_date_time
by an hour. Note that many cameras do not 
implement this function in which case setting
this field will have no effect on the displayed 
time. Some cameras may require a full POSIX 
time zone string representation in the time_zone
field for this settings to take hold.
""",

    "system_date_and_time.time_zone.tz":
"""
Specifications indicate a POSIX compliant time
zone string for this field. Most cameras will 
accept something like UTC+4:00, where 4 is the
number of hours behind UTC time, set for the 
camera time zone. If you are unsure of your 
time zone, use the 's' key command from the 
system_date_and_time node branch to automatically
set the time zone.

Some cameras will accept a full POSIX time_zone
such as 

UTC+4:00:00DST01:00:00,M4.1.0/02:00:00,M10.5.0/02:00:00

where 

UTC+4:00:00 is the number of hours behind UTC 
DST01:00:00 is the number of hours to adjust for 
            daylight savings
M4.1.0/02:00:00 means the the transition to DST 
            occurs on the first Sunday of April 
            at 2:00 AM
M10.5.0/02:00:00 means that the transition back 
            to standard time occurs on the last 
            Sunday of October at 2:00 AM

""",

    "system_date_and_time.utc_date_time":
"""
UTC is Universal Time Coordinated and is the 
same for all clocks everywhere and does not 
depend on Daylight Savings. This field is used 
during authentication and is the basis for the 
time_offset field.
""",

    "system_date_and_time.local_date_time":
"""
This is the time displayed in the camera video
feed and accounts for time zone and daylight 
savings if properly implemented.
""",

        "time_offset":
"""
This is the difference in UTC time from the 
computer to the camera and is used during 
authentication as a hash.
""",

    "capabilities":
"""
Settings supported by the camera.
""",

    "event_properties":
"""
Open the branch and select events from the
topic_set for observation.

Events from the camera can be captured by
selecting events from the topic_set and 
starting either a listener or polling operation.
""",

    "event_properties.topic_set":
"""
Events can be selected from the list for 
monitoring. Select an event by highlighting it 
then using either the space or enter key. 
Highlighted events will show a star next to the 
event name. Once events have been selected, 
navigate back to this node and start the event 
monitoring algorithm of yhour preference.

There are two types of monitoring available, 
receive and pull. The receive type will start an 
http server that the camera will use to push an 
event to the host computer. Pull type will cause 
the host computer to poll the camera at an 
interval of five seconds to query for events.

You can also select all events in the list for
monitoring.

r - recieve selected events

R - receive all events 

p - pull selected events 

P - pull all events 

u - unsubscribe all events
""",

    "relay_outputs":
"""
Camera relays can be controlled from the sub
branches of this node.
""",

    "relay_outputs.[*].properties":
"""
Available settings and ranges for one or all 
relay outputs. A device that has one or more 
RelayOutputs will show these fields.
""",

    "relay_outputs.[*].properties.mode":
"""
'Bistable' or 'Monostable'

Bistable – After setting the state, the relay 
           remains in this state.

Monostable – After setting the state, the relay 
             returns to its idle state after 
             the specified time.
""",

    "relay_outputs.[*].properties.delay_time":
"""
Time after which the relay returns to its idle 
state if it is in monostable mode. If the Mode 
field is set to bistable mode the value of the 
parameter can be ignored.
""",

    "relay_outputs.[*].properties.idle_state":
"""
'open' or 'closed'

'open' means that the relay is open when the 
relay state is set to 'inactive' through the 
trigger command and closed when the state is 
set to 'active' through the same command.

'closed' means that the relay is closed when 
the relay state is set to 'inactive' through 
the trigger command and open when the state 
is set to 'active' through the same command.
""",

    "relay_outputs.[*]":
"""
If the camera is equipped with a relay, it 
can be configured and triggered from here. 
Set the relay parameters in the editable 
fields in the properties field in the list 
below. Consult the options branch for allowed 
values.

w - Write user edited values to the camera

a - Activate the relay

i - Inactivate the relay
""",

    "relay_outputs.[*].token":
"""
A read only field referencing the relay output.
""",

    "profiles":
"""
Media settings are found here. The first profile 
is generally the main profile and the subsequent 
profiles show sub streams. Both video and audio 
parameters may be observed and set here.

Branches labelled _options show available 
settings for fields.
""",

    "profiles.[*]":
"""
Open the branches below to view profile fields.

Multicast for this profile can be started and 
stopped from this node.

's' - Start multicast streaming

't' - Stop multicast streaming
""",

    "profiles.[*].video_encoder":
"""
Video encoder configuration controls how video
from the profile is encoded and streamed.

It includes the codec, resolution, quality,
rate control, multicast behavior, and session
timeout for the video stream.
""",

    "profiles.[*].video_encoder.token":
"""
Unique identifier for this video encoder
configuration.

This token is used when referencing or updating
the encoder configuration.
""",

    "profiles.[*].video_encoder.name":
"""
Human-readable name assigned to this video
encoder configuration.
""",

    "profiles.[*].video_encoder.use_count":
"""
Number of media profiles or consumers currently
using this configuration.

Changing a configuration with a nonzero use count
may affect other profiles or users.
""",

    "profiles.[*].video_encoder.encoding":
"""
Encoding format used for the video stream.

Typical ONVIF values include JPEG, MPEG4, and H264.
""",

    "profiles.[*].video_encoder.resolution":
"""
Pixel resolution of the encoded video stream.

This represents the width and height used for
the encoded image.
""",

    "profiles.[*].video_encoder.quality":
"""
Relative video quality value.

A higher value within the supported quality range
means higher video quality.
""",

    "profiles.[*].video_encoder.rate_control":
"""
Rate control settings define how the device limits
frame rate, encoding interval, and bitrate.

If rate control is omitted by a device, the current
rate control behavior may be vendor-specific.
""",

    "profiles.[*].video_encoder.rate_control.frame_rate_limit":
"""
Maximum output frame rate in frames per second.

This limits how many encoded video frames may be
transmitted per second.
""",

    "profiles.[*].video_encoder.rate_control.encoding_interval":
"""
Interval at which images are encoded and transmitted.

A value of 1 means every frame is encoded. A value
of 2 means every second frame is encoded.
""",

    "profiles.[*].video_encoder.rate_control.bitrate_limit":
"""
Maximum output bitrate for the encoded stream.

The ONVIF media specification describes this value
as a bitrate limit in kbps.
""",

    "profiles.[*].video_encoder.multicast":
"""
Multicast configuration for the video stream.

This controls the multicast address, port, TTL, and
whether multicast streaming starts automatically.


""",

    "profiles.[*].video_encoder.multicast.ip_address":
"""
Multicast IP address used for streaming.

The address may be derived from either the IPv4 or
IPv6 address element in the ONVIF multicast
configuration.
""",

    "profiles.[*].video_encoder.multicast.port":
"""
UDP port used for multicast video streaming.
""",

    "profiles.[*].video_encoder.multicast.ttl":
"""
Time-to-live value for multicast packets.

This controls how far multicast packets may travel
across routed networks.
""",

    "profiles.[*].video_encoder.multicast.auto_start":
"""
Indicates whether multicast streaming should start
automatically.
""",

    "profiles.[*].video_encoder.session_timeout":
"""
Session timeout for the video stream.

This value defines stream session behavior and is
usually expressed as an XML duration.
""",

    "profiles.[*].video_encoder.gov_length":
"""
Group of Video length for MPEG4 or H264 encoding.

This value controls the interval of intra-coded
frames used by the encoder.
""",

    "profiles.[*].video_encoder.profile":
"""
Encoding profile used for codec-specific settings.

For H264 this corresponds to the H264 profile, such
as Baseline, Main, or High when supported.
""",

    "profiles.[*].audio_encoder":
"""
Audio encoder configuration controls how audio
captured by the device is encoded and streamed.

It includes the codec, bitrate, sample rate,
multicast settings, and session timeout.
""",

    "profiles.[*].audio_encoder.token":
"""
Unique identifier for this audio encoder
configuration.

This token is used when referencing or updating
the encoder configuration.
""",

    "profiles.[*].audio_encoder.name":
"""
Human-readable name assigned to this audio
encoder configuration.
""",

    "profiles.[*].audio_encoder.use_count":
"""
Number of media profiles or consumers currently
using this configuration.

Changing a configuration with a nonzero use count
may affect other profiles or users.
""",

    "profiles.[*].audio_encoder.encoding":
"""
Encoding format used for the audio stream.

Typical ONVIF values include G711, G726, AAC,
and other codecs supported by the device.
""",

    "profiles.[*].audio_encoder.bitrate":
"""
Target audio bitrate used by the encoder.

Higher bitrates generally improve audio quality
while increasing network bandwidth usage.
""",

    "profiles.[*].audio_encoder.sample_rate":
"""
Audio sampling frequency in Hertz.

Common values include 8000, 16000, 32000,
44100, and 48000 depending on codec support.
""",

    "profiles.[*].audio_encoder.multicast":
"""
Multicast configuration for the audio stream.

This controls the multicast address, port, TTL,
and automatic startup behavior.
""",

    "profiles.[*].audio_encoder.multicast.ip_address":
"""
Multicast IP address used for audio streaming.

The address may be derived from either the IPv4
or IPv6 multicast configuration.
""",

    "profiles.[*].audio_encoder.multicast.port":
"""
UDP port used for multicast audio streaming.
""",

    "profiles.[*].audio_encoder.multicast.ttl":
"""
Time-to-live value for multicast packets.

This controls how far multicast packets may
travel across routed networks.
""",

    "profiles.[*].audio_encoder.multicast.auto_start":
"""
Indicates whether multicast audio streaming
should start automatically.
""",

    "profiles.[*].audio_encoder.session_timeout":
"""
Session timeout for the audio stream.

This value defines stream session behavior and
is typically expressed as an XML duration.
""",

}
