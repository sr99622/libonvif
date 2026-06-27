from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
from lxml import etree
from libonvif.utils.xml import text, int_text, bool_text, attr, float_text, text_list, NS
from .imaging import ImagingOptions, ImagingSettings, Bounds

@dataclass
class VideoSourceConfiguration:
    token: Optional[str] = None
    name: Optional[str] = None
    use_count: Optional[int] = None
    source_token: Optional[str] = None
    bounds: Bounds = field(default_factory=Bounds)

@dataclass
class Resolution:
    width: Optional[int] = None
    height: Optional[int] = None

@dataclass
class RateControl:
    frame_rate_limit: Optional[int] = None
    encoding_interval: Optional[int] = None
    bitrate_limit: Optional[int] = None

@dataclass
class MulticastConfiguration:
    ip_address: Optional[str] = None
    port: Optional[int] = None
    ttl: Optional[int] = None
    auto_start: Optional[bool] = None

@dataclass
class VideoEncoderConfiguration:
    token: Optional[str] = None
    name: Optional[str] = None
    use_count: Optional[int] = None
    encoding: Optional[str] = None
    resolution: Optional[str] = None
    quality: Optional[float] = None
    rate_control: RateControl = field(default_factory=RateControl)
    multicast: MulticastConfiguration = field(default_factory=MulticastConfiguration)
    session_timeout: Optional[str] = None
    gov_length: Optional[int] = 1
    profile: Optional[str] = None

@dataclass
class IntRange:
    min: Optional[int] = None
    max: Optional[int] = None

@dataclass
class JpegOptions:
    resolutions_available: list[Resolution] = field(default_factory=list)
    frame_rate_range: IntRange = field(default_factory=IntRange)
    encoding_interval_range: IntRange = field(default_factory=IntRange)

@dataclass
class Mpeg4Options:
    resolutions_available: list[Resolution] = field(default_factory=list)
    gov_length_range: IntRange = field(default_factory=IntRange)
    frame_rate_range: IntRange = field(default_factory=IntRange)
    encoding_interval_range: IntRange = field(default_factory=IntRange)
    profiles_supported: list[str] = field(default_factory=list)

@dataclass
class H264Options:
    resolutions_available: list[Resolution] = field(default_factory=list)
    gov_length_range: IntRange = field(default_factory=IntRange)
    frame_rate_range: IntRange = field(default_factory=IntRange)
    encoding_interval_range: IntRange = field(default_factory=IntRange)
    profiles_supported: list[str] = field(default_factory=list)

@dataclass
class VideoEncoderConfigurationOptions:
    quality_range: IntRange = field(default_factory=IntRange)
    jpeg: Optional[JpegOptions] = None
    mpeg4: Optional[Mpeg4Options] = None
    h264: Optional[H264Options] = None

@dataclass
class AudioEncoderConfigurationOption:
    encoding: Optional[str] = None
    bitrate_list: list[int] = field(default_factory=list)
    sample_rate_list: list[int] = field(default_factory=list)

@dataclass
class AudioEncoderConfigurationOptions:
    options: list[AudioEncoderConfigurationOption] = field(default_factory=list)    

@dataclass
class AudioSourceConfiguration:
    token: Optional[str] = None
    name: Optional[str] = None
    use_count: Optional[int] = None
    source_token: Optional[str] = None

@dataclass
class AudioEncoderConfiguration:
    token: Optional[str] = None
    name: Optional[str] = None
    use_count: Optional[int] = None
    encoding: Optional[str] = None
    bitrate: Optional[int] = None
    sample_rate: Optional[int] = None
    multicast: MulticastConfiguration = field(default_factory=MulticastConfiguration)
    session_timeout: Optional[str] = None

@dataclass
class PTZConfiguration:
    token: Optional[str] = None
    name: Optional[str] = None
    use_count: Optional[int] = None
    node_token: Optional[str] = None
    default_absolute_pant_tilt_position_space: Optional[str] = None
    default_absolute_zoom_position_space: Optional[str] = None
    default_relative_pan_tilt_translation_space: Optional[str] = None
    default_relative_zoom_translation_space: Optional[str] = None
    default_continuous_pan_tilt_velocity_space: Optional[str] = None
    default_continuous_zoom_velocity_space: Optional[str] = None
    default_ptz_timeout: Optional[str] = None

@dataclass
class VideoAnalyticsConfiguration:
    token: Optional[str] = None
    name: Optional[str] = None
    use_count: Optional[int] = None

@dataclass
class MetadataConfiguration:
    token: Optional[str] = None
    name: Optional[str] = None
    use_count: Optional[int] = None
    ptz_status: Optional[bool] = None
    events: Optional[bool] = None
    multicast: MulticastConfiguration = field(default_factory=MulticastConfiguration)
    session_timeout: Optional[str] = None

@dataclass
class Profile:
    token: Optional[str] = None
    fixed: Optional[bool] = None
    name: Optional[str] = None
    video_source: Optional[VideoSourceConfiguration] = None
    video_encoder: Optional[VideoEncoderConfiguration] = None
    video_encoder_options: Optional[VideoEncoderConfigurationOptions] = None
    audio_source: Optional[AudioSourceConfiguration] = None
    audio_encoder: Optional[AudioEncoderConfiguration] = None
    audio_encoder_options: Optional[AudioEncoderConfigurationOptions] = None
    ptz: Optional[PTZConfiguration] = None
    video_analytics: Optional[VideoAnalyticsConfiguration] = None
    metadata: Optional[MetadataConfiguration] = None
    stream_uri: Optional[str] = None
    snapshot_uri: Optional[str] = None
    imaging_settings: Optional[ImagingSettings] = None
    imaging_options: Optional[ImagingOptions] = None

@dataclass
class GetProfilesResponse:
    profiles: list[Profile] = field(default_factory=list)

def parse_multicast(elem: Optional[etree._Element]) -> MulticastConfiguration:
    if elem is None:
        return MulticastConfiguration()

    address_type = text(elem, "tt:Address/tt:Type")
    ipv4_address = text(elem, "tt:Address/tt:IPv4Address")
    ipv6_address = text(elem, "tt:Address/tt:IPv6Address")
    ip_address = None
    if address_type == "IPv4":
        ip_address = ipv4_address
    elif address_type == "IPv6":
        ip_address = ipv6_address
    else:
        ip_address = ipv4_address or ipv6_address
    return MulticastConfiguration(
        ip_address=ip_address,
        port=int_text(elem, "tt:Port"),
        ttl=int_text(elem, "tt:TTL"),
        auto_start=bool_text(elem, "tt:AutoStart"),
    )

def parse_profiles_response(xml: str) -> GetProfilesResponse:
    if not xml: return
    root = etree.fromstring(xml.encode('utf-8'))

    profile_elems = root.findall(".//trt:GetProfilesResponse/trt:Profiles", NS)
    if not profile_elems:
        raise ValueError("Could not find trt:GetProfilesResponse/trt:Profiles")

    response = GetProfilesResponse()

    for p in profile_elems:
        profile = Profile(
            token=attr(p, "token"),
            fixed=bool_text(p, "fixed"),
            name=text(p, "tt:Name"),
        )

        video_source = p.find("tt:VideoSourceConfiguration", NS)
        if video_source is not None:
            bounds = video_source.find("tt:Bounds", NS)

            profile.video_source = VideoSourceConfiguration(
                token=attr(video_source, "token"),
                name=text(video_source, "tt:Name"),
                use_count=int_text(video_source, "tt:UseCount"),
                source_token=text(video_source, "tt:SourceToken"),
                bounds=Bounds(
                    x=int(attr(bounds, "x")) if bounds is not None and attr(bounds, "x") else None,
                    y=int(attr(bounds, "y")) if bounds is not None and attr(bounds, "y") else None,
                    width=int(attr(bounds, "width")) if bounds is not None and attr(bounds, "width") else None,
                    height=int(attr(bounds, "height")) if bounds is not None and attr(bounds, "height") else None,
                ),
            )

        video_encoder = p.find("tt:VideoEncoderConfiguration", NS)
        if video_encoder is not None:
            encoding = text(video_encoder, "tt:Encoding")
            gov_length = None
            encoder_profile = None
            if encoding == "H264":
                gov_length=int_text(video_encoder, "tt:H264/tt:GovLength")
                encoder_profile=text(video_encoder, "tt:H264/tt:H264Profile")
            elif encoding == "MPEG4":
                gov_length=int_text(video_encoder, "tt:H264/tt:GovLength")
                encoder_profile=text(video_encoder, "tt:H264/tt:H264Profile")
                
            profile.video_encoder = VideoEncoderConfiguration(
                token=attr(video_encoder, "token"),
                name=text(video_encoder, "tt:Name"),
                use_count=int_text(video_encoder, "tt:UseCount"),
                encoding=encoding,
                resolution=f"{int_text(video_encoder, "tt:Resolution/tt:Width")} x {int_text(video_encoder, "tt:Resolution/tt:Height")}",
                quality=float_text(video_encoder, "tt:Quality"),
                rate_control=RateControl(
                    frame_rate_limit=int_text(
                        video_encoder, "tt:RateControl/tt:FrameRateLimit"
                    ),
                    encoding_interval=int_text(
                        video_encoder, "tt:RateControl/tt:EncodingInterval"
                    ),
                    bitrate_limit=int_text(
                        video_encoder, "tt:RateControl/tt:BitrateLimit"
                    ),
                ),
                multicast=parse_multicast(video_encoder.find("tt:Multicast", NS)),
                session_timeout=text(video_encoder, "tt:SessionTimeout"),
                gov_length=gov_length,
                profile=encoder_profile,
            )

        audio_source = p.find("tt:AudioSourceConfiguration", NS)
        if audio_source is not None:
            profile.audio_source = AudioSourceConfiguration(
                token=attr(audio_source, "token"),
                name=text(audio_source, "tt:Name"),
                use_count=int_text(audio_source, "tt:UseCount"),
                source_token=text(audio_source, "tt:SourceToken"),
            )

        audio_encoder = p.find("tt:AudioEncoderConfiguration", NS)
        if audio_encoder is not None:
            profile.audio_encoder = AudioEncoderConfiguration(
                token=attr(audio_encoder, "token"),
                name=text(audio_encoder, "tt:Name"),
                use_count=int_text(audio_encoder, "tt:UseCount"),
                encoding=text(audio_encoder, "tt:Encoding"),
                bitrate=int_text(audio_encoder, "tt:Bitrate"),
                sample_rate=int_text(audio_encoder, "tt:SampleRate"),
                multicast=parse_multicast(audio_encoder.find("tt:Multicast", NS)),
                session_timeout=text(audio_encoder, "tt:SessionTimeout"),
            )

        ptz = p.find("tt:PTZConfiguration", NS)
        if ptz is not None:
            profile.ptz = PTZConfiguration(
                token=attr(ptz, "token"),
                name=text(ptz, "tt:Name"),
                use_count=int_text(ptz, "tt:UseCount"),
                node_token=text(ptz, "tt:NodeToken"),
                default_absolute_pant_tilt_position_space=text(
                    ptz, "tt:DefaultAbsolutePantTiltPositionSpace"
                ),
                default_absolute_zoom_position_space=text(
                    ptz, "tt:DefaultAbsoluteZoomPositionSpace"
                ),
                default_relative_pan_tilt_translation_space=text(
                    ptz, "tt:DefaultRelativePanTiltTranslationSpace"
                ),
                default_relative_zoom_translation_space=text(
                    ptz, "tt:DefaultRelativeZoomTranslationSpace"
                ),
                default_continuous_pan_tilt_velocity_space=text(
                    ptz, "tt:DefaultContinuousPanTiltVelocitySpace"
                ),
                default_continuous_zoom_velocity_space=text(
                    ptz, "tt:DefaultContinuousZoomVelocitySpace"
                ),
                default_ptz_timeout=text(ptz, "tt:DefaultPTZTimeout"),
            )

        analytics = p.find("tt:VideoAnalyticsConfiguration", NS)
        if analytics is not None:
            profile.video_analytics = VideoAnalyticsConfiguration(
                token=attr(analytics, "token"),
                name=text(analytics, "tt:Name"),
                use_count=int_text(analytics, "tt:UseCount"),
            )

        metadata = p.find("tt:MetadataConfiguration", NS)
        if metadata is not None:
            profile.metadata = MetadataConfiguration(
                token=attr(metadata, "token"),
                name=text(metadata, "tt:Name"),
                use_count=int_text(metadata, "tt:UseCount"),
                ptz_status=bool_text(metadata, "tt:PTZStatus/tt:Status"),
                events=metadata.find("tt:Events", NS) is not None,
                multicast=parse_multicast(metadata.find("tt:Multicast", NS)),
                session_timeout=text(metadata, "tt:SessionTimeout"),
            )

        response.profiles.append(profile)

    return response.profiles

def parse_int_range(elem: Optional[etree._Element]) -> IntRange:
    if elem is None:
        return IntRange()

    return IntRange(
        min=int_text(elem, "tt:Min"),
        max=int_text(elem, "tt:Max"),
    )

def parse_resolutions(parent: etree._Element) -> list[Resolution]:
    return [
        Resolution(
            width=int_text(r, "tt:Width"),
            height=int_text(r, "tt:Height"),
        )
        for r in parent.findall("tt:ResolutionsAvailable", NS)
    ]

def parse_jpeg_options(elem: Optional[etree._Element]) -> Optional[JpegOptions]:
    if elem is None:
        return None

    return JpegOptions(
        resolutions_available=parse_resolutions(elem),
        frame_rate_range=parse_int_range(elem.find("tt:FrameRateRange", NS)),
        encoding_interval_range=parse_int_range(
            elem.find("tt:EncodingIntervalRange", NS)
        ),
    )

def parse_mpeg4_options(elem: Optional[etree._Element]) -> Optional[Mpeg4Options]:
    if elem is None:
        return None

    return Mpeg4Options(
        resolutions_available=parse_resolutions(elem),
        gov_length_range=parse_int_range(elem.find("tt:GovLengthRange", NS)),
        frame_rate_range=parse_int_range(elem.find("tt:FrameRateRange", NS)),
        encoding_interval_range=parse_int_range(
            elem.find("tt:EncodingIntervalRange", NS)
        ),
        profiles_supported=[
            e.text.strip()
            for e in elem.findall("tt:MPEG4ProfilesSupported", NS)
            if e.text
        ],
    )

def parse_h264_options(elem: Optional[etree._Element]) -> Optional[H264Options]:
    if elem is None:
        return None

    return H264Options(
        resolutions_available=parse_resolutions(elem),
        gov_length_range=parse_int_range(elem.find("tt:GovLengthRange", NS)),
        frame_rate_range=parse_int_range(elem.find("tt:FrameRateRange", NS)),
        encoding_interval_range=parse_int_range(
            elem.find("tt:EncodingIntervalRange", NS)
        ),
        profiles_supported=[
            e.text.strip()
            for e in elem.findall("tt:H264ProfilesSupported", NS)
            if e.text
        ],
    )

def parse_video_encoder_configuration_options_response(xml: str) -> VideoEncoderConfigurationOptions:
    if not xml: return
    root = etree.fromstring(xml.encode('utf-8'))

    options = root.find(
        ".//trt:GetVideoEncoderConfigurationOptionsResponse/trt:Options",
        NS,
    )
    if options is None:
        raise ValueError(
            "Could not find trt:GetVideoEncoderConfigurationOptionsResponse/trt:Options"
        )

    return VideoEncoderConfigurationOptions(
        quality_range=parse_int_range(options.find("tt:QualityRange", NS)),
        jpeg=parse_jpeg_options(options.find("tt:JPEG", NS)),
        mpeg4=parse_mpeg4_options(options.find("tt:MPEG4", NS)),
        h264=parse_h264_options(options.find("tt:H264", NS)),
    )

def parse_audio_encoder_configuration_option(elem: etree._Element) -> AudioEncoderConfigurationOption:
    return AudioEncoderConfigurationOption(
        encoding=text(elem, "tt:Encoding"),
        bitrate_list=[
            int(e.text.strip())
            for e in elem.findall("tt:BitrateList/tt:Items", NS)
            if e.text
        ],
        sample_rate_list=[
            int(e.text.strip())
            for e in elem.findall("tt:SampleRateList/tt:Items", NS)
            if e.text
        ],
    )

def parse_audio_encoder_configuration_options_response(xml: str) -> list[AudioEncoderConfigurationOption]:
    if not xml: return
    root = etree.fromstring(xml.encode('utf-8'))

    options_elem = root.find(
        ".//trt:GetAudioEncoderConfigurationOptionsResponse/trt:Options",
        NS,
    )
    if options_elem is None:
        raise ValueError(
            "Could not find trt:GetAudioEncoderConfigurationOptionsResponse/trt:Options"
        )

    return [
        parse_audio_encoder_configuration_option(option)
        for option in options_elem.findall("tt:Options", NS)
    ]


@dataclass
class IntRange:
    min: int | None = None
    max: int | None = None

@dataclass
class AudioOutputConfigurationOptions:
    output_tokens_available: list[str] = field(default_factory=list)
    send_primacy_options: list[str] = field(default_factory=list)
    output_level_range: IntRange | None = None

@dataclass
class AudioDecoderConfiguration:
    token: Optional[str] = None
    name: Optional[str] = None
    use_count: Optional[int] = None

@dataclass
class AudioOutputConfiguration:
    token: Optional[str] = None
    name: Optional[str] = None
    use_count: Optional[int] = None
    output_token: Optional[str] = None
    send_primacy: Optional[str] = None
    output_level: Optional[int] = None

@dataclass
class AudioDecoderCodecOptions:
    bitrate_list: list[int] = field(default_factory=list)
    sample_rate_list: list[int] = field(default_factory=list)

@dataclass
class AudioDecoderConfigurationOptions:
    aac: Optional[AudioDecoderCodecOptions] = None
    g711: Optional[AudioDecoderCodecOptions] = None
    g726: Optional[AudioDecoderCodecOptions] = None

@dataclass
class AudioOut:
    audio_outputs: list[AudioOutputConfiguration] = None
    audio_output_options: Optional[AudioOutputConfigurationOptions] = None
    audio_decoder: Optional[AudioDecoderConfiguration] = None
    audio_decoder_options: Optional[AudioDecoderConfigurationOptions] = None

def parse_int_list(elem: Optional[etree._Element]) -> list[int]:
    if elem is None:
        return []

    return [
        int(e.text.strip())
        for e in elem.findall("tt:Items", NS)
        if e.text
    ]

def parse_int_items(elem: Optional[etree._Element]) -> list[int]:
    if elem is None:
        return []

    values = []
    for item in elem.findall("tt:Items", NS):
        if item.text and item.text.strip():
            values.append(int(item.text.strip()))
    return values

def parse_audio_decoder_codec_options(
    elem: Optional[etree._Element],
) -> Optional[AudioDecoderCodecOptions]:
    if elem is None:
        return None

    return AudioDecoderCodecOptions(
        bitrate_list=parse_int_items(elem.find("tt:Bitrate", NS)),
        sample_rate_list=parse_int_items(elem.find("tt:SampleRateRange", NS)),
    )

def parse_audio_decoder_configuration(elem: etree._Element) -> AudioDecoderConfiguration:
    return AudioDecoderConfiguration(
        token=attr(elem, "token"),
        name=text(elem, "tt:Name"),
        use_count=int_text(elem, "tt:UseCount"),
    )

def parse_audio_output_configuration(elem: etree._Element) -> AudioOutputConfiguration:
    return AudioOutputConfiguration(
        token=attr(elem, "token"),
        name=text(elem, "tt:Name"),
        use_count=int_text(elem, "tt:UseCount"),
        output_token=text(elem, "tt:OutputToken"),
        send_primacy=text(elem, "tt:SendPrimacy"),
        output_level=int_text(elem, "tt:OutputLevel"),
    )

def parse_audio_decoder_configurations_response(xml: str) -> list[AudioDecoderConfiguration]:
    if not xml: return
    root = etree.fromstring(xml.encode('utf-8'))

    elems = root.findall(
        ".//trt:GetAudioDecoderConfigurationsResponse/trt:Configurations",
        NS,
    )

    return [
        parse_audio_decoder_configuration(elem)
        for elem in elems
    ]

def parse_audio_output_configurations_response(xml: str) -> list[AudioOutputConfiguration]:
    if not xml: return
    root = etree.fromstring(xml.encode('utf-8'))

    elems = root.findall(
        ".//trt:GetAudioOutputConfigurationsResponse/trt:Configurations",
        NS,
    )

    return [
        parse_audio_output_configuration(elem)
        for elem in elems
    ]

def parse_audio_decoder_configuration_options_response(xml: str) -> AudioDecoderConfigurationOptions:
    if not xml: return
    root = etree.fromstring(xml.encode('utf-8'))

    response = root.find(
        ".//trt:GetAudioDecoderConfigurationOptionsResponse",
        NS,
    )
    if response is None:
        raise ValueError(
            "Could not find trt:GetAudioDecoderConfigurationOptionsResponse"
        )

    return AudioDecoderConfigurationOptions(
        aac=parse_audio_decoder_codec_options(
            response.find(".//tt:AACDecOptions", NS)
        ),
        g711=parse_audio_decoder_codec_options(
            response.find(".//tt:G711DecOptions", NS)
        ),
        g726=parse_audio_decoder_codec_options(
            response.find(".//tt:G726DecOptions", NS)
        ),
    )

def parse_audio_output_configuration_options(
    xml: str,
) -> AudioOutputConfigurationOptions:
    root = etree.fromstring(xml.encode("utf-8"))

    options_elem = root.xpath(
        ".//trt:GetAudioOutputConfigurationOptionsResponse/trt:Options",
        namespaces=root.nsmap,
    )

    if not options_elem:
        return AudioOutputConfigurationOptions()

    options = options_elem[0]

    range_elem = options.xpath("tt:OutputLevelRange", namespaces=root.nsmap)

    output_level_range = None
    if range_elem:
        output_level_range = IntRange(
            min=int_text(range_elem[0], "tt:Min"),
            max=int_text(range_elem[0], "tt:Max"),
        )

    return AudioOutputConfigurationOptions(
        output_tokens_available=text_list(options, "tt:OutputTokensAvailable"),
        send_primacy_options=text_list(options, "tt:SendPrimacyOptions"),
        output_level_range=output_level_range,
    )