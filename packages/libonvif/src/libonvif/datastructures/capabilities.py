from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
from lxml import etree
from libonvif.utils.xml import text, int_text, bool_text, NS
from .ptz import PTZPreset, PresetTour, PresetTourOptions
from .event import EventServiceCapabilities, EventProperties
from .device_io import DeviceIOServiceCapabilities, RelayOutput

@dataclass
class OnvifVersion:
    major: int
    minor: int

    def __str__(self) -> str:
        return f"{self.major}.{self.minor}"

@dataclass
class AnalyticsCapabilities:
    xaddr: Optional[str] = None
    rule_support: Optional[bool] = None
    analytics_module_support: Optional[bool] = None

@dataclass
class NetworkCapabilities:
    ip_filter: Optional[bool] = None
    zero_configuration: Optional[bool] = None
    ip_version6: Optional[bool] = None
    dyn_dns: Optional[bool] = None
    dot11_configuration: Optional[bool] = None

@dataclass
class SystemCapabilities:
    discovery_resolve: Optional[bool] = None
    discovery_bye: Optional[bool] = None
    remote_discovery: Optional[bool] = None
    system_backup: Optional[bool] = None
    system_logging: Optional[bool] = None
    firmware_upgrade: Optional[bool] = None
    supported_versions: list[OnvifVersion] = field(default_factory=list)
    http_firmware_upgrade: Optional[bool] = None
    http_system_backup: Optional[bool] = None
    http_system_logging: Optional[bool] = None
    http_support_information: Optional[bool] = None

@dataclass
class IOCapabilities:
    input_connectors: Optional[int] = None
    relay_outputs: Optional[int] = None
    auxiliary: Optional[bool] = None

@dataclass
class SecurityCapabilities:
    tls_1_0: Optional[bool] = None
    tls_1_1: Optional[bool] = None
    tls_1_2: Optional[bool] = None
    onboard_key_generation: Optional[bool] = None
    access_policy_config: Optional[bool] = None
    x509_token: Optional[bool] = None
    saml_token: Optional[bool] = None
    kerberos_token: Optional[bool] = None
    rel_token: Optional[bool] = None
    dot1x: Optional[bool] = None
    supported_eap_method: Optional[int] = None
    remote_user_handling: Optional[bool] = None

@dataclass
class DeviceCapabilities:       
    xaddr: Optional[str] = None
    network: NetworkCapabilities = field(default_factory=NetworkCapabilities)
    system: SystemCapabilities = field(default_factory=SystemCapabilities)
    io: IOCapabilities = field(default_factory=IOCapabilities)
    security: SecurityCapabilities = field(default_factory=SecurityCapabilities)

@dataclass
class EventsCapabilities:
    xaddr: Optional[str] = None
    ws_subscription_policy_support: Optional[bool] = None
    ws_pull_point_support: Optional[bool] = None
    ws_pausable_subscription_manager_interface_support: Optional[bool] = None
    service_capabilities: EventServiceCapabilities = field(default_factory=EventServiceCapabilities)

@dataclass
class ImagingCapabilities:
    xaddr: Optional[str] = None

@dataclass
class StreamingCapabilities:
    rtp_multicast: Optional[bool] = None
    rtp_tcp: Optional[bool] = None
    rtp_rtsp_tcp: Optional[bool] = None

@dataclass
class MediaCapabilities:
    xaddr: Optional[str] = None
    streaming: StreamingCapabilities = field(default_factory=StreamingCapabilities)
    maximum_number_of_profiles: Optional[int] = None

@dataclass
class PTZCapabilities:
    xaddr: Optional[str] = None

@dataclass
class DeviceIOCapabilities:
    xaddr: Optional[str] = None
    video_sources: Optional[int] = None
    video_outputs: Optional[int] = None
    audio_sources: Optional[int] = None
    audio_outputs: Optional[int] = None
    relay_outputs: Optional[int] = None
    service_capabilities: DeviceIOServiceCapabilities = field(default_factory=DeviceIOServiceCapabilities) 

@dataclass
class TelexCapabilities:
    xaddr: Optional[str] = None
    time_osd_support: Optional[bool] = None
    title_osd_support: Optional[bool] = None
    ptz_3d_zoom_support: Optional[bool] = None
    ptz_aux_switch_support: Optional[bool] = None
    motion_detector_support: Optional[bool] = None
    tamper_detector_support: Optional[bool] = None

@dataclass
class Capabilities:
    analytics: Optional[AnalyticsCapabilities] = None
    device: Optional[DeviceCapabilities] = None
    events: Optional[EventsCapabilities] = None
    imaging: Optional[ImagingCapabilities] = None
    media: Optional[MediaCapabilities] = None
    ptz: Optional[PTZCapabilities] = None
    device_io: Optional[DeviceIOCapabilities] = None
    telex: Optional[TelexCapabilities] = None

def parse_capabilities_response(xml: str) -> Capabilities:
    if not xml: return
    root = etree.fromstring(xml.encode('utf-8'))

    caps = root.find(".//tds:GetCapabilitiesResponse/tds:Capabilities", NS)
    if caps is None:
        raise ValueError("Could not find tds:GetCapabilitiesResponse/tds:Capabilities")

    result = Capabilities()

    analytics = caps.find("tt:Analytics", NS)
    if analytics is not None:
        result.analytics = AnalyticsCapabilities(
            xaddr=text(analytics, "tt:XAddr"),
            rule_support=bool_text(analytics, "tt:RuleSupport"),
            analytics_module_support=bool_text(analytics, "tt:AnalyticsModuleSupport"),
        )

    device = caps.find("tt:Device", NS)
    if device is not None:
        d = DeviceCapabilities(
            xaddr=text(device, "tt:XAddr"),
        )

        net = device.find("tt:Network", NS)
        if net is not None:
            d.network = NetworkCapabilities(
                ip_filter=bool_text(net, "tt:IPFilter"),
                zero_configuration=bool_text(net, "tt:ZeroConfiguration"),
                ip_version6=bool_text(net, "tt:IPVersion6"),
                dyn_dns=bool_text(net, "tt:DynDNS"),
                dot11_configuration=bool_text(
                    net, "tt:Extension/tt:Dot11Configuration"
                ),
            )

        system = device.find("tt:System", NS)
        if system is not None:
            versions: list[OnvifVersion] = []
            for ver in system.findall("tt:SupportedVersions", NS):
                major = int_text(ver, "tt:Major")
                minor = int_text(ver, "tt:Minor")
                if major is not None and minor is not None:
                    versions.append(OnvifVersion(major, minor))

            d.system = SystemCapabilities(
                discovery_resolve=bool_text(system, "tt:DiscoveryResolve"),
                discovery_bye=bool_text(system, "tt:DiscoveryBye"),
                remote_discovery=bool_text(system, "tt:RemoteDiscovery"),
                system_backup=bool_text(system, "tt:SystemBackup"),
                system_logging=bool_text(system, "tt:SystemLogging"),
                firmware_upgrade=bool_text(system, "tt:FirmwareUpgrade"),
                supported_versions=versions,
                http_firmware_upgrade=bool_text(
                    system, "tt:Extension/tt:HttpFirmwareUpgrade"
                ),
                http_system_backup=bool_text(
                    system, "tt:Extension/tt:HttpSystemBackup"
                ),
                http_system_logging=bool_text(
                    system, "tt:Extension/tt:HttpSystemLogging"
                ),
                http_support_information=bool_text(
                    system, "tt:Extension/tt:HttpSupportInformation"
                ),
            )

        io = device.find("tt:IO", NS)
        if io is not None:
            d.io = IOCapabilities(
                input_connectors=int_text(io, "tt:InputConnectors"),
                relay_outputs=int_text(io, "tt:RelayOutputs"),
                auxiliary=bool_text(io, "tt:Extension/tt:Auxiliary"),
            )

        sec = device.find("tt:Security", NS)
        if sec is not None:
            d.security = SecurityCapabilities(
                tls_1_1=bool_text(sec, "tt:TLS1.1"),
                tls_1_2=bool_text(sec, "tt:TLS1.2"),
                onboard_key_generation=bool_text(sec, "tt:OnboardKeyGeneration"),
                access_policy_config=bool_text(sec, "tt:AccessPolicyConfig"),
                x509_token=bool_text(sec, "tt:X.509Token"),
                saml_token=bool_text(sec, "tt:SAMLToken"),
                kerberos_token=bool_text(sec, "tt:KerberosToken"),
                rel_token=bool_text(sec, "tt:RELToken"),
                tls_1_0=bool_text(sec, "tt:Extension/tt:TLS1.0"),
                dot1x=bool_text(sec, "tt:Extension/tt:Extension/tt:Dot1X"),
                supported_eap_method=int_text(
                    sec, "tt:Extension/tt:Extension/tt:SupportedEAPMethod"
                ),
                remote_user_handling=bool_text(
                    sec, "tt:Extension/tt:Extension/tt:RemoteUserHandling"
                ),
            )

        result.device = d

    events = caps.find("tt:Events", NS)
    if events is not None:
        result.events = EventsCapabilities(
            xaddr=text(events, "tt:XAddr"),
            ws_subscription_policy_support=bool_text(
                events, "tt:WSSubscriptionPolicySupport"
            ),
            ws_pull_point_support=bool_text(events, "tt:WSPullPointSupport"),
            ws_pausable_subscription_manager_interface_support=bool_text(
                events, "tt:WSPausableSubscriptionManagerInterfaceSupport"
            ),
        )

    imaging = caps.find("tt:Imaging", NS)
    if imaging is not None:
        result.imaging = ImagingCapabilities(
            xaddr=text(imaging, "tt:XAddr"),
        )

    media = caps.find("tt:Media", NS)
    if media is not None:
        result.media = MediaCapabilities(
            xaddr=text(media, "tt:XAddr"),
            streaming=StreamingCapabilities(
                rtp_multicast=bool_text(
                    media, "tt:StreamingCapabilities/tt:RTPMulticast"
                ),
                rtp_tcp=bool_text(
                    media, "tt:StreamingCapabilities/tt:RTP_TCP"
                ),
                rtp_rtsp_tcp=bool_text(
                    media, "tt:StreamingCapabilities/tt:RTP_RTSP_TCP"
                ),
            ),
            maximum_number_of_profiles=int_text(
                media,
                "tt:Extension/tt:ProfileCapabilities/tt:MaximumNumberOfProfiles",
            ),
        )

    ptz = caps.find("tt:PTZ", NS)
    if ptz is not None:
        result.ptz = PTZCapabilities(
            xaddr=text(ptz, "tt:XAddr"),
        )

    device_io = caps.find("tt:Extension/tt:DeviceIO", NS)
    if device_io is not None:
        result.device_io = DeviceIOCapabilities(
            xaddr=text(device_io, "tt:XAddr"),
            video_sources=int_text(device_io, "tt:VideoSources"),
            video_outputs=int_text(device_io, "tt:VideoOutputs"),
            audio_sources=int_text(device_io, "tt:AudioSources"),
            audio_outputs=int_text(device_io, "tt:AudioOutputs"),
            relay_outputs=int_text(device_io, "tt:RelayOutputs"),
        )

    telex = caps.find("tt:Extension/tt:Extensions/tt:TelexCapabilities", NS)
    if telex is not None:
        result.telex = TelexCapabilities(
            xaddr=text(telex, "tt:XAddr"),
            time_osd_support=bool_text(telex, "tt:TimeOSDSupport"),
            title_osd_support=bool_text(telex, "tt:TitleOSDSupport"),
            ptz_3d_zoom_support=bool_text(telex, "tt:PTZ3DZoomSupport"),
            ptz_aux_switch_support=bool_text(telex, "tt:PTZAuxSwitchSupport"),
            motion_detector_support=bool_text(telex, "tt:MotionDetectorSupport"),
            tamper_detector_support=bool_text(telex, "tt:TamperDetectorSupport"),
        )

    return result
