from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
from lxml import etree
from libonvif.utils.xml import int_attr, text, text_list, bool_text, NS

@dataclass
class RelayOutputOptions:
    token: Optional[str] = None
    modes: list[str] = field(default_factory=list)
    delay_times: list[str] = field(default_factory=list)
    discrete: Optional[bool] = None

@dataclass
class RelayOutputProperties:
    mode: Optional[str] = None
    delay_time: Optional[str] = None
    idle_state: Optional[str] = None

@dataclass
class RelayOutput:
    token: Optional[str] = None
    properties: RelayOutputProperties = field(default_factory=RelayOutputProperties)
    options: RelayOutputOptions = field(default_factory=RelayOutputOptions)

@dataclass
class DeviceIOServiceCapabilities:
    video_sources: Optional[int] = None
    video_outputs: Optional[int] = None
    audio_sources: Optional[int] = None
    audio_outputs: Optional[int] = None
    relay_outputs: Optional[int] = None
    serial_ports: Optional[int] = None
    digital_inputs: Optional[int] = None


def parse_deviceio_service_capabilities_response(xml: str) -> Optional[DeviceIOServiceCapabilities]:
    if not xml: return
    root = etree.fromstring(xml.encode("utf-8"))

    result = root.xpath(
        ".//tmd:GetServiceCapabilitiesResponse/tmd:Capabilities",
        namespaces=NS,
    )

    if not result: return

    cap = result[0]

    return DeviceIOServiceCapabilities(
        video_sources=int_attr(cap, "VideoSources"),
        video_outputs=int_attr(cap, "VideoOutputs"),
        audio_sources=int_attr(cap, "AudioSources"),
        audio_outputs=int_attr(cap, "AudioOutputs"),
        relay_outputs=int_attr(cap, "RelayOutputs"),
        serial_ports=int_attr(cap, "SerialPorts"),
        digital_inputs=int_attr(cap, "DigitalInputs"),
    )

def parse_get_relay_outputs_response(xml: str) -> list[RelayOutput]:
    if not xml:
        return []

    root = etree.fromstring(xml.encode("utf-8"))

    outputs: list[RelayOutput] = []

    for output_el in root.xpath(
        ".//*[local-name()='GetRelayOutputsResponse']/*[local-name()='RelayOutputs']"
    ):
        outputs.append(
            RelayOutput(
                token=output_el.get("token"),
                properties=RelayOutputProperties(
                    mode=text(output_el, "./tt:Properties/tt:Mode"),
                    delay_time=text(output_el, "./tt:Properties/tt:DelayTime"),
                    idle_state=text(output_el, "./tt:Properties/tt:IdleState"),
                ),
            )
        )

    return outputs

def parse_get_relay_output_options_response(xml: str) -> RelayOutputOptions:
    if not xml:
        return RelayOutputOptions()

    root = etree.fromstring(xml.encode("utf-8"))

    result = root.xpath(
        ".//tmd:GetRelayOutputOptionsResponse/tmd:RelayOutputOptions",
        namespaces=NS,
    )

    if not result:
        return RelayOutputOptions()

    options_el = result[0]

    delay_time_text = text(options_el, "./tmd:DelayTimes")

    return RelayOutputOptions(
        token=options_el.get("token"),
        modes=text_list(options_el, "./tmd:Mode"),
        delay_times=delay_time_text.split() if delay_time_text else [],
        discrete=bool_text(options_el, "./tmd:Discrete"),
    )

