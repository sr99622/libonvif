from __future__ import annotations

from dataclasses import dataclass, field
from textual.timer import Timer
from typing import Optional, Any
from lxml import etree
from enum import Enum
from libonvif.utils.xml import int_attr, attr, text_list, bool_attr, text, NS

WSTOP_NS = "http://docs.oasis-open.org/wsn/t-1"
WSTOP_TOPIC_ATTR = f"{{{WSTOP_NS}}}topic"

class SubscriptionType(Enum):
    PULL = 0
    PUSH = 1

@dataclass
class SimpleItem:
    name: Optional[str] = None
    value: Optional[str] = None

@dataclass
class EventMessage:
    utc_time: Optional[str] = None
    property_operation: Optional[str] = None
    source: list[SimpleItem] = field(default_factory=list)
    data: list[SimpleItem] = field(default_factory=list)

@dataclass
class NotificationMessage:
    topic: Optional[str] = None
    topic_dialect: Optional[str] = None
    message: EventMessage = field(default_factory=EventMessage)

@dataclass
class PullMessagesResponse:
    current_time: Optional[str] = None
    termination_time: Optional[str] = None
    notifications: list[NotificationMessage] = field(default_factory=list)

@dataclass
class EventServiceCapabilities:
    ws_subscription_policy_support: Optional[bool] = None
    ws_pausable_subscription_manager_interface_support: Optional[bool] = None
    max_notification_producers: Optional[int] = None
    max_pull_points: Optional[int] = None
    persistent_notification_storage: Optional[bool] = None
    event_broker_protocols: Optional[str] = None
    max_event_brokers: Optional[int] = None
    metadata_over_mqtt: Optional[bool] = None

@dataclass
class TopicNamespaceLocation:
    uri: Optional[str] = None

@dataclass
class SubscriptionReference:
    xaddr: Optional[str] = None
    event: Optional[str] = None
    subscription_type: Optional[SubscriptionType] = None
    termination_time: Optional[str] = None
    resubscribe_timer: Optional[Timer] = None

@dataclass
class EventProperties:
    topic_set: list[str] = field(default_factory=list)
    topic_namespace_location: list[str] = field(default_factory=list)
    message_content_filter_dialect: list[str] = field(default_factory=list)
    message_content_schema_location: list[str] = field(default_factory=list)
    fixed_topic_set: Optional[bool] = None
    producer_properties_filter_dialect: list[str] = field(default_factory=list)
    topic_expression_dialect: list[str] = field(default_factory=list)

def parse_event_service_capabilities_response(xml: str) -> EventServiceCapabilities:
    if not xml: return
    root = etree.fromstring(xml.encode('utf-8'))

    elem = root.find(
        ".//tev:GetServiceCapabilitiesResponse/tev:Capabilities",
        NS,
    )
    if elem is None:
        raise ValueError(
            "Could not find tev:GetServiceCapabilitiesResponse/tev:Capabilities"
        )

    return EventServiceCapabilities(
        ws_subscription_policy_support=bool_attr(
            elem, "WSSubscriptionPolicySupport"
        ),
        ws_pausable_subscription_manager_interface_support=bool_attr(
            elem, "WSPausableSubscriptionManagerInterfaceSupport"
        ),
        max_notification_producers=int_attr(
            elem, "MaxNotificationProducers"
        ),
        max_pull_points=int_attr(
            elem, "MaxPullPoints"
        ),
        persistent_notification_storage=bool_attr(
            elem, "PersistentNotificationStorage"
        ),
        event_broker_protocols=attr(
            elem, "EventBrokerProtocols"
        ),
        max_event_brokers=int_attr(
            elem, "MaxEventBrokers"
        ),
        metadata_over_mqtt=bool_attr(
            elem, "MetadataOverMQTT"
        ),
    )

def strip_ns(tag: str) -> str:
    return tag.split("}", 1)[-1] if "}" in tag else tag

def is_topic_node(elem: etree._Element) -> bool:
    return elem.attrib.get(WSTOP_TOPIC_ATTR) == "true"

def has_topic_child(elem: etree._Element) -> bool:
    for child in list(elem):
        if strip_ns(child.tag) == "MessageDescription":
            continue
        if is_topic_node(child):
            return True
        if has_topic_child(child):
            return True
    return False

def collect_topic_paths(elem: etree._Element, prefix: str = "") -> list[str]:
    topics: list[str] = []
    for child in list(elem):
        name = strip_ns(child.tag)
        if name == "MessageDescription":
            continue
        path = f"{prefix}/{name}" if prefix else name
        if is_topic_node(child) and not has_topic_child(child):
            topics.append(path)
        topics.extend(collect_topic_paths(child, path))
    return topics

def parse_topic_set(elem: Optional[etree._Element]) -> list[str]:
    if elem is None:
        return []
    return collect_topic_paths(elem)

def parse_event_properties_response(xml: str) -> EventProperties:
    if not xml: return
    root = etree.fromstring(xml.encode('utf-8'))
    elem = root.find(
        ".//tev:GetEventPropertiesResponse",
        NS,
    )
    if elem is None:
        raise ValueError("Could not find tev:GetEventPropertiesResponse")

    return EventProperties(
        topic_namespace_location=text_list(
            elem,
            "tev:TopicNamespaceLocation",
        ),
        topic_set=parse_topic_set(
            elem.find("wstop:TopicSet", NS),
        ),
        message_content_filter_dialect=text_list(
            elem,
            "tev:MessageContentFilterDialect",
        ),
        message_content_schema_location=text_list(
            elem,
            "tev:MessageContentSchemaLocation",
        ),
        fixed_topic_set=bool_attr(
            elem,
            "tev:FixedTopicSet",
        ),
        producer_properties_filter_dialect=text_list(
            elem,
            "tev:ProducerPropertiesFilterDialect",
        ),
        topic_expression_dialect=text_list(
            elem,
            "tev:TopicExpressionDialect",
        ),
    )

def parse_simple_items(parent, xpath: str) -> list[SimpleItem]:
    return [
        SimpleItem(
            name=item.get("Name"),
            value=item.get("Value"),
        )
        for item in parent.xpath(xpath, namespaces=NS)
    ]


def parse_pull_messages_response(xml: str) -> PullMessagesResponse:
    if not xml: return
    root = etree.fromstring(xml.encode('utf-8'))

    response_el = root.xpath(".//tev:PullMessagesResponse", namespaces=NS)
    if not response_el:
        return PullMessagesResponse()

    response_el = response_el[0]

    response = PullMessagesResponse(
        current_time=text(response_el, "./tev:CurrentTime"),
        termination_time=text(response_el, "./tev:TerminationTime"),
    )

    for notification_el in response_el.xpath("./wsnt:NotificationMessage", namespaces=NS):
        topic_el = notification_el.xpath("./wsnt:Topic", namespaces=NS)
        topic = topic_el[0].text.strip() if topic_el and topic_el[0].text else None
        topic_dialect = topic_el[0].get("Dialect") if topic_el else None

        message_el = notification_el.xpath("./wsnt:Message/tt:Message", namespaces=NS)

        event_message = EventMessage()

        if message_el:
            msg = message_el[0]

            event_message = EventMessage(
                utc_time=msg.get("UtcTime"),
                property_operation=msg.get("PropertyOperation"),
                source=parse_simple_items(msg, "./tt:Source/tt:SimpleItem"),
                data=parse_simple_items(msg, "./tt:Data/tt:SimpleItem"),
            )

        response.notifications.append(
            NotificationMessage(
                topic=topic,
                topic_dialect=topic_dialect,
                message=event_message,
            )
        )

    return response

def strip_topic_prefix(topic: str) -> str:
    if ":" in topic:
        return topic.split(":", 1)[1]
    return topic

def simple_items(elem: etree._Element | None) -> dict[str, str]:
    if elem is None:
        return {}

    return {
        item.attrib.get("Name", ""): item.attrib.get("Value", "")
        for item in elem.findall("tt:SimpleItem", NS)
        if item.attrib.get("Name")
    }

def parse_notify(ip_address: str, xml: str) -> list[dict[str, Any]]:
    root = etree.fromstring(xml.encode("utf-8"))
    output = []
    for msg in root.findall(".//wsnt:NotificationMessage", NS):
        topic = text(msg, "wsnt:Topic")
        topic = strip_topic_prefix(topic) if topic else None
        alarm = {"ip_address": ip_address, "topic": topic}
        message = msg.find("wsnt:Message/tt:Message", NS)
        if message is not None:
            alarm["utc_time"] = message.attrib.get("UtcTime")
            alarm["operation"] = message.attrib.get("PropertyOperation")
            alarm["source"] = simple_items(message.find("tt:Source", NS))
            alarm["data"] = simple_items(message.find("tt:Data", NS)) 
            output.append(alarm)
    return output
