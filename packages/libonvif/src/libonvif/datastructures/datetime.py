from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
from lxml import etree
from libonvif.utils.xml import text, int_text, bool_text, NS

@dataclass
class NTPInformation:
    from_dhcp: Optional[bool] = None
    ntp_from_dhcp: list[str] = field(default_factory=list)
    ntp_manual: list[str] = field(default_factory=list)

@dataclass
class Time:
    hour: Optional[int] = None
    minute: Optional[int] = None
    second: Optional[int] = None

@dataclass
class Date:
    year: Optional[int] = None
    month: Optional[int] = None
    day: Optional[int] = None

@dataclass
class DateTime:
    time: Time = field(default_factory=Time)
    date: Date = field(default_factory=Date)

@dataclass
class TimeZone:
    tz: Optional[str] = None

@dataclass
class SystemDateAndTime:
    date_time_type: Optional[str] = None
    daylight_savings: Optional[bool] = None
    time_zone: Optional[TimeZone] = None
    utc_date_time: Optional[DateTime] = None
    local_date_time: Optional[DateTime] = None

def parse_time(elem: Optional[etree._Element]) -> Time:
    if elem is None:
        return Time()

    return Time(
        hour=int_text(elem, "tt:Hour"),
        minute=int_text(elem, "tt:Minute"),
        second=int_text(elem, "tt:Second"),
    )

def parse_date(elem: Optional[etree._Element]) -> Date:
    if elem is None:
        return Date()

    return Date(
        year=int_text(elem, "tt:Year"),
        month=int_text(elem, "tt:Month"),
        day=int_text(elem, "tt:Day"),
    )

def parse_datetime(elem: Optional[etree._Element]) -> Optional[DateTime]:
    if elem is None:
        return None

    return DateTime(
        time=parse_time(elem.find("tt:Time", NS)),
        date=parse_date(elem.find("tt:Date", NS)),
    )

def parse_timezone(elem: Optional[etree._Element]) -> Optional[TimeZone]:
    if elem is None:
        return None

    return TimeZone(
        tz=text(elem, "tt:TZ"),
    )

def parse_system_date_and_time_response(xml: str) -> SystemDateAndTime:
    if not xml: return
    root = etree.fromstring(xml.encode('utf-8'))

    elem = root.find(
        ".//tds:GetSystemDateAndTimeResponse/tds:SystemDateAndTime",
        NS,
    )
    if elem is None:
        raise ValueError(
            "Could not find tds:GetSystemDateAndTimeResponse/tds:SystemDateAndTime"
        )

    return SystemDateAndTime(
        date_time_type=text(elem, "tt:DateTimeType"),
        daylight_savings=bool_text(elem, "tt:DaylightSavings"),
        time_zone=parse_timezone(elem.find("tt:TimeZone", NS)),
        utc_date_time=parse_datetime(elem.find("tt:UTCDateTime", NS)),
        local_date_time=parse_datetime(elem.find("tt:LocalDateTime", NS)),
    )

def parse_ip_address(elem):
    return (
        text(elem, "tt:IPv4Address")
        or text(elem, "tt:IPv6Address")
        or text(elem, "tt:DNSname") 
    )

def parse_ntp_response(xml: str) -> NTPInformation:
    if not xml: return
    root = etree.fromstring(xml.encode('utf-8'))

    ntp_elem = root.find(".//tds:NTPInformation", NS)
    if ntp_elem is None:
        raise ValueError("Missing NTPInformation")

    return NTPInformation(
        from_dhcp=bool_text(ntp_elem, "tt:FromDHCP"),
        ntp_from_dhcp=[
            parse_ip_address(e)
            for e in ntp_elem.findall("tt:NTPFromDHCP", NS)
        ],
        ntp_manual=[
            parse_ip_address(e)
            for e in ntp_elem.findall("tt:NTPManual", NS)
        ],
    )
