from __future__ import annotations

from typing import Optional
from lxml import etree

NS = {
    "s": "http://www.w3.org/2003/05/soap-envelope",
    "trt": "http://www.onvif.org/ver10/media/wsdl",
    "tt": "http://www.onvif.org/ver10/schema",
    "tds": "http://www.onvif.org/ver10/device/wsdl",
    "timg": "http://www.onvif.org/ver20/imaging/wsdl",
    "wsa5": "http://www.w3.org/2005/08/addressing",
    "wsnt": "http://docs.oasis-open.org/wsn/b-2",
    "d": "http://schemas.xmlsoap.org/ws/2005/04/discovery",
    "tmd": "http://www.onvif.org/ver10/deviceIO/wsdl",
    "ter": "http://www.onvif.org/ver10/error",
    "a": "http://schemas.xmlsoap.org/ws/2004/08/addressing",
    "wsse": "http://docs.oasis-open.org/wss/2004/01/oasis-200401-wss-wssecurity-secext-1.0.xsd",
    "wsu": "http://docs.oasis-open.org/wss/2004/01/oasis-200401-wss-wssecurity-utility-1.0.xsd",
    "tptz": "http://www.onvif.org/ver20/ptz/wsdl",
    "tev": "http://www.onvif.org/ver10/events/wsdl",
    "wstop": "http://docs.oasis-open.org/wsn/t-1",
    "wsa": "http://www.w3.org/2005/08/addressing",
    "tns1": "http://www.onvif.org/ver10/topics",
}

def attr(elem: etree._Element, name: str) -> Optional[str]:
    if elem is None: return
    return elem.attrib.get(name)

def text(elem: etree._Element, path: str) -> Optional[str]:
    try:
        if not (result := elem.xpath(path, namespaces=NS)): return
        found = result[0]
        if isinstance(found, etree._Element):
            return "".join(found.itertext()).strip()
        return str(found).strip()
    except Exception as ex:
        print(f"text parsing exception: {ex}")

def text_list(elem: etree._Element, path: str) -> list[str]:
    values: list[str] = []
    try:
        items = elem.xpath(path, namespaces=NS)
        for item in items:
            if isinstance(item, etree._Element):
                value = "".join(item.itertext()).strip()
            else:
                value = str(item).strip()
            if value:
                values.append(value)
    except Exception as ex:
        print(f"text_list exception: {ex}")
    return values

def bool_text(elem: etree._Element, path: str) -> Optional[bool]:
    if not (value := text(elem, path)): return
    return value.strip().lower() in ("true", "1", "yes", "on")

def bool_attr(elem: etree._Element, name: str) -> Optional[bool]:
    if not (value := attr(elem, name)): return
    return value.strip().lower() in ("true", "1", "yes", "on")

def int_text(elem: etree._Element, path: str) -> Optional[int]:
    if not (value := text(elem, path)): return
    return int(value)

def int_attr(elem: etree._Element, name: str) -> Optional[int]:
    if not (value := attr(elem, name)): return
    return int(value)

def float_text(elem: etree._Element, path: str) -> Optional[float]:
    if not (value := text(elem, path)): return
    return float(value)

def get_xml_value(xml: str, xpath: str) -> str:
    if not xml: raise ValueError("The provided xml is None")
    try:
        if (doc := etree.fromstring(xml.encode('utf-8'))) is None: raise ValueError("Invalid xml")
        if (found := doc.xpath(xpath, namespaces=NS)[0]) is None: raise ValueError("Invalid xpath")
        if isinstance(found, etree._Element):
            return "".join(found.itertext()).strip()
        return str(found).strip()
    except Exception as ex:
        print(f'get_xml_value exception: {ex}')
    return ""

