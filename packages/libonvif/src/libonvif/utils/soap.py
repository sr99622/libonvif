from __future__ import annotations

import os
import base64
import hashlib
import niquests as requests
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field
from typing import Optional
from lxml import etree
from .xml import NS

POST_TIMEOUT = 10

@dataclass
class SoapFault:
    code: Optional[str] = None
    subcodes: list[str] = field(default_factory=list)
    reason: Optional[str] = None
    detail: Optional[str] = None

    def __repr__(self) -> str:
        return (
            "SoapFault(\n"
            f"    code={self.code!r},\n"
            f"    subcodes={self.subcodes!r},\n"
            f"    reason={self.reason!r},\n"
            f"    detail={self.detail!r}\n"
            ")"
        )

def parse_soap_fault(xml_text: str) -> Optional[SoapFault]:
    root = etree.fromstring(xml_text.encode("utf-8"))

    fault = root.find(".//s:Fault", namespaces=NS)
    if fault is None:
        return None

    out = SoapFault()

    code_value = fault.find(".//s:Code//s:Value", namespaces=NS)
    if code_value is not None and code_value.text:
        out.code = code_value.text.strip()

    subcode = fault.find(".//s:Code//s:Subcode", namespaces=NS)
    while subcode is not None:
        value = subcode.find(".//s:Value", namespaces=NS)
        if value is not None and value.text:
            out.subcodes.append(value.text.strip())
        subcode = subcode.find(".//s:Subcode", namespaces=NS)

    reason = fault.find(".//s:Reason//s:Text", namespaces=NS)
    if reason is not None and reason.text:
        out.reason = reason.text.strip()

    detail = fault.find(".//s:Detail", namespaces=NS)
    if detail is not None:
        out.detail = "".join(detail.itertext()).strip() or None

    return out

def create_wsse_header_data(password: str, offset_seconds: int) -> tuple[str, str, str]:
    nonce_raw = os.urandom(20)
    nonce_b64 = base64.b64encode(nonce_raw).decode("ascii")
    created_dt = datetime.now(timezone.utc) + timedelta(seconds=offset_seconds)
    created = created_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    digest_raw = hashlib.sha1(nonce_raw + created.encode("utf-8") + password.encode("utf-8")).digest()
    password_digest = base64.b64encode(digest_raw).decode("ascii")
    return password_digest, nonce_b64, created

def build_wsse_header(username: str, password: str, time_offset: int) -> str:
    password_digest, nonce, created = create_wsse_header_data(password, time_offset)

    return f"""
<s:Header>
  <wsse:Security s:mustUnderstand="1">
    <wsse:UsernameToken>
      <wsse:Username>{username}</wsse:Username>
      <wsse:Password Type="http://docs.oasis-open.org/wss/2004/01/oasis-200401-wss-username-token-profile-1.0#PasswordDigest">{password_digest}</wsse:Password>
      <wsse:Nonce EncodingType="http://docs.oasis-open.org/wss/2004/01/oasis-200401-wss-soap-message-security-1.0#Base64Binary">{nonce}</wsse:Nonce>
      <wsu:Created>{created}</wsu:Created>
    </wsse:UsernameToken>
  </wsse:Security>
</s:Header>
""".strip()

def build_soap_envelope(body: str, username: str, password: str, time_offset: int) -> str:
    header = build_wsse_header(username, password, time_offset)
    args = []
    for key in NS:
        args.append(f'xmlns:{key}=\"{NS[key]}\"')
    namespace = " ".join(args)
    return f"""<s:Envelope {namespace}>{header}<s:Body>{body}</s:Body></s:Envelope>"""

def onvif_post(url: str, body: str, username: str, password: str, time_offset: int) -> str:
    if os.environ.get("LIBONVIF_VERBOSE"):
        print(f"XML Input:\n{body}\n")
    soap = build_soap_envelope(body, username, password, time_offset)
    headers = {
        "User-Agent": "Generic",
        "Connection": "Close",
        "Accept-Encoding": "gzip, deflate",
        "Content-Type": "application/soap+xml; charset=utf-8",
    }
    response = requests.post(url, data=soap, headers=headers, timeout=POST_TIMEOUT)
    if os.environ.get("LIBONVIF_VERBOSE"):
        print(f"Camera Response:\n{response.text}\n")
    fault = parse_soap_fault(response.text)
    if fault:
        raise ValueError(f"Input:\n{body}\n\nFault:\n{fault}\n\nURL:\n{url}")
    response.raise_for_status()
    output = response.text
    return output
