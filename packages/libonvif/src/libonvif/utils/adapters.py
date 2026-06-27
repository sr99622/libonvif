import psutil
import socket
import ipaddress

def find_adapters(include_virtual=False) -> list[str]:
    ips = []
    VIRTUAL_KEYWORDS = {'docker', 'veth', 'vboxnet', 'vmware', 'virtual', 'wsl'}
    stats = psutil.net_if_stats()
    for interface, addrs in psutil.net_if_addrs().items():
        if not (data := stats.get(interface)):
            continue
        if not data.isup:
            continue
        if not include_virtual:
            if any(keyword in interface.lower() for keyword in VIRTUAL_KEYWORDS):
                continue
        for addr in addrs:
            if addr.family == socket.AF_INET:
                p_obj = ipaddress.ip_address(addr.address)
                if p_obj.is_loopback or p_obj.is_link_local:
                    continue
                ips.append(addr.address)
    return ips
