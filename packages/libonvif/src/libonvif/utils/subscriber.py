import traceback
from datetime import datetime, timezone
from libonvif.utils.xml import get_xml_value
from urllib.parse import urlparse
import ipaddress
import socket
from threading import Timer, RLock
import psutil
from libonvif.devices.camera import Camera, unsubscribe, subscribe_event, \
    create_pull_point_subscription, create_pull_point_subscriptions, \
    subscribe_events
from libonvif.datastructures.event import SubscriptionReference, SubscriptionType

RESUBSCRIBE_MARGIN_SECONDS = 10

class SubscriptionManager:
    def __init__(self, ip_address: str):
        self.ip_address = ip_address
        self.event_server = None
        self.subscription_lock = RLock()

    def find_local_subnet_matches(self, remote_target_ip: str) -> str:
        target = ipaddress.IPv4Address(remote_target_ip)

        local_addresses = []
        for interface, addrs in psutil.net_if_addrs().items():
            for addr in addrs:
                # Look only for active IPv4 configurations with a valid netmask
                if addr.family == socket.AF_INET and addr.netmask:
                    p_obj = ipaddress.ip_address(addr.address)
                    if p_obj.is_loopback or p_obj.is_link_local:
                        continue
                    try:
                        network = ipaddress.IPv4Interface(f"{addr.address}/{addr.netmask}").network
                        if target in network:
                            #print(f"Match found! {remote_target_ip} is on the same subnet as interface '{interface}' ({addr.address})")
                            return addr.address
                        local_addresses.append(addr.address)
                    except ValueError:
                        continue

        # did not find local subnet match, return first non loopback, non link-local address if available
        if len(local_addresses):
            return local_addresses[0]

    def unsubscribe_events(self, camera: Camera):
        try:
            for reference in camera.subscription_references:
                if reference.resubscribe_timer: reference.resubscribe_timer.cancel()
                unsubscribe(camera, reference.xaddr)
            camera.subscription_references.clear()
        except Exception as ex:
            if camera.on_error:
                camera.on_error(f"unsubscribe events error: {ex}")
            else:
                print(traceback.format_exc(), flush=True)

    def schedule_resubscribe_event(self, camera: Camera, server_ip_address: str, port: int, delay: float, event: str | None) -> Timer:
        timer = Timer(
            max(1.0, delay),
            lambda: self.subscribe_push_event(camera, server_ip_address, port, event),
        )
        timer.daemon = True
        timer.start()
        return timer
 
    def subscribe_push_event(self, camera: Camera, server_ip_address: str, port: int, event: str | None = None) -> None:
        try:
            ip_obj = ipaddress.ip_address(urlparse(camera.xaddr).hostname)
            event_callback_address = server_ip_address
            if event_callback_address == "0.0.0.0":
                event_callback_address = self.find_local_subnet_matches(ip_obj)

            xml = subscribe_event(camera, event_callback_address, port, event)
            subscription_reference = get_xml_value(xml, "//s:Body//wsnt:SubscribeResponse//wsnt:SubscriptionReference//wsa:Address")
            termination_time = get_xml_value(xml, "//s:Body//wsnt:TerminationTime")
            dt = datetime.fromisoformat(termination_time.replace("Z", "+00:00"))
            delay = (dt - datetime.now(timezone.utc)).total_seconds() - camera.time_offset - RESUBSCRIBE_MARGIN_SECONDS
            resubscribe_timer = self.schedule_resubscribe_event(camera, server_ip_address, port, delay, event)

            with self.subscription_lock:
                camera.subscription_references = [ref for ref in camera.subscription_references if ref.event != event]
                reference = SubscriptionReference(
                    xaddr=subscription_reference, 
                    event=event, 
                    subscription_type=SubscriptionType.PUSH,
                    termination_time=termination_time,
                    resubscribe_timer=resubscribe_timer
                )
                camera.subscription_references.append(reference)

        except Exception as ex:
            if camera.on_error: 
                camera.on_error(f"resubscribe event error: {ex}")
            else:
                print(traceback.format_exc(), flush=True)

    def schedule_resubscribe_events(self, camera: Camera, server_ip_address: str, port: int, delay: float, events: list[str]) -> Timer:
        timer = Timer(
            max(1.0, delay),
            lambda: self.subscribe_push_events(camera, server_ip_address, port, events),
        )
        timer.daemon = True
        timer.start()
        return timer
 
    def subscribe_push_events(self, camera: Camera, server_ip_address: str, port: int, events: list[str]) -> None:
        try:
            ip_obj = ipaddress.ip_address(urlparse(camera.xaddr).hostname)
            event_callback_address = server_ip_address
            if event_callback_address == "0.0.0.0":
                event_callback_address = self.find_local_subnet_matches(ip_obj)

            xml = subscribe_events(camera, event_callback_address, port, events)
            subscription_reference = get_xml_value(xml, "//s:Body//wsnt:SubscribeResponse//wsnt:SubscriptionReference//wsa:Address")
            termination_time = get_xml_value(xml, "//s:Body//wsnt:TerminationTime")
            dt = datetime.fromisoformat(termination_time.replace("Z", "+00:00"))
            delay = (dt - datetime.now(timezone.utc)).total_seconds() - camera.time_offset - RESUBSCRIBE_MARGIN_SECONDS
            resubscribe_timer = self.schedule_resubscribe_events(camera, server_ip_address, port, delay, events)

            with self.subscription_lock:
                camera.subscription_references = [ref for ref in camera.subscription_references if ref.event != events]
                reference = SubscriptionReference(
                    xaddr=subscription_reference, 
                    event=events, 
                    subscription_type=SubscriptionType.PUSH,
                    termination_time=termination_time,
                    resubscribe_timer=resubscribe_timer
                )
                camera.subscription_references.append(reference)

        except Exception as ex:
            if camera.on_error:
                camera.on_error(f"resubscribe event error: {ex}")
            else:
                print(traceback.format_exc(), flush=True)

    def subscribe_pull_event(self, camera: Camera, event: str | None = None) -> None:
        try:
            xml = create_pull_point_subscription(camera, event)
            address = get_xml_value(xml, ".//tev:CreatePullPointSubscriptionResponse/tev:SubscriptionReference/wsa5:Address")
            termination_time = get_xml_value(xml, ".//tev:CreatePullPointSubscriptionResponse/wsnt:TerminationTime")
            reference = SubscriptionReference(
                xaddr=address,
                subscription_type=SubscriptionType.PULL,
                termination_time=termination_time
            )
            camera.subscription_references.append(reference)
        except Exception as ex:
            if camera.on_error:
                camera.on_error(f"subscribe pull event error: {ex}")
            else:
                print(traceback.format_exc(), flush=True)
    
    def subscribe_pull_events(self, camera: Camera, events: list[str]) -> None:
        try:
            xml = create_pull_point_subscriptions(camera, events)
            address = get_xml_value(xml, ".//tev:CreatePullPointSubscriptionResponse/tev:SubscriptionReference/wsa5:Address")
            termination_time = get_xml_value(xml, ".//tev:CreatePullPointSubscriptionResponse/wsnt:TerminationTime")
            reference = SubscriptionReference(
                xaddr=address,
                subscription_type=SubscriptionType.PULL,
                termination_time=termination_time
            )
            camera.subscription_references.append(reference)
        except Exception as ex:
            if camera.on_error:
                camera.on_error(f"subscribe pull events error: {ex}")
            else:
                print(traceback.format_exc(), flush=True)
