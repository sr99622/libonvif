from textual.app import App, ComposeResult
from textual.containers import Horizontal
from textual.widgets import Header, Footer, Input, RichLog
from textual.widgets.tree import TreeNode
from textual.binding import Binding
from textual import events
from libonvif.utils.xml import get_xml_value
from libonvif.utils.soap import onvif_post
from libonvif.utils.adapters import find_adapters
from libonvif.utils.server import EventServer
from libonvif.utils.subscriber import SubscriptionManager
from .fields import UNUSED_FIELDS, HIDDEN_FIELDS, resolve_fqn_owner, \
        convert_string_value, is_editable_field, normalize_fqn, analyze_field_type
from libonvif.devices.camera import Camera, \
        discover, set_network_default_gateway, set_hostname_from_dhcp, \
        set_hostname, set_dns, set_ntp, set_network_interfaces, reboot, set_imaging_settings, \
        set_audio_encoder_configuration, set_video_encoder_configuration, \
        unsubscribe, get_status, continuous_move, move_stop, set_preset, \
        remove_preset, goto_preset, operate_preset_tour, remove_preset_tour, create_preset_tour, \
        parse_get_preset_tours_response, modify_preset_tour, pull_messages, \
        set_relay_output_settings, set_relay_output_state, get_local_date_and_time, \
        set_system_date_and_time, get_time_offset, get_local_date_and_time_as_utc, \
        start_multicast_streaming, stop_multicast_streaming, find_camera_manually
from libonvif.datastructures.event import SubscriptionType, SubscriptionReference, \
        parse_notify
from libonvif.datastructures.ptz import TourSpot, parse_get_presets_response
import traceback
import argparse
from .camera_tree import CameraTree
import re
from threading import RLock
from datetime import datetime, timezone
import sys



from libonvif.datastructures.datetime import Date, DateTime, SystemDateAndTime,  NTPInformation, Time, TimeZone, \
        parse_system_date_and_time_response, parse_ntp_response
import time

PORT = 8856

class ObjectBrowser(App):

    TITLE="Onvif TUI"

    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__()
        self.ips = find_adapters()
        self.ip_address = args.ip_address
        if sys.platform == "win32" and self.ip_address == "0.0.0.0" and len(self.ips):
            self.ip_address = self.ips[0]
        self.manual = args.manual
        self.username = args.username
        self.password = args.password
        self.subscription_lock = RLock()
        self.event_server = None
        self.subscription_manager = SubscriptionManager(self.ip_address)

    BINDINGS = [
        ("q", "quit", "Quit"),
        Binding("f2", "edit_selected", "Edit"),
        Binding("escape", "cancel_edit", "Cancel"),
    ]

    CSS = """
    #main {
        height: 1fr;
    }

    CameraTree {
        width: 50%;
        height: 1fr;
        border: solid green;
        padding: 1 2;
    }

    #debug_log {
        width: 50%;
        height: 1fr;
        border: solid blue;
        padding: 1;
    }

    #edit_box {
        dock: bottom;
        height: 3;
        border: solid yellow;
        padding: 0 1;
    }

    #confirm_dialog {
        width: 50;
        height: auto;
        border: solid red;
        padding: 1 2;
        background: $surface;
    }

    .hidden {
        display: none;
    }
    """
 
    def update_tree_time(self, camera: Camera, node: TreeNode) -> None:
        expanded = self.camera_tree.capture_expanded_nodes(node)
        node.remove_children()
        self.camera_tree._add_value(node, "date_time_type", camera.system_date_and_time.date_time_type, camera)
        self.camera_tree._add_value(node, "daylight_savings", camera.system_date_and_time.daylight_savings, camera)
        self.camera_tree._add_value(node, "time_zone", camera.system_date_and_time.time_zone, camera)
        self.camera_tree._add_value(node, "utc_date_time", camera.system_date_and_time.utc_date_time, camera)
        self.camera_tree._add_value(node, "local_date_time", camera.system_date_and_time.local_date_time, camera)
        self.camera_tree.restore_expanded_nodes(node, expanded)

    def show_system_date_and_time(self, camera: Camera) -> None:
        c = camera.system_date_and_time
        u = c.utc_date_time
        l = c.local_date_time

        local = ""
        if l:
            local = f"local date time: {l.date.year}-{l.date.month:02}-{l.date.day:02} {l.time.hour:02}:{l.time.minute:02}:{l.time.second:02}"

        time_str = f"""
Camera Time:

time zone: {c.time_zone.tz} 
daylight savings: {c.daylight_savings}
date time type: {c.date_time_type}
utc date time: {u.date.year}-{u.date.month:02}-{u.date.day:02} {u.time.hour:02}:{u.time.minute:02}:{u.time.second:02} 
{local} 
"""
        self.debug_log.write(time_str)

    def on_key(self, event: events.Key) -> None:
        node = self.camera_tree.cursor_node
        if not node.data:
            self.debug_log.write("No data associated with the selected node.") 
            return
        if not (camera := node.data.get("camera")):
            self.debug_log.write("No camera associated with the selected node.") 
            return
        if not len(camera.profiles): 
            if event.key in ('r', 's', 't', 'u', 'w', 'a', 'd', 'z'):
                self.debug_log.write("\nUnable to perform the requested action.\nThe selected camera has no profiles.")
            return
        
        profile_token = camera.profiles[0].token

        if node.label.plain == camera.name and event.key == 'r':
            try:
                xml = reboot(camera)
                msg = get_xml_value(xml, "//s:Body//tds:SystemRebootResponse//tds:Message")
                self.app.debug_log.write(f"{camera.name}: {msg}")
            except Exception as ex:
                self.app.debug_log.write(f"{ex}")
            return

        if not (fqn := node.data.get("fqn")): return
        normalized_fqn = normalize_fqn(fqn)
        #print(f"FQN: {fqn}, normalized: {fqn}")

        try:
            if normalized_fqn == "system_date_and_time":
                # system_date_and_time_management
                match event.key:
                    case 'u':
                        self.debug_log.write("synchronizing camera time to computer time as UTC ...")
                        # some kind of bug fouls this call when installed from pypi so it is done manually below
                        # set_system_date_and_time(camera, get_local_date_and_time_as_utc())

                        local_time = time.localtime()
                        sdt = SystemDateAndTime(
                            date_time_type="Manual",
                            daylight_savings=False,
                            time_zone=TimeZone("UTC0"),
                            local_date_time=DateTime(
                                date=Date(year=local_time.tm_year, month=local_time.tm_mon, day=local_time.tm_mday),
                                time=Time(hour=local_time.tm_hour, minute=local_time.tm_min, second=local_time.tm_sec)
                            ),
                            utc_date_time=DateTime(
                                date=Date(year=local_time.tm_year, month=local_time.tm_mon, day=local_time.tm_mday),
                                time=Time(hour=local_time.tm_hour, minute=local_time.tm_min, second=local_time.tm_sec)
                            )
                        )
                        set_system_date_and_time(camera, sdt)

                        get_time_offset(camera)
                        self.show_system_date_and_time(camera)
                        self.update_tree_time(camera, node)
                    case 't':
                        get_time_offset(camera)
                        self.show_system_date_and_time(camera)
                        self.update_tree_time(camera, node)
                    case 's':
                        self.debug_log.write("synchronizing camera time to computer time ...")
                        # some kind of bug fouls this call when installed from pypi so it is done manually below
                        # sdt = get_local_date_and_time()

                        ignore_dst = True
                        local_time = time.localtime()
                        utc_time = time.gmtime()
                        is_dst = False if ignore_dst else local_time.tm_isdst > 0
                        offset = -local_time.tm_gmtoff if ignore_dst else time.timezone
                        offset_hours = offset // 3600
                        offset_minutes = (offset % 3600) // 60
                        timezone = f"UTC{offset_hours:+03d}:{offset_minutes:02d}"

                        sdt = SystemDateAndTime(
                            date_time_type="Manual",
                            daylight_savings=is_dst,
                            time_zone=TimeZone(timezone),
                            local_date_time=DateTime(
                                date=Date(year=local_time.tm_year, month=local_time.tm_mon, day=local_time.tm_mday),
                                time=Time(hour=local_time.tm_hour, minute=local_time.tm_min, second=local_time.tm_sec)
                            ),
                            utc_date_time=DateTime(
                                date=Date(year=utc_time.tm_year, month=utc_time.tm_mon, day=utc_time.tm_mday),
                                time=Time(hour=utc_time.tm_hour, minute=utc_time.tm_min, second=utc_time.tm_sec)
                            )
                        )

                        set_system_date_and_time(camera, sdt)
                        get_time_offset(camera)
                        self.show_system_date_and_time(camera)
                        self.update_tree_time(camera, node)
                    case 'w':
                        if node.label.plain.endswith("(* modified)"):
                            sdt = get_local_date_and_time()
                            sdt.date_time_type = camera.system_date_and_time.date_time_type
                            sdt.daylight_savings = camera.system_date_and_time.daylight_savings
                            sdt.time_zone.tz = camera.system_date_and_time.time_zone.tz
                            set_system_date_and_time(camera, sdt)
                            get_time_offset(camera)
                            self.show_system_date_and_time(camera)
                            self.update_tree_time(camera, node)
                            node.set_label("system_date_and_time")
                            self.debug_log.write("\nsystem_date_and_time has been updated successfully")

                for child in node.parent.children:
                    if child.label.plain.startswith("time_offset:"):
                        child.set_label(f"time_offset: {camera.time_offset}")
                        break

            elif normalized_fqn == "profiles.[*]":
                # profile_multicast_streaming
                if not (found := re.fullmatch(r"profiles\.\[(\d+)\]", fqn)):
                    return
                index = int(found[1])
                profile_token = camera.profiles[index].token
                match event.key:
                    case 's':
                        start_multicast_streaming(camera, profile_token)
                        self.debug_log.write("Multicast streaming started")
                    case 't':
                        stop_multicast_streaming(camera, profile_token)
                        self.debug_log.write("Multicast streaming stopped")

            elif normalized_fqn == "event_properties.topic_set.[*]":
                # event_manage_topic_set
                if not (found := re.fullmatch(r"event_properties\.topic_set\.\[(\d+)\]", fqn)):
                    return
                index = int(found[1])
                topic = camera.event_properties.topic_set[index]
                match event.key:
                    case 'space' | 'enter':
                        if node.label.plain.startswith("*"):
                            node.set_label(f"[{index}]: {topic}")
                        else:
                            node.set_label(f"* [{index}]: {topic}")
                        node.parent.set_label(f"topic_set: [{len(camera.event_properties.topic_set)}] (* modified)")

            elif normalized_fqn == "event_properties.topic_set":
                # event_manage_subscription 
                match event.key:
                    case 'u':
                        self.subscription_manager.unsubscribe_events(camera)
                        for i, child in enumerate(node.children):
                            topic = camera.event_properties.topic_set[i]
                            child.set_label(f"[{i}]: {topic}")
                        node.set_label(f"topic_set: [{len(camera.event_properties.topic_set)}]")
                    case 'R':
                        self.subscription_manager.unsubscribe_events(camera)
                        with self.subscription_lock:
                            if not self.event_server:
                                self.event_server = EventServer(self.ip_address, PORT, self.on_camera_events_from_thread)
                                self.event_server.start()
                        self.subscription_manager.subscribe_push_event(camera, self.ip_address, PORT, None)
                        node.set_label(f"topic_set: [{len(camera.event_properties.topic_set)}] (receive ALL)")
                    case 'r':
                        if node.label.plain.endswith("(* modified)"):
                            self.subscription_manager.unsubscribe_events(camera)
                            with self.subscription_lock:
                                if not self.event_server:
                                    self.event_server = EventServer(self.ip_address, PORT, self.on_camera_events_from_thread)
                                    self.event_server.start()
                            for i, child in enumerate(node.children):
                                if child.label.plain.startswith("*"):
                                    topic = camera.event_properties.topic_set[i]
                                    self.subscription_manager.subscribe_push_event(camera, self.ip_address, PORT, topic)
                            status = "" if not len(camera.subscription_references) else " (receive)"
                            node.set_label(f"topic_set: [{len(camera.event_properties.topic_set)}]{status}")
                    # proper ONVIF push implementation, but not implemented by most cameras
                    case 's':
                        if node.label.plain.endswith("(* modified)"):
                            self.subscription_manager.unsubscribe_events(camera)
                            with self.subscription_lock:
                                if not self.event_server:
                                    self.event_server = EventServer(self.ip_address, PORT, self.on_camera_events_from_thread)
                                    self.event_server.start()
                            topics = []
                            for i, child in enumerate(node.children):
                                if child.label.plain.startswith("*"):
                                    topic = camera.event_properties.topic_set[i]
                                    topics.append(topic)
                            self.subscription_manager.subscribe_push_events(camera, self.ip_address, PORT, topics)
                            status = "" if not len(camera.subscription_references) else " (receive)"
                            node.set_label(f"topic_set: [{len(camera.event_properties.topic_set)}]{status}")
                    case 'P':
                        self.subscription_manager.unsubscribe_events(camera)
                        self.subscription_manager.subscribe_pull_event(camera, None)
                        node.set_label(f"topic_set: [{len(camera.event_properties.topic_set)}] (pull ALL)")
                    case 'p':
                        if node.label.plain.endswith("(* modified)"):
                            self.subscription_manager.unsubscribe_events(camera)
                            for i, child in enumerate(node.children):
                                if child.label.plain.startswith("*"):
                                    topic = camera.event_properties.topic_set[i]
                                    self.subscription_manager.subscribe_pull_event(camera, topic)
                            status = "" if not len(camera.subscription_references) else " (pull)"
                            node.set_label(f"topic_set: [{len(camera.event_properties.topic_set)}]{status}")
                    # proper ONVIF pull implementation, but not implemented by most cameras
                    case 'd':
                        if node.label.plain.endswith("(* modified)"):
                            self.subscription_manager.unsubscribe_events(camera)
                            topics = []
                            for i, child in enumerate(node.children):
                                if child.label.plain.startswith("*"):
                                    topic = camera.event_properties.topic_set[i]
                                    topics.append(topic)
                            self.subscription_manager.subscribe_pull_events(camera, topics)
                            status = "" if not len(camera.subscription_references) else " (pull)"
                            node.set_label(f"topic_set: [{len(camera.event_properties.topic_set)}]{status}")

            elif normalized_fqn == "relay_outputs.[*]":
                # relay_output_manage_state
                if not (found := re.fullmatch(r"relay_outputs\.\[(\d+)\]", fqn)):
                    return
                index = int(found[1])
                relay_output = camera.relay_outputs[index]
                match event.key:
                    case 'w':
                        if node.label.plain.endswith("(* modified)"):
                            set_relay_output_settings(camera, relay_output)
                            node.set_label(f"[{index}]")
                    case 'a':
                        self.debug_log.write("RELAY ACTIVATE")
                        set_relay_output_state(camera, relay_output, "active")
                    case 'i':
                        self.debug_log.write("RELAY DEACTIVATE")
                        set_relay_output_state(camera, relay_output, "inactive")

            elif normalized_fqn == "ptz.presets":
                # ptz_add_preset
                match event.key:
                    case 'n':
                        xml = set_preset(camera, profile_token)
                        token = get_xml_value(xml, ".//tptz:SetPresetResponse/tptz:PresetToken")
                        body = f"""<tptz:GetPresets><tptz:ProfileToken>{profile_token}</tptz:ProfileToken></tptz:GetPresets>"""
                        xml = onvif_post(camera.capabilities.ptz.xaddr, body, camera.username, camera.password, camera.time_offset)
                        presets = parse_get_presets_response(xml)
                        for preset in presets:
                            if token == preset.token:
                                camera.ptz.presets.append(preset)
                                length = len(camera.ptz.presets)
                                self.camera_tree._add_value(node, f"[{length-1}]", preset, camera)
                                node.set_label(f"presets: [{length}]")
                                self.camera_tree.refresh()
                                break

            elif normalized_fqn == "ptz.presets.[*]":
                # ptz_manage_preset"
                if not (found := re.fullmatch(r"ptz\.presets\.\[(\d+)\]", fqn)):
                    return
                index = int(found[1])
                preset = camera.ptz.presets[index]
                match event.key:
                    case 's':
                        set_preset(camera, profile_token, preset)
                    case 'd':
                        remove_preset(camera, profile_token, preset)
                        if node := self.camera_tree.cursor_node:
                            parent = node.parent
                            self.camera_tree.move_cursor(parent)
                            print(camera.ptz.presets)
                            node.remove()
                            print(camera.ptz.presets)
                            camera.ptz.presets.pop(index)
                            for idx, preset in enumerate(camera.ptz.presets):
                                preset.token = idx
                            for idx, child in enumerate(parent.children):
                                child.set_label(f"[{idx}]")
                            new_count = len(camera.ptz.presets)
                            parent.set_label(f"presets: [{new_count}]")
                            self.camera_tree.refresh()
                    case 'g':
                        goto_preset(camera, profile_token, preset)
    
            elif normalized_fqn == "ptz.tours":
                # ptz_add_tour
                match event.key:
                    case 'n':
                        xml = create_preset_tour(camera, profile_token)
                        preset_tour_token = get_xml_value(xml, ".//tptz:CreatePresetTourResponse/tptz:PresetTourToken")
                        body = f"""<tptz:GetPresetTours><tptz:ProfileToken>{profile_token}</tptz:ProfileToken></tptz:GetPresetTours>"""
                        xml = onvif_post(camera.capabilities.ptz.xaddr, body, camera.username, camera.password, camera.time_offset)
                        preset_tours = parse_get_preset_tours_response(xml)
                        for preset_tour in preset_tours:
                            if preset_tour_token == preset_tour.token:
                                camera.ptz.tours.append(preset_tour)
                                length = len(camera.ptz.tours)
                                self.camera_tree._add_value(node, f"[{length-1}]", preset_tour, camera)
                                node.set_label(f"tours [{length}]")
                                self.camera_tree.refresh()
                                break

            elif normalized_fqn == "ptz.tours.[*].spots.[*]":
                # ptz_delete_tour_spot
                if not (found := re.fullmatch(r"ptz\.tours\.\[(\d+)\]\.spots\.\[(\d+)\]", fqn)):
                    return
                tour_index = int(found[1])
                spot_index = int(found[2])
                match event.key:
                    case 'd':
                        parent = node.parent
                        self.camera_tree.move_cursor(parent)
                        node.remove()
                        del camera.ptz.tours[tour_index].spots[spot_index]
                        length = len(camera.ptz.tours[tour_index].spots)
                        if length == 1:
                            parent.allow_expand = False
                        parent.set_label(f"spots: [{length}]")
                        grand_parent = parent.parent
                        grand_parent.set_label(f"[{tour_index}] (* modified)")
                        for i, child in enumerate(parent.children):
                            child.set_label(f"[{i}]")
                            child.data["fqn"] = f"ptz.tours.[{tour_index}].spots.[{i}]"

            elif normalized_fqn == "ptz.tours.[*].spots":
                # ptz_add_tour_spot
                if not (found := re.fullmatch(r"ptz\.tours\.\[(\d+)\]\.spots", fqn)):
                    return
                tour_index = int(found[1])
                preset_tour_token = camera.ptz.tours[tour_index].token
                match event.key:
                    case 'n':
                        tour_spot = TourSpot("1", "PT25S")
                        camera.ptz.tours[tour_index].spots.append(tour_spot)
                        length = len(camera.ptz.tours[tour_index].spots)
                        node.allow_expand = True
                        self.camera_tree._add_value(node, f"[{length-1}]", tour_spot, camera)
                        node.set_label(f"spots: [{length}]")
                        node.parent.set_label(f"[{tour_index}] (* modified)")
                        self.camera_tree.refresh()

            elif normalized_fqn == "ptz.tours.[*]":
                # ptz_manage_tour
                if not (found := re.fullmatch(r"ptz\.tours\.\[(\d+)\]", fqn)):
                    return
                tour_index = int(found[1])
                preset_tour = camera.ptz.tours[tour_index]
                match event.key:
                    case 's':
                        operate_preset_tour(camera, profile_token, preset_tour, 'Start')
                    case 't':
                        operate_preset_tour(camera, profile_token, preset_tour, 'Stop')
                    case 'd':
                        remove_preset_tour(camera, profile_token, preset_tour)
                        parent = node.parent
                        self.camera_tree.move_cursor(parent)
                        node.remove()
                        del camera.ptz.tours[tour_index]
                        new_count = len(camera.ptz.tours)
                        parent.set_label(f"tours: [{new_count}]")
                        self.camera_tree.refresh()
                    case 'w':
                        if node.label.plain.endswith("(* modified)"):
                            modify_preset_tour(camera, profile_token, preset_tour)
                            node.set_label(f"[{tour_index}]")

            elif normalized_fqn == "ptz":
                # ptz_move_camera
                self.is_zoom_move = False
                match event.key:
                    case 'w':
                        self.debug_log.write(f"\nmoving up...")
                        continuous_move(camera, profile_token, 0, 0.5, 0)
                    case 's':
                        self.debug_log.write(f"\nmoving down...")
                        continuous_move(camera, profile_token, 0, -0.5, 0)
                    case 'a':
                        self.debug_log.write(f"\npanning right...")
                        continuous_move(camera, profile_token, 0.5, 0, 0)
                    case 'd':
                        self.debug_log.write(f"\npanning left...")
                        continuous_move(camera, profile_token, -0.5, 0, 0)
                    case 'z':
                        self.debug_log.write(f"\nzooming in...")
                        continuous_move(camera, profile_token, 0, 0, 0.5)
                        self.is_zoom_move = True 
                    case 'x':
                        self.debug_log.write(f"\nzooming out...")
                        continuous_move(camera, profile_token, 0, 0, -0.5)
                        self.is_zoom_move = True
                    case 'c':
                        self.debug_log.write(f"\nstop move")
                        move_stop(camera, profile_token, self.is_zoom_move)
                    case 'i':
                        self.debug_log.write(f"\ninformation\n")
                        xml = get_status(camera, profile_token)
                        pan_x = get_xml_value(xml, ".//tptz:GetStatusResponse/tptz:PTZStatus/tt:Position/tt:PanTilt/@x")
                        pan_y = get_xml_value(xml, ".//tptz:GetStatusResponse/tptz:PTZStatus/tt:Position/tt:PanTilt/@y")
                        zoom_x = get_xml_value(xml, ".//tptz:GetStatusResponse/tptz:PTZStatus/tt:Position/tt:Zoom/@x")
                        pan_tilt_status = get_xml_value(xml, ".//tptz:GetStatusResponse/tptz:PTZStatus/tt:MoveStatus/tt:PanTilt")
                        zoom_status = get_xml_value(xml, ".//tptz:GetStatusResponse/tptz:PTZStatus/tt:MoveStatus/tt:Zoom")
                        self.debug_log.write(f"X:    {pan_x}\nY:    {pan_y}\nZOOM: {zoom_x}\nPAN TILT STATUS: {pan_tilt_status}\nZOOM STATUS: {zoom_status}")

        except Exception as ex:
            self.debug_log.write(f"exception editing field: {fqn} - {ex}")
            print(traceback.format_exc())


    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input is not self.edit_input:
            return

        try:
            old_value = getattr(self.editing_owner, self.editing_field)
            setattr(self.editing_owner, self.editing_field, convert_string_value(event.value.strip(), self.editing_field_type))

            msg = "\n"
            fqn = self.editing_node.data["fqn"]
            normalized_fqn = normalize_fqn(fqn)

            if normalized_fqn == "network_gateway":
                if "RebootNeeded" in set_network_default_gateway(self.editing_camera):
                    msg += "Updated successfully, please reboot the camera to enact the update\n"

            elif normalized_fqn == "hostname.from_dhcp":
                if "RebootNeeded" in set_hostname_from_dhcp(self.editing_camera):
                    msg += "Updated successfully, please reboot the camera to enact the update\n"

            elif normalized_fqn == "hostname.name":
                xml = set_hostname(self.editing_camera)
                msg = "set hostname error" if xml is None else "Updated successfully.\n"

            elif normalized_fqn.startswith("dns."):
                xml = set_dns(self.editing_camera)
                msg = "set dns error" if xml is None else "Updated successfully.\n"

            elif normalized_fqn.startswith("ntp."):
                xml = set_ntp(self.editing_camera)
                msg = "set ntp error" if xml is None else "Updated successfully.\n"

            elif normalized_fqn.startswith("network_interfaces.[*].ipv4"):
                index = self.editing_indicies[-1]
                interface = self.editing_camera.network_interfaces[index]
                manual = interface.ipv4.manual
                if "RebootNeeded" in set_network_interfaces(self.editing_camera, interface, manual):
                    msg += "Updated successfully, please reboot the camera to enact the update\n"

            elif normalized_fqn.startswith("profiles.[*].imaging_settings"):
                index = self.editing_indicies[0]
                profile = self.editing_camera.profiles[index]
                xml = set_imaging_settings(self.editing_camera, profile.video_source.source_token, profile.imaging_settings)
                msg = "set imaging settings error" if xml is None else "Updated successfully.\n"

            elif normalized_fqn.startswith("profiles.[*].audio_encoder"):
                index = self.editing_indicies[0]
                profile = self.editing_camera.profiles[index]
                xml = set_audio_encoder_configuration(self.editing_camera, profile.audio_encoder)
                msg = "set audio encoder error" if xml is None else "Updated successfully.\n"

            elif normalized_fqn.startswith("profiles.[*].video_encoder"):
                index = self.editing_indicies[0]
                profile = self.editing_camera.profiles[index]
                xml = set_video_encoder_configuration(self.editing_camera, profile.video_encoder)
                msg = "set video encoder error" if xml is None else "Updated successfully.\n"

            elif normalized_fqn.startswith("ptz.tours.[*].spots.[*]"):
                parent = self.editing_node.parent.parent.parent
                index = self.editing_indicies[0]
                parent.set_label(f"[{index}] (* modified)")
                msg = f"{fqn}\nhas been modified, navigate to\n{parent.label.plain}\nand use the 'w' key to commit the change"

            elif normalized_fqn.startswith("ptz.tours.[*]"):
                parent = self.editing_node.parent
                index = self.editing_indicies[0]
                parent.set_label(f"[{index}] (* modified)")
                msg = f"{fqn}\nhas been modified, navigate to\n{parent.label.plain}\nand use the 'w' key to commit the change"

            elif normalized_fqn.startswith("relay_outputs.[*].properties"):
                parent = self.editing_node.parent.parent
                index = self.editing_indicies[0]
                parent.set_label(f"[{index}] (* modified)")
                msg = f"{fqn}\nhas been modified, navigate to\n{parent.label.plain}\nand use the 'w' key to commit the change"

            elif normalized_fqn.startswith("system_date_and_time"):
                search_node = self.editing_node
                while search_node.parent.label.plain != "system_date_and_time" and search_node.parent.label.plain != "system_date_and_time (* modified)":
                    search_node = search_node.parent
                search_node.parent.set_label("system_date_and_time (* modified)")
                msg = f"{fqn}\nhas been modified, navigate to\nsystem_date_and_time (* modified)\nand use the 'w' key to commit the change"

            self.debug_log.write(f"\n{msg}")
        except Exception as ex:
            setattr(self.editing_owner, self.editing_field, old_value)
            self.debug_log.write(f"\nUpdate Failure:\n\n{ex}")

        self.editing_node.set_label(self.camera_tree._make_editable_label(self.editing_field, str(getattr(self.editing_owner, self.editing_field))))
        self.edit_input.add_class("hidden")
        self.set_focus(self.camera_tree)

    def action_cancel_edit(self) -> None:
        if self.edit_input.has_class("hidden"):
            return

        self.edit_input.add_class("hidden")
        self.set_focus(self.camera_tree)

    def action_edit_selected(self) -> None:
        node = self.camera_tree.cursor_node
        if node is None or not node.data:
            return

        fqn = node.data["fqn"]

        if not is_editable_field(fqn): 
            return

        camera = node.data["camera"]
        owner, field_name, field_type, indices = resolve_fqn_owner(camera, fqn)
        base_type, is_optional, is_list = analyze_field_type(field_type)
        default_value = "False" if base_type is bool else ""

        self.editing_node = node
        self.editing_camera = camera
        self.editing_owner = owner
        self.editing_field = field_name
        self.editing_field_type = field_type
        self.editing_indicies = indices

        self.edit_input.value = str(getattr(owner, field_name) or default_value)
        self.edit_input.remove_class("hidden")
        self.set_focus(self.edit_input)

    def compose(self) -> ComposeResult:
        self.camera_tree = CameraTree()
        self.edit_input = Input(id="edit_box", placeholder="New value")
        self.edit_input.add_class("hidden")
        self.debug_log = RichLog(id="debug_log", highlight=True, wrap=True)

        yield Header()
        with Horizontal(id="main"):
            yield self.camera_tree
            yield self.debug_log
        yield self.edit_input
        yield Footer()

    def handle_camera_events(self, alarms: list[dict[str, str]]) -> None:
        for alarm in alarms:
            for key, value in alarm.items():
                self.debug_log.write(f"{key}: {value}")
            self.debug_log.write("\n")

    def on_camera_events_from_thread(self, alarms: list[dict[str, str]]) -> None:
        self.call_from_thread(self.handle_camera_events, alarms)

    def on_mount(self) -> None:
        self.run_worker(self.discover_worker, thread=True)
        self.debug_log.write(f"Available network interfaces: {self.ips}")
        self.debug_log.write(f"Discovering cameras on {self.ip_address} ...")
        self.loop_callback = self.set_interval(5, self.main_loop)

    def on_unmount(self) -> None:
        if self.event_server is not None:
            self.event_server.stop()
        for child in self.camera_tree.root.children:
            if not child.data:
                continue
            if camera := child.data.get("camera"):
                for reference in camera.subscription_references:
                    unsubscribe(camera, reference.xaddr)

    def on_error(self, xaddr: str, ex: Exception) -> None:
        for child in self.camera_tree.root.children:
            if not child.data:
                continue
            if camera := child.data.get("camera"):
                if camera.xaddr == xaddr:
                    for grand_child in child.children:
                        if grand_child.label.plain.startswith("errors"):
                            self.debug_log.write(f"Error with camera at {camera.name}: {ex}")
                            grand_child.set_label("errors: ** Error")
                            self.camera_tree.refresh()
                            break

    def discover_worker(self) -> None:
        def camera_filled(camera: Camera) -> None:
            self.call_from_thread(self.camera_tree.add_camera, camera)

        def get_camera_credentials(camera: Camera) -> None:
            camera.username = self.username
            camera.password = self.password

        try:
            if self.manual:
                cameras = find_camera_manually(self.manual, get_camera_credentials, on_error=self.on_error, camera_filled=camera_filled)
            else:
                cameras = discover(self.ip_address, get_camera_credentials, on_error=self.on_error, camera_filled=camera_filled)

            self.debug_log.write(f"Found {len(cameras)} {"camera" if len(cameras) == 1 else "cameras"}")
        except Exception as ex:
            self.debug_log.write(f"Discovery error: {ex}")
            print(traceback.format_exc(), flush=True)

    def pull_worker(self, camera: Camera, reference: SubscriptionReference) -> None:
        try:
            xml = pull_messages(camera, reference.xaddr)
            alarms = parse_notify(camera.xaddr, xml)
            for alarm in alarms:
                for key, value in alarm.items():
                    self.call_from_thread(self.debug_log.write, f"{key}: {value}")
        except Exception as ex:
            self.call_from_thread(self.debug_log.write, f"Error pulling messages from {camera.name}\n{ex}\nUTC: {datetime.now(timezone.utc)}")
            print(traceback.format_exc(), flush=True)

    def main_loop(self) -> None:
        for child in self.camera_tree.root.children:
            if not child.data: return
            if not (camera := child.data.get("camera")): return
            for reference in camera.subscription_references:
                if reference.subscription_type == SubscriptionType.PULL:
                    self.run_worker(lambda cam=camera, ref=reference: self.pull_worker(cam, ref), thread=True)

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--ip_address", default="0.0.0.0", help="Local IP address binding for ONVIF discover/event callback")
    parser.add_argument("-m", "--manual", default=None, help="Camera IP address for manual camera discovery")
    parser.add_argument("-u", "--username", default="", help="username for camera authentication")
    parser.add_argument("-p", "--password", default="", help="password for camera authentication")
    args = parser.parse_args()
    app = ObjectBrowser(args)
    app.run() 

if __name__ == "__main__":
    main()