from textual.widgets import Tree
from textual.widgets.tree import TreeNode
from dataclasses import is_dataclass, fields
from rich.text import Text
from .fields import UNUSED_FIELDS, HIDDEN_FIELDS, field_descriptions, join_fqn, is_editable_field, main_screen_text
from libonvif.devices.camera import Camera
import re

class CameraTree(Tree):
    def __init__(self) -> None:
        super().__init__("Cameras")
        self.show_root = True

    def get_fqn(self, node: TreeNode) -> str:
        parts = []
        current = node
        while current is not None:
            parent = current.parent
            if parent is None:
                break
            data = getattr(current, "data", None)
            if data and "field" in data:
                parts.append(data["field"])
            current = parent
        return ".".join(reversed(parts))

    def capture_expanded_nodes(self, node) -> set[str]:
        expanded: set[str] = set()

        def walk(n):
            data = n.data or {}
            fqn = data.get("fqn")

            if fqn and n.is_expanded:
                expanded.add(fqn)

            for child in n.children:
                walk(child)

        walk(node)
        return expanded

    def restore_expanded_nodes(self, node, expanded: set[str]) -> None:
        def walk(n):
            data = n.data or {}
            fqn = data.get("fqn")

            if fqn in expanded:
                n.allow_expand = True
                n.expand()

            for child in n.children:
                walk(child)

        walk(node)

    def on_tree_node_highlighted(self, event: Tree.NodeHighlighted) -> None:

        if event.node.parent is None:
            self.app.debug_log.clear()
            self.app.debug_log.write(main_screen_text)

        if not event.node.data: return

        if camera := event.node.data.get("camera"):
            if event.node.label.plain == camera.name:
                self.app.debug_log.clear()
                self.app.debug_log.write("The camera can be rebooted using the 'r' key")

        if not (fqn := event.node.data.get("fqn")): return

        sfqn = re.sub(r"\[\d+\]", "[*]", fqn)
        self.app.debug_log.clear()
        self.app.debug_log.write(fqn)
        if desc := field_descriptions.get(sfqn):
            self.app.debug_log.write(desc)
        if fqn == "errors" and camera.errors:
            self.app.debug_log.clear()
            self.app.debug_log.write(camera.errors)

    def _make_editable_label(self, field: str, value: str) -> Text:
        label = Text()
        label.append("✎  ", style="#66cc66")
        label.append(f"{field}: ")
        label.append(str(value))
        return label

    def add_camera(self, camera: Camera) -> None:
        label = camera.name
        camera_node = self.root.add(label, expand=False)
        camera_node.data = { "camera": camera }
        for field in fields(camera):
            value = getattr(camera, field.name)
            self._add_value(camera_node, field.name, value, camera)
        if len(self.root.children) == 1:
            self.root.expand()

    def _add_value(self, parent: TreeNode, name: str, value: object, camera: Camera) -> TreeNode | None:

        fqn = join_fqn(self.get_fqn(parent), name)

        if fqn in HIDDEN_FIELDS:
            return

        if is_editable_field(fqn) and value is not None:
            node = parent.add_leaf(self._make_editable_label(name, str(value)))
            node.data = {"camera": camera, "field": name, "fqn": fqn}
            return

        if value is None:
            if fqn in UNUSED_FIELDS:
                return
            node = parent.add_leaf(Text(f"{name}: None", style="dim"))
            node.data = {"camera": camera, "field": name, "fqn": fqn}
            return

        if is_dataclass(value):
            node = parent.add(name, expand=False)
            node.data = {"camera": camera, "field": name, "fqn": fqn}
            for field in fields(value):
                child_value = getattr(value, field.name)
                self._add_value(node, field.name, child_value, camera)

        elif isinstance(value, list):
            if not value:
                if fqn in UNUSED_FIELDS:
                    return
                label = Text(f"{name}: list[0]", style="dim")
                node = parent.add_leaf(label)
                node.data = {"camera": camera, "field": name, "fqn": fqn}
                return
            node = parent.add(f"{name}: [{len(value)}]", expand=False)
            node.data = {"camera": camera, "field": name, "fqn": fqn}
            for index, item in enumerate(value):
                self._add_value(node, f"[{index}]", item, camera)

        elif isinstance(value, dict):
            node = parent.add(f"{name}: dict[{len(value)}]", expand=False)
            node.data = {"camera": camera, "field": name, "fqn": fqn}
            for key, item in value.items():
                self._add_value(node, str(key), item, camera)

        else:
            node = parent.add_leaf(f"{name}: {value}")
            node.data = {"camera": camera, "field": name, "fqn": fqn}

        return node
