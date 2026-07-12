"""
Generic recursive dataclass <-> plain-dict codec.

This module provides to_dict/from_dict helpers that walk an arbitrary
graph of @dataclass instances (including nested dataclasses, lists of
dataclasses, and Enum members) and convert it to/from plain,
JSON-serializable Python structures (dict/list/str/int/float/bool/None).

Design notes
------------
Some dataclass fields hold live, non-serializable runtime objects that
have no meaningful JSON representation - e.g. Camera.on_error (a callback
function) or SubscriptionReference.resubscribe_timer (a running
threading.Timer). These fields are tagged in their dataclass definition
with:

    field(default=None, metadata={"transient": True})

to_dict() always omits transient fields. from_dict() always skips them,
leaving them at their declared default. This keeps the "what should
never leave this process" decision declared once, on the data model
itself, rather than scattered as defensive .pop() calls at every
serialization call site.
"""

from __future__ import annotations

from dataclasses import fields, is_dataclass
from enum import Enum
from typing import Any, Union, get_args, get_origin, get_type_hints


def _is_transient(f) -> bool:
    return bool(f.metadata.get("transient"))


def to_dict(obj: Any) -> Any:
    """Recursively convert a dataclass instance into a plain,
    JSON-serializable structure. Transient fields are always omitted.
    """
    if is_dataclass(obj) and not isinstance(obj, type):
        result = {}
        for f in fields(obj):
            if _is_transient(f):
                continue
            result[f.name] = to_dict(getattr(obj, f.name))
        return result

    if isinstance(obj, Enum):
        return obj.value

    if isinstance(obj, list):
        return [to_dict(item) for item in obj]

    if isinstance(obj, dict):
        return {key: to_dict(value) for key, value in obj.items()}

    # str, int, float, bool, None all pass through unchanged
    return obj


def _unwrap_optional(tp: Any) -> Any:
    """Given Optional[X] (i.e. Union[X, None]), return X. Any other type
    is returned unchanged.
    """
    if get_origin(tp) is Union:
        args = [a for a in get_args(tp) if a is not type(None)]
        if len(args) == 1:
            return args[0]
    return tp


def _from_value(field_type: Any, value: Any) -> Any:
    if value is None:
        return None

    field_type = _unwrap_optional(field_type)
    origin = get_origin(field_type)

    if origin is list:
        (item_type,) = get_args(field_type) or (Any,)
        return [_from_value(item_type, item) for item in value]

    if is_dataclass(field_type):
        return from_dict(field_type, value)

    if isinstance(field_type, type) and issubclass(field_type, Enum):
        return field_type(value)

    # str, int, float, bool, and anything else pass through unchanged
    return value


def from_dict(cls: type, data: Any) -> Any:
    """Recursively reconstruct a dataclass instance of type `cls` from a
    plain dict, as produced by to_dict() (typically after a json.loads()
    round trip). Transient fields are always skipped and left at their
    declared default - a live callback or timer object cannot be
    meaningfully reconstructed from JSON.
    """
    if data is None:
        return None

    if not is_dataclass(cls):
        return data

    hints = get_type_hints(cls)
    kwargs = {}
    for f in fields(cls):
        if _is_transient(f):
            continue
        if f.name not in data:
            continue
        field_type = hints.get(f.name, f.type)
        kwargs[f.name] = _from_value(field_type, data[f.name])

    return cls(**kwargs)
