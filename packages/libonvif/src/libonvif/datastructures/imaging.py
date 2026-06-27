from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
from lxml import etree
from libonvif.utils.xml import text, attr, float_text, text_list, NS

@dataclass
class Bounds:
    x: Optional[int] = None
    y: Optional[int] = None
    width: Optional[int] = None
    height: Optional[int] = None

@dataclass
class FloatRange:
    min: Optional[float] = None
    max: Optional[float] = None

@dataclass
class BacklightCompensation:
    mode: Optional[str] = None
    level: Optional[float] = None

@dataclass
class Exposure:
    mode: Optional[str] = None
    priority: Optional[str] = None
    window: Optional[Bounds] = None
    min_exposure_time: Optional[float] = None
    max_exposure_time: Optional[float] = None
    min_gain: Optional[float] = None
    max_gain: Optional[float] = None
    min_iris: Optional[float] = None
    max_iris: Optional[float] = None
    exposure_time: Optional[float] = None
    gain: Optional[float] = None
    iris: Optional[float] = None

@dataclass
class Focus:
    auto_focus_mode: Optional[str] = None
    default_speed: Optional[float] = None
    near_limit: Optional[float] = None
    far_limit: Optional[float] = None

@dataclass
class WideDynamicRange:
    mode: Optional[str] = None
    level: Optional[float] = None

@dataclass
class WhiteBalance:
    mode: Optional[str] = None
    cr_gain: Optional[float] = None
    cb_gain: Optional[float] = None

@dataclass
class ImagingSettings:
    backlight_compensation: Optional[BacklightCompensation] = None
    brightness: Optional[float] = None
    color_saturation: Optional[float] = None
    contrast: Optional[float] = None
    exposure: Optional[Exposure] = None
    focus: Optional[Focus] = None
    ir_cut_filter: Optional[str] = None
    sharpness: Optional[float] = None
    wide_dynamic_range: Optional[WideDynamicRange] = None
    white_balance: Optional[WhiteBalance] = None

@dataclass
class ImagingOptions:
    backlight_compensation: Optional[BacklightCompensationOptions] = None
    brightness: Optional[FloatRange] = None
    color_saturation: Optional[FloatRange] = None
    contrast: Optional[FloatRange] = None
    exposure: Optional[ExposureOptions] = None
    focus: Optional[FocusOptions] = None
    ir_cut_filter_modes: list[str] = field(default_factory=list)
    sharpness: Optional[FloatRange] = None
    wide_dynamic_range: Optional[WideDynamicRangeOptions] = None
    white_balance: Optional[WhiteBalanceOptions] = None

@dataclass
class BacklightCompensationOptions:
    modes: list[str] = field(default_factory=list)
    level: Optional[FloatRange] = None

@dataclass
class ExposureOptions:
    modes: list[str] = field(default_factory=list)
    priority: list[str] = field(default_factory=list)
    min_exposure_time: Optional[FloatRange] = None
    max_exposure_time: Optional[FloatRange] = None
    min_gain: Optional[FloatRange] = None
    max_gain: Optional[FloatRange] = None
    min_iris: Optional[FloatRange] = None
    max_iris: Optional[FloatRange] = None
    exposure_time: Optional[FloatRange] = None
    gain: Optional[FloatRange] = None
    iris: Optional[FloatRange] = None

@dataclass
class FocusOptions:
    auto_focus_modes: list[str] = field(default_factory=list)
    default_speed: Optional[FloatRange] = None
    near_limit: Optional[FloatRange] = None
    far_limit: Optional[FloatRange] = None

@dataclass
class WideDynamicRangeOptions:
    modes: list[str] = field(default_factory=list)
    level: Optional[FloatRange] = None

@dataclass
class WhiteBalanceOptions:
    modes: list[str] = field(default_factory=list)
    yr_gain: Optional[FloatRange] = None
    yb_gain: Optional[FloatRange] = None

def parse_bounds(elem: Optional[etree._Element]) -> Optional[Bounds]:
    if elem is None:
        return None

    return Bounds(
        x=int(attr(elem, "x")) if attr(elem, "x") else None,
        y=int(attr(elem, "y")) if attr(elem, "y") else None,
        width=int(attr(elem, "width")) if attr(elem, "width") else None,
        height=int(attr(elem, "height")) if attr(elem, "height") else None,
    )

def parse_backlight_compensation(
    elem: Optional[etree._Element],
) -> Optional[BacklightCompensation]:
    if elem is None:
        return None

    return BacklightCompensation(
        mode=text(elem, "tt:Mode"),
        level=float_text(elem, "tt:Level"),
    )

def parse_exposure(elem: Optional[etree._Element]) -> Optional[Exposure]:
    if elem is None:
        return None

    return Exposure(
        mode=text(elem, "tt:Mode"),
        priority=text(elem, "tt:Priority"),
        window=parse_bounds(elem.find("tt:Window", NS)),
        min_exposure_time=float_text(elem, "tt:MinExposureTime"),
        max_exposure_time=float_text(elem, "tt:MaxExposureTime"),
        min_gain=float_text(elem, "tt:MinGain"),
        max_gain=float_text(elem, "tt:MaxGain"),
        min_iris=float_text(elem, "tt:MinIris"),
        max_iris=float_text(elem, "tt:MaxIris"),
        exposure_time=float_text(elem, "tt:ExposureTime"),
        gain=float_text(elem, "tt:Gain"),
        iris=float_text(elem, "tt:Iris"),
    )

def parse_focus(elem: Optional[etree._Element]) -> Optional[Focus]:
    if elem is None:
        return None

    return Focus(
        auto_focus_mode=text(elem, "tt:AutoFocusMode"),
        default_speed=float_text(elem, "tt:DefaultSpeed"),
        near_limit=float_text(elem, "tt:NearLimit"),
        far_limit=float_text(elem, "tt:FarLimit"),
    )

def parse_wide_dynamic_range(
    elem: Optional[etree._Element],
) -> Optional[WideDynamicRange]:
    if elem is None:
        return None

    return WideDynamicRange(
        mode=text(elem, "tt:Mode"),
        level=float_text(elem, "tt:Level"),
    )

def parse_white_balance(elem: Optional[etree._Element]) -> Optional[WhiteBalance]:
    if elem is None:
        return None

    return WhiteBalance(
        mode=text(elem, "tt:Mode"),
        cr_gain=float_text(elem, "tt:CrGain"),
        cb_gain=float_text(elem, "tt:CbGain"),
    )

def parse_bounds(elem: Optional[etree._Element]) -> Optional[Bounds]:
    if elem is None:
        return None

    return Bounds(
        x=int(attr(elem, "x")) if attr(elem, "x") else None,
        y=int(attr(elem, "y")) if attr(elem, "y") else None,
        width=int(attr(elem, "width")) if attr(elem, "width") else None,
        height=int(attr(elem, "height")) if attr(elem, "height") else None,
    )

def parse_backlight_compensation(
    elem: Optional[etree._Element],
) -> Optional[BacklightCompensation]:
    if elem is None:
        return None

    return BacklightCompensation(
        mode=text(elem, "tt:Mode"),
        level=float_text(elem, "tt:Level"),
    )

def parse_exposure(elem: Optional[etree._Element]) -> Optional[Exposure]:
    if elem is None:
        return None

    return Exposure(
        mode=text(elem, "tt:Mode"),
        priority=text(elem, "tt:Priority"),
        window=parse_bounds(elem.find("tt:Window", NS)),
        min_exposure_time=float_text(elem, "tt:MinExposureTime"),
        max_exposure_time=float_text(elem, "tt:MaxExposureTime"),
        min_gain=float_text(elem, "tt:MinGain"),
        max_gain=float_text(elem, "tt:MaxGain"),
        min_iris=float_text(elem, "tt:MinIris"),
        max_iris=float_text(elem, "tt:MaxIris"),
        exposure_time=float_text(elem, "tt:ExposureTime"),
        gain=float_text(elem, "tt:Gain"),
        iris=float_text(elem, "tt:Iris"),
    )

def parse_focus(elem: Optional[etree._Element]) -> Optional[Focus]:
    if elem is None:
        return None

    return Focus(
        auto_focus_mode=text(elem, "tt:AutoFocusMode"),
        default_speed=float_text(elem, "tt:DefaultSpeed"),
        near_limit=float_text(elem, "tt:NearLimit"),
        far_limit=float_text(elem, "tt:FarLimit"),
    )

def parse_wide_dynamic_range(
    elem: Optional[etree._Element],
) -> Optional[WideDynamicRange]:
    if elem is None:
        return None

    return WideDynamicRange(
        mode=text(elem, "tt:Mode"),
        level=float_text(elem, "tt:Level"),
    )

def parse_white_balance(elem: Optional[etree._Element]) -> Optional[WhiteBalance]:
    if elem is None:
        return None

    return WhiteBalance(
        mode=text(elem, "tt:Mode"),
        cr_gain=float_text(elem, "tt:CrGain"),
        cb_gain=float_text(elem, "tt:CbGain"),
    )

def parse_imaging_settings_response(xml: str) -> ImagingSettings:
    if not xml: return
    root = etree.fromstring(xml.encode('utf-8'))

    settings = root.find(
        ".//timg:GetImagingSettingsResponse/timg:ImagingSettings",
        NS,
    )
    if settings is None:
        raise ValueError(
            "Could not find timg:GetImagingSettingsResponse/timg:ImagingSettings"
        )

    return ImagingSettings(
        backlight_compensation=parse_backlight_compensation(
            settings.find("tt:BacklightCompensation", NS)
        ),
        brightness=float_text(settings, "tt:Brightness"),
        color_saturation=float_text(settings, "tt:ColorSaturation"),
        contrast=float_text(settings, "tt:Contrast"),
        exposure=parse_exposure(settings.find("tt:Exposure", NS)),
        focus=parse_focus(settings.find("tt:Focus", NS)),
        ir_cut_filter=text(settings, "tt:IrCutFilter"),
        sharpness=float_text(settings, "tt:Sharpness"),
        wide_dynamic_range=parse_wide_dynamic_range(
            settings.find("tt:WideDynamicRange", NS)
        ),
        white_balance=parse_white_balance(settings.find("tt:WhiteBalance", NS)),
    )

def parse_float_range(elem: Optional[etree._Element]) -> Optional[FloatRange]:
    if elem is None:
        return None

    return FloatRange(
        min=float_text(elem, "tt:Min"),
        max=float_text(elem, "tt:Max"),
    )

def parse_backlight_compensation_options(
    elem: Optional[etree._Element],
) -> Optional[BacklightCompensationOptions]:
    if elem is None:
        return None

    return BacklightCompensationOptions(
        modes=text_list(elem, "tt:Mode"),
        level=parse_float_range(elem.find("tt:Level", NS)),
    )

def parse_exposure_options(elem: Optional[etree._Element]) -> Optional[ExposureOptions]:
    if elem is None:
        return None

    return ExposureOptions(
        modes=text_list(elem, "tt:Mode"),
        priority=text_list(elem, "tt:Priority"),
        min_exposure_time=parse_float_range(elem.find("tt:MinExposureTime", NS)),
        max_exposure_time=parse_float_range(elem.find("tt:MaxExposureTime", NS)),
        min_gain=parse_float_range(elem.find("tt:MinGain", NS)),
        max_gain=parse_float_range(elem.find("tt:MaxGain", NS)),
        min_iris=parse_float_range(elem.find("tt:MinIris", NS)),
        max_iris=parse_float_range(elem.find("tt:MaxIris", NS)),
        exposure_time=parse_float_range(elem.find("tt:ExposureTime", NS)),
        gain=parse_float_range(elem.find("tt:Gain", NS)),
        iris=parse_float_range(elem.find("tt:Iris", NS)),
    )

def parse_focus_options(elem: Optional[etree._Element]) -> Optional[FocusOptions]:
    if elem is None:
        return None

    return FocusOptions(
        auto_focus_modes=text_list(elem, "tt:AutoFocusModes"),
        default_speed=parse_float_range(elem.find("tt:DefaultSpeed", NS)),
        near_limit=parse_float_range(elem.find("tt:NearLimit", NS)),
        far_limit=parse_float_range(elem.find("tt:FarLimit", NS)),
    )

def parse_wide_dynamic_range_options(
    elem: Optional[etree._Element],
) -> Optional[WideDynamicRangeOptions]:
    if elem is None:
        return None

    return WideDynamicRangeOptions(
        modes=text_list(elem, "tt:Mode"),
        level=parse_float_range(elem.find("tt:Level", NS)),
    )

def parse_white_balance_options(
    elem: Optional[etree._Element],
) -> Optional[WhiteBalanceOptions]:
    if elem is None:
        return None

    return WhiteBalanceOptions(
        modes=text_list(elem, "tt:Mode"),
        yr_gain=parse_float_range(elem.find("tt:YrGain", NS)),
        yb_gain=parse_float_range(elem.find("tt:YbGain", NS)),
    )
    
def parse_imaging_options_response(xml: str) -> ImagingOptions:
    if not xml: return
    root = etree.fromstring(xml.encode('utf-8'))

    options = root.find(
        ".//timg:GetOptionsResponse/timg:ImagingOptions",
        NS,
    )
    if options is None:
        raise ValueError("Could not find timg:GetOptionsResponse/timg:ImagingOptions")

    return ImagingOptions(
        backlight_compensation=parse_backlight_compensation_options(
            options.find("tt:BacklightCompensation", NS)
        ),
        brightness=parse_float_range(options.find("tt:Brightness", NS)),
        color_saturation=parse_float_range(options.find("tt:ColorSaturation", NS)),
        contrast=parse_float_range(options.find("tt:Contrast", NS)),
        exposure=parse_exposure_options(options.find("tt:Exposure", NS)),
        focus=parse_focus_options(options.find("tt:Focus", NS)),
        ir_cut_filter_modes=text_list(options, "tt:IrCutFilterModes"),
        sharpness=parse_float_range(options.find("tt:Sharpness", NS)),
        wide_dynamic_range=parse_wide_dynamic_range_options(
            options.find("tt:WideDynamicRange", NS)
        ),
        white_balance=parse_white_balance_options(
            options.find("tt:WhiteBalance", NS)
        ),
    )
