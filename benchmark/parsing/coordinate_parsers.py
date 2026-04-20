"""
Coordinate parsers for different model output formats.

Each parser takes raw model output (string) and returns a normalized (x, y)
point in [0, 1], or None if parsing failed.

When a model returns a bounding box instead of a point, we take the box center.
"""

import re
from typing import Optional, Tuple

Point = Optional[Tuple[float, float]]


def parse_moondream_loc_tokens(text: str) -> Point:
    """Moondream: '<loc_231><loc_025>' -> (0.231, 0.025)."""
    matches = re.findall(r'<loc_(\d+)>', text)
    if len(matches) >= 2:
        x = int(matches[0]) / 999
        y = int(matches[1]) / 999
        return (x, y)
    return None


def parse_florence_loc_tokens(text: str, take: str = "first") -> Point:
    """Florence-2: '<loc_X1><loc_Y1><loc_X2><loc_Y2>' (bbox). Returns center.

    take: "first" for first two (point), "center" for bbox center.
    """
    matches = re.findall(r'<loc_(\d+)>', text)
    if take == "first" and len(matches) >= 2:
        return (int(matches[0]) / 999, int(matches[1]) / 999)
    if take == "center" and len(matches) >= 4:
        x1, y1, x2, y2 = (int(m) / 999 for m in matches[:4])
        return ((x1 + x2) / 2, (y1 + y2) / 2)
    return None


def parse_json_point(text: str, image_w: int = None, image_h: int = None) -> Point:
    """Parse '{"x": 0.5, "y": 0.3}' or '{"x": 960, "y": 540}' (pixels).

    If x or y > 1.0 and image dims given, normalize by dividing by image dims.
    """
    import json
    match = re.search(r'\{[^}]*"x"[^}]*"y"[^}]*\}', text)
    if not match:
        match = re.search(r'\{[^}]+\}', text)
    if not match:
        return None
    try:
        data = json.loads(match.group())
        x = float(data.get("x", -1))
        y = float(data.get("y", -1))
        if x < 0 or y < 0:
            return None
        if x > 1.0 and image_w:
            x = x / image_w
        if y > 1.0 and image_h:
            y = y / image_h
        return (max(0.0, min(1.0, x)), max(0.0, min(1.0, y)))
    except (ValueError, TypeError):
        return None


def parse_seeclick_output(text: str) -> Point:
    """SeeClick: '(x, y)' or '[x, y]' in normalized [0, 1].

    Reference: https://github.com/njucckevin/SeeClick
    """
    match = re.search(r'[\(\[]\s*([\d.]+)\s*,\s*([\d.]+)\s*[\)\]]', text)
    if match:
        x, y = float(match.group(1)), float(match.group(2))
        if 0 <= x <= 1 and 0 <= y <= 1:
            return (x, y)
    return None


def parse_qwen2_vl_output(text: str, image_w: int = None, image_h: int = None) -> Point:
    """Qwen2-VL: '<|box_start|>(x1,y1),(x2,y2)<|box_end|>' or plain bbox.

    Returns box center in normalized coords.
    """
    match = re.search(r'\((\d+),\s*(\d+)\),\s*\((\d+),\s*(\d+)\)', text)
    if match:
        x1, y1, x2, y2 = (int(m) for m in match.groups())
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        if image_w and image_h:
            return (cx / image_w, cy / image_h)
        # Qwen2-VL typically uses 0-1000 internal coords
        return (cx / 1000, cy / 1000)
    return None


def parse_os_atlas_output(text: str) -> Point:
    """OS-Atlas: structured bbox or point output, normalized.

    Reference: https://github.com/OS-Copilot/OS-Atlas
    """
    # OS-Atlas uses <|box_start|>(x1,y1),(x2,y2)<|box_end|> in internal 0-1000 scale
    return parse_qwen2_vl_output(text)
