"""Unit tests for coordinate parsers."""

import pytest

from benchmark.parsing.coordinate_parsers import (
    parse_moondream_loc_tokens,
    parse_florence_loc_tokens,
    parse_json_point,
    parse_seeclick_output,
    parse_qwen2_vl_output,
)


def test_moondream_basic():
    assert parse_moondream_loc_tokens("<loc_500><loc_250>") == pytest.approx((0.5005, 0.2503), rel=1e-3)


def test_moondream_with_text():
    assert parse_moondream_loc_tokens("answer: <loc_100><loc_200>") is not None


def test_moondream_missing():
    assert parse_moondream_loc_tokens("no coordinates here") is None


def test_florence_bbox_first():
    out = parse_florence_loc_tokens("<loc_100><loc_200><loc_300><loc_400>", take="first")
    assert out == pytest.approx((0.1001, 0.2002), rel=1e-3)


def test_florence_bbox_center():
    out = parse_florence_loc_tokens("<loc_100><loc_200><loc_300><loc_400>", take="center")
    assert out == pytest.approx((0.2002, 0.3003), rel=1e-3)


def test_json_normalized():
    assert parse_json_point('{"x": 0.5, "y": 0.3}') == (0.5, 0.3)


def test_json_pixels():
    assert parse_json_point('{"x": 960, "y": 540}', image_w=1920, image_h=1080) == (0.5, 0.5)


def test_json_missing():
    assert parse_json_point("not json") is None


def test_seeclick_parens():
    assert parse_seeclick_output("(0.5, 0.3)") == (0.5, 0.3)


def test_seeclick_brackets():
    assert parse_seeclick_output("[0.25, 0.75]") == (0.25, 0.75)


def test_qwen_bbox():
    out = parse_qwen2_vl_output("<|box_start|>(100,200),(300,400)<|box_end|>")
    assert out == (0.2, 0.3)
