"""Unit and integration tests for LunarVision edge screen perception and YOLO detection."""

import json
import pytest
from click.testing import CliRunner

from lunar_core.vision import LunarVisionEngine, VisionAnalysisResult, UIElement
from lunar_core.cli import cli


def test_vision_engine_initialization():
    vision = LunarVisionEngine()
    assert vision.engine is not None
    assert vision.device in ("NPU", "GPU", "CPU")


def test_vision_analyze_active_desktop():
    vision = LunarVisionEngine()
    res = vision.analyze(image_input=None)
    assert isinstance(res, VisionAnalysisResult)
    assert res.source == "active_desktop"
    assert res.image_width > 0
    assert res.image_height > 0
    assert res.latency_ms > 0.0
    assert isinstance(res.elements, list)
    assert res.elements_detected >= 0
    if res.elements:
        first = res.elements[0]
        assert isinstance(first, UIElement)
        assert "x" in first.bounding_box
        assert "y" in first.bounding_box


def test_vision_cli_screen():
    runner = CliRunner()
    result = runner.invoke(cli, ["screen", "--json"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert data["source"] == "active_desktop"
    assert "elements_detected" in data
    assert "elements" in data
