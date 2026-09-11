"""
Unit and physical integration tests for Sprint 3 (JOB P1-01):
Native DirectX DXGI Desktop Duplication Capture Engine.
"""

import numpy as np
import pytest
from PIL import Image

from lunar_core.dxgi_capture import (
    DXGICaptureEngine,
    DXGIFrame,
    get_dxgi_capture_engine,
)
from lunar_core.vision import LunarVisionEngine


def test_dxgi_capture_initialization():
    """Verify DXGICaptureEngine initializes cleanly and detects platform capabilities."""
    engine = DXGICaptureEngine()
    assert engine is not None
    assert engine.active_method in ["dxgi_directx", "win32_dib", "pil_fallback"]


def test_dxgi_capture_frame_structure():
    """Verify captured desktop surface contains valid dimensions, latency, and numpy data."""
    engine = DXGICaptureEngine()
    frame = engine.capture_frame()

    assert isinstance(frame, DXGIFrame)
    assert frame.width > 0
    assert frame.height > 0
    assert frame.latency_ms >= 0.0
    assert frame.data is not None
    assert isinstance(frame.data, np.ndarray)
    assert frame.data.shape[0] == frame.height
    assert frame.data.shape[1] == frame.width
    assert frame.method in ["dxgi_directx", "win32_dib", "pil_fallback", "mock_frame"]


def test_dxgi_frame_to_pil_and_rgb():
    """Verify frame conversion to PIL Image and RGB numpy array."""
    # Test synthetic frame conversion
    arr = np.zeros((100, 100, 4), dtype=np.uint8)
    arr[..., 0] = 255  # Blue in BGRA
    arr[..., 3] = 255

    frame = DXGIFrame(
        data=arr,
        width=100,
        height=100,
        latency_ms=1.5,
        method="win32_dib",
        timestamp=1000.0,
    )

    pil_img = frame.to_pil()
    assert isinstance(pil_img, Image.Image)
    assert pil_img.size == (100, 100)

    rgb = frame.to_rgb_array()
    assert isinstance(rgb, np.ndarray)
    assert rgb.shape == (100, 100, 3)
    # Blue channel in RGB (index 2) should match Blue in BGRA (index 0)
    assert rgb[0, 0, 2] == 255


def test_dxgi_capture_benchmark():
    """Verify benchmark mode returns valid FPS, resolution, and memory statistics."""
    engine = DXGICaptureEngine()
    bench = engine.benchmark(num_frames=5)

    assert bench["num_frames"] == 5
    assert "resolution" in bench
    assert "fps" in bench
    assert bench["fps"] > 0.0
    assert "mean_latency_ms" in bench
    assert bench["mean_latency_ms"] > 0.0
    assert bench["frame_memory_bytes"] > 0


def test_vision_engine_integration_dxgi():
    """Verify LunarVisionEngine seamlessly consumes frames from DXGICaptureEngine."""
    vision = LunarVisionEngine()
    screen_img = vision.capture_screen()
    assert isinstance(screen_img, Image.Image)
    assert screen_img.width > 0
    assert screen_img.height > 0

    # Analyze active desktop
    result = vision.analyze(image_input=None)
    assert result is not None
    assert result.image_width == screen_img.width
    assert result.image_height == screen_img.height
    assert "active_desktop" in result.source
    assert result.latency_ms > 0.0


def test_dxgi_capture_release():
    """Verify clean release of COM resources without throwing exceptions."""
    engine = DXGICaptureEngine()
    engine.capture_frame()
    engine.release()
    assert engine.is_dxgi_available is False
    assert engine.duplication is None
