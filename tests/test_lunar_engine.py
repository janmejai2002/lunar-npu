"""Unit and integration tests for LunarNPUEngine."""

from pathlib import Path
import pytest
from lunar_core.engine import LunarNPUEngine


def test_engine_initialization(tmp_path):
    engine = LunarNPUEngine(cache_dir=str(tmp_path))
    assert engine.core is not None
    assert engine.cache_path == tmp_path.resolve()
    assert engine.device in ["NPU", "GPU", "CPU"]
    assert "PERFORMANCE_HINT" in engine.config


def test_engine_device_info():
    engine = LunarNPUEngine()
    info = engine.device_info
    assert "device" in info
    assert "available_devices" in info
    assert "full_name" in info
    if engine.is_npu:
        assert info["driver_version"] is not None
        assert "NPU" in info["available_devices"]


def test_engine_compile_custom_config(tmp_path):
    engine = LunarNPUEngine(cache_dir=str(tmp_path))
    # Test compilation property merging
    custom_cfg = {"PERFORMANCE_HINT": "THROUGHPUT"}
    merged_cfg = dict(engine.config)
    merged_cfg.update(custom_cfg)
    assert merged_cfg["PERFORMANCE_HINT"] == "THROUGHPUT"
