"""
Test Suite: OpenVINO NPU Compilation & Blob Cache Hit Latency (<25ms).
"""

import pytest
from pathlib import Path
from src.hardware_rig.compilation_rig import NPUCompilationRig
from src.hardware_rig.config import COMPILATION_CONFIG, MODEL_SPECS


class TestNPUCompilationCache:
    @classmethod
    def setup_class(cls):
        cls.rig = NPUCompilationRig()

    def test_npu_hardware_readiness(self):
        """Verifies that Intel Lunar Lake NPU 4000 is detected and operational."""
        status = self.rig.verify_hardware_readiness()
        assert status["ready"] is True, f"NPU not available. Available: {status['available_devices']}"
        assert status["target_device"] == "NPU"
        assert status["details"]["architecture"] == "4000"
        assert status["details"]["max_tiles"] == 6

    def test_cached_blobs_exist_and_valid(self):
        """Verifies that precompiled .blob files exist in cache and meet min size."""
        blobs = self.rig.inspect_cached_blobs()
        assert len(blobs) >= 1, "No cached hardware blobs found in cache directory"
        for b in blobs:
            assert b["valid_size"] is True, f"Blob {b['filename']} is below min size ({b['size_bytes']} bytes)"
            assert len(b["header_hex"]) > 0

    def test_cache_hit_latency_sub_25ms_fastminilm(self):
        """
        Hard Gate: Tests that warm cache hit compilation for FastMiniLM is strictly under 25ms.
        """
        res = self.rig.validate_cache_hit_latency("fastminilm", warmup_compile=True)
        assert res["passed"] is True, (
            f"Cache hit compilation failed budget: {res['cache_hit_time_ms']}ms >= {res['budget_ms']}ms"
        )
        assert res["cache_hit_time_ms"] < COMPILATION_CONFIG.cache_hit_budget_ms
        assert res["request_creation_time_ms"] < 5.0
        assert res["first_infer_time_ms"] < 35.0

    def test_cache_hit_latency_sub_25ms_mobilenet(self):
        """
        Hard Gate: Tests that warm cache hit compilation for MobileNet is strictly under 25ms.
        """
        res = self.rig.validate_cache_hit_latency("mobilenet", warmup_compile=True)
        assert res["passed"] is True, (
            f"Cache hit compilation failed budget: {res['cache_hit_time_ms']}ms >= {res['budget_ms']}ms"
        )
        assert res["cache_hit_time_ms"] < COMPILATION_CONFIG.cache_hit_budget_ms
