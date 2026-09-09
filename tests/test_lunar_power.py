"""Unit and integration tests for LunarPowerTelemetry."""

import pytest
from lunar_core.power_telemetry import LunarPowerTelemetry, get_power_telemetry


def test_power_telemetry_instance():
    telemetry = LunarPowerTelemetry()
    assert telemetry is not None
    # Close resources cleanly
    telemetry.close()


def test_power_telemetry_sample_schema():
    telemetry = get_power_telemetry()
    sample = telemetry.sample()

    required_keys = [
        "is_live",
        "sensor_backend",
        "package_power_w",
        "core_power_w",
        "uncore_power_w",
        "dram_power_w",
        "npu_power_est_w",
        "temperature_c",
        "timestamp",
    ]
    for k in required_keys:
        assert k in sample, f"Missing key {k} in sample"

    assert isinstance(sample["is_live"], bool)
    assert sample["package_power_w"] >= 0.0
    assert sample["core_power_w"] >= 0.0
    assert sample["uncore_power_w"] >= 0.0
    assert sample["dram_power_w"] >= 0.0
    assert sample["npu_power_est_w"] >= 0.0
    assert sample["temperature_c"] >= 0.0


def test_power_telemetry_simulated_fallback():
    telemetry = LunarPowerTelemetry()
    sim = telemetry._simulated_sample(100.0)
    assert sim["is_live"] is False
    assert "package_power_w" in sim
    assert sim["package_power_w"] == 18.5
    assert sim["npu_power_est_w"] == 2.2
    telemetry.close()
