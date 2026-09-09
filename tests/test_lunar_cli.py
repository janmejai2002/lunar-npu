"""Unit tests for Lunar Core Command-Line Interface and benchmark suite."""

import json
from click.testing import CliRunner
from lunar_core.cli import cli


def test_cli_status():
    runner = CliRunner()
    result = runner.invoke(cli, ["status", "--json"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert "device" in data
    assert "available_devices" in data


def test_cli_embed():
    runner = CliRunner()
    result = runner.invoke(cli, ["embed", "Intel Lunar Lake physical silicon acceleration", "--json"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert data["dimension"] in (384, 768)
    assert data["latency_ms"] > 0.0
    assert abs(data["l2_norm"] - 1.0) < 1e-4


def test_cli_mamba():
    runner = CliRunner()
    result = runner.invoke(cli, ["mamba", "--steps", "20", "--json"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert data["steps"] == 20
    assert "mean_step_latency_ms" in data
    assert data["tokens_per_second"] > 0.0


def test_cli_route():
    runner = CliRunner()
    result = runner.invoke(cli, ["route", "Write a python function to implement quicksort", "--json"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert data["target_agent"] == "CODER"
    assert data["confidence"] > 0.3
    assert data["latency_ms"] > 0.0


def test_cli_audit():
    runner = CliRunner()
    res_safe = runner.invoke(cli, ["audit", "git status", "--json"])
    assert res_safe.exit_code == 0
    data_safe = json.loads(res_safe.output)
    assert data_safe["verdict"] == "ALLOWED"

    res_hazard = runner.invoke(cli, ["audit", "rm -rf /", "--json"])
    assert res_hazard.exit_code == 0
    data_hazard = json.loads(res_hazard.output)
    assert data_hazard["verdict"] == "BLOCKED"


def test_cli_benchmark():
    runner = CliRunner()
    result = runner.invoke(cli, ["benchmark", "--mamba-steps", "20", "--audit-runs", "50", "--json"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert "hardware" in data
    assert "vector_memory" in data
    assert "mamba_ssm" in data
    assert "micro_router" in data
    assert "circuit_breaker" in data
    assert data["overall_status"] == "QUALIFIED_PRODUCTION_GRADE"
    assert data["circuit_breaker"]["accuracy_pct"] == 100.0
    assert data["micro_router"]["routing_accuracy_pct"] == 100.0


def test_cli_stress():
    runner = CliRunner()
    result = runner.invoke(cli, ["stress", "--iterations", "5", "--json"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert data["iterations"] == 5
    assert "sustained_tflops" in data
    assert "package_power_w" in data


def test_cli_power():
    runner = CliRunner()
    result = runner.invoke(cli, ["power", "--json"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert "package_power_w" in data
    assert "sensor_backend" in data


def test_cli_audits():
    runner = CliRunner()
    result = runner.invoke(cli, ["audits", "--limit", "5", "--json"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert isinstance(data, list)
