"""Unit and integration tests for Pillar 5: Packaging, CLI Integration & Hardware Qualification."""

import json
from click.testing import CliRunner
from lunar_core.cli import cli
from lunar_core.benchmark import run_hardware_qualification_suite


def test_hardware_qualification_suite_quick():
    res = run_hardware_qualification_suite(quick=True)
    assert res is not None
    assert "hardware" in res
    assert "pillar1_usm_shave" in res
    assert "pillar2_mamba2_pq8" in res
    assert "pillar3_ocr_sovereign" in res
    assert "pillar4_circuit_breaker_router" in res
    assert "pillar5_power_thermals" in res
    assert res["overall_status"] == "SOVEREIGN_PRODUCTION_QUALIFIED"


def test_cli_mamba2():
    runner = CliRunner()
    result = runner.invoke(cli, ["mamba2", "--steps", "10", "--restore-runs", "5", "--json"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert data["model"] == "Mamba-2 SSD (Static OpenVINO IR)"
    assert data["state_shape"] == [1, 4, 64, 16]
    assert data["uma_restore_latency_us"] < 25.0
    assert data["qualification_status"] == "QUALIFIED_PRODUCTION_GRADE"


def test_cli_pq8():
    runner = CliRunner()
    result = runner.invoke(cli, ["pq8", "--items", "1000", "--json"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert data["dimension"] == 384
    assert data["subspaces"] == 48
    assert data["compression_ratio"] == "32.0x"
    assert data["mean_scan_latency_ms"] >= 0.0


def test_cli_ocr():
    runner = CliRunner()
    result = runner.invoke(cli, ["ocr", "--json"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert "full_text" in data
    assert "total_latency_ms" in data
    assert len(data["lines"]) > 0


def test_cli_circuit_breaker():
    runner = CliRunner()
    res_safe = runner.invoke(cli, ["circuit-breaker", "git status", "--json"])
    assert res_safe.exit_code == 0
    data_safe = json.loads(res_safe.output)
    assert data_safe["verdict"] == "ALLOWED"


def test_cli_router():
    runner = CliRunner()
    result = runner.invoke(cli, ["router", "Write a python function to implement quicksort", "--adapt", "--json"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert data["target_agent"] == "CODER"
    assert "riemannian_adaptation" in data
    assert data["riemannian_adaptation"]["persona"] == "CODER"


def test_cli_swarm_cycle():
    runner = CliRunner()
    result = runner.invoke(cli, ["swarm-cycle", "Build a microservice", "--max-iterations", "2", "--json"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert "task" in data
    assert "converged" in data
    assert "trajectory" in data


def test_cli_benchmark_all():
    runner = CliRunner()
    result = runner.invoke(cli, ["benchmark-all", "--quick", "--json"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert data["overall_status"] == "SOVEREIGN_PRODUCTION_QUALIFIED"
