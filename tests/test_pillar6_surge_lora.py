"""
Test Pillar 6: Dual-Profile Governor (Ambient vs Surge), Micro-LoRA, GhostHUD & FastMCP
======================================================================================
Verifies:
1. Dual-Profile NPU Governor (Ambient 2.5W vs Surge 47 TOPS Max)
2. Mamba-2 3-Phase Chunked Systolic GEMM sequence processor
3. On-Device Micro-LoRA continuous backpropagation via Adjoint Forward Graph
4. DirectComposition Transparent GhostHUD (WDA_EXCLUDEFROMCAPTURE)
5. FastMCP 2.0 universal client auto-installer & status inspection
6. New CLI subcommands (profile, lora, ghost-hud, install-mcp)
"""

import json
import tempfile
import numpy as np
import pytest
from click.testing import CliRunner

from lunar_core.cli import cli
from lunar_core.engine import LunarNPUEngine, NPUProfile
from lunar_core.power_telemetry import get_power_governor, RAPLPowerGovernor
from lunar_core.mamba_ssm import LunarMamba2Engine
from lunar_core.micro_lora import MicroLoRAEngine, SRAMAdamW
from lunar_core.ghost_hud import GhostHUDController, get_ghost_hud
from lunar_core.install_mcp import inspect_client_status, install_lunar_mcp


def test_npu_governor_profile_switching():
    engine = LunarNPUEngine()
    
    # Switch to Ambient Mode
    res_amb = engine.set_profile("ambient")
    assert engine.current_profile == "ambient"
    assert res_amb["profile"] == "ambient"
    assert res_amb["max_tiles"] == 2
    assert res_amb["turbo"] is False
    assert res_amb["target_power_w"] == 2.50

    # Switch to Lunar Surge Mode (Max 47 TOPS)
    res_surge = engine.set_profile("surge")
    assert engine.current_profile == "surge"
    assert res_surge["profile"] == "surge"
    assert res_surge["max_tiles"] == 6
    assert res_surge["turbo"] is True
    assert res_surge["target_power_w"] == 28.0


def test_rapl_power_governor_evaluation():
    gov = get_power_governor()
    
    amb_eval = gov.evaluate("ambient")
    assert amb_eval["profile"] == "ambient"
    assert amb_eval["target_power_w"] == 2.50
    assert "status" in amb_eval

    surge_eval = gov.evaluate("surge")
    assert surge_eval["profile"] == "surge"
    assert surge_eval["target_power_w"] == 28.0
    assert "LUNAR_SURGE_UNLEASHED" in surge_eval["status"]
    assert surge_eval["peak_int8_tops"] > 40.0


def test_mamba2_chunked_systolic_gemm():
    engine = LunarNPUEngine()
    mamba2 = LunarMamba2Engine(engine=engine, n_heads=4, d_head=64, d_state=16)

    # 128 tokens sequence
    seq_tokens = np.random.randn(128, 256).astype(np.float32)
    out, metrics = mamba2.forward_chunked_ssd(seq_tokens, chunk_size=32)

    assert out.shape == (128, 4, 64)
    assert metrics["sequence_length"] == 128
    assert metrics["num_chunks"] == 4
    assert metrics["latency_ms"] > 0
    assert metrics["tokens_per_second"] > 0
    assert metrics["algorithm"] == "Mamba-2-SSD-Chunked-GEMM"


def test_micro_lora_continuous_adaptation():
    engine = LunarNPUEngine()
    lora = MicroLoRAEngine(in_features=64, out_features=64, rank=4, alpha=8.0, engine=engine)

    # Forward pass
    x = np.random.randn(2, 64).astype(np.float32)
    y = lora.forward(x)
    assert y.shape == (2, 64)

    # Single train step
    target = np.random.randn(2, 64).astype(np.float32)
    step_res = lora.train_step(x, target)
    assert step_res["step"] == 1
    assert "loss" in step_res
    assert step_res["loss"] >= 0.0
    assert step_res["latency_ms"] > 0.0

    # Benchmark adaptation convergence
    bench = lora.benchmark_adaptation(steps=15, batch_size=2)
    assert bench["steps"] == 15
    assert bench["throughput_tokens_per_sec"] > 0
    assert bench["sram_footprint_bytes"] > 0


def test_micro_lora_checkpoint_save_and_load():
    with tempfile.TemporaryDirectory() as tmpdir:
        lora1 = MicroLoRAEngine(in_features=32, out_features=32, rank=4)
        x = np.random.randn(1, 32).astype(np.float32)
        y1_before = lora1.forward(x)

        # Train a step
        lora1.train_step(x, x * 2.0)
        y1_after = lora1.forward(x)

        # Save
        paths = lora1.save_checkpoint(tmpdir)
        assert "weights_file" in paths
        assert "config_file" in paths

        # Load into new instance
        lora2 = MicroLoRAEngine(in_features=32, out_features=32, rank=4)
        lora2.load_checkpoint(tmpdir)
        y2 = lora2.forward(x)

        np.testing.assert_allclose(y1_after, y2, rtol=1e-4)


def test_ghosthud_controller_and_masking():
    hud = GhostHUDController(width=400, height=200)
    status_init = hud.get_status()
    assert status_init["is_running"] is False
    assert status_init["click_through"] is True

    # Post message
    post_res = hud.post_message("Test Hint", role="assistant")
    assert post_res["status"] == "posted"
    assert post_res["total_messages"] == 1

    # Run self-contained demo
    demo_res = hud.run_demo(duration_s=0.2)
    assert "glass_to_glass_budget_ms" in demo_res
    assert demo_res["glass_to_glass_budget_ms"] < 20.0
    assert "WDA_EXCLUDEFROMCAPTURE" in demo_res["display_affinity"] or not hud.is_windows


def test_fastmcp_installer_and_inspection():
    clients = inspect_client_status()
    assert isinstance(clients, list)
    assert len(clients) >= 4

    client_ids = [c["id"] for c in clients]
    assert "claude" in client_ids
    assert "cursor" in client_ids

    # Dry-run install
    install_res = install_lunar_mcp(client_target="all", dry_run=True)
    assert install_res["dry_run"] is True
    assert len(install_res["results"]) >= 4


def test_cli_profile_command():
    runner = CliRunner()
    # Read profile
    r1 = runner.invoke(cli, ["profile", "--json"])
    assert r1.exit_code == 0
    d1 = json.loads(r1.output)
    assert "current_profile" in d1

    # Set profile to surge
    r2 = runner.invoke(cli, ["profile", "--set", "surge", "--json"])
    assert r2.exit_code == 0
    d2 = json.loads(r2.output)
    assert d2["profile"] == "surge"


def test_cli_lora_command():
    runner = CliRunner()
    res = runner.invoke(cli, ["lora", "--steps", "5", "--surge", "--json"])
    assert res.exit_code == 0
    data = json.loads(res.output)
    assert data["steps"] == 5
    assert data["rank"] == 8
    assert data["profile"] == "surge"


def test_cli_ghost_hud_command():
    runner = CliRunner()
    res = runner.invoke(cli, ["ghost-hud", "--text", "Test Note", "--json"])
    assert res.exit_code == 0
    data = json.loads(res.output)
    assert data["status"] == "posted"


def test_cli_install_mcp_command():
    runner = CliRunner()
    res = runner.invoke(cli, ["install-mcp", "--status", "--json"])
    assert res.exit_code == 0
    data = json.loads(res.output)
    assert isinstance(data, list)
