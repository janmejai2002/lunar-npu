"""
Unit and physical silicon tests for Sprint 2 (JOB P0-02):
Compiling Micro-LoRA Adjoint Backward Graph into OpenVINO NPU IR.
"""

import json
import tempfile
from pathlib import Path
import numpy as np
import pytest

from lunar_core.engine import LunarNPUEngine
from lunar_core.micro_lora import (
    MicroLoRAEngine,
    SRAMAdamW,
    build_lora_forward_openvino_model,
    build_lora_backward_openvino_model,
)


def test_build_lora_forward_openvino_model():
    """Verify forward OpenVINO model compiles and matches NumPy forward pass."""
    batch_size = 2
    in_features = 64
    out_features = 64
    rank = 4
    scaling = 2.0

    model = build_lora_forward_openvino_model(
        batch_size=batch_size,
        in_features=in_features,
        out_features=out_features,
        rank=rank,
        scaling=scaling,
    )
    assert model.get_friendly_name() == "MicroLoRAForward"
    assert len(model.inputs) == 4
    assert len(model.outputs) == 2

    engine = LunarNPUEngine()
    compiled = engine.compile_model(model)
    req = compiled.create_infer_request()

    np.random.seed(42)
    x = np.random.randn(batch_size, in_features).astype(np.float32)
    W_0 = (np.random.randn(out_features, in_features) * 0.02).astype(np.float32)
    A = np.random.randn(rank, in_features).astype(np.float32)
    B = np.random.randn(out_features, rank).astype(np.float32)

    res = req.infer({"x": x, "W_0": W_0, "A": A, "B": B})
    y_ov = res[compiled.output(0)]
    mid_ov = res[compiled.output(1)]

    # Ground truth NumPy
    base_np = np.matmul(x, W_0.T)
    mid_np = np.matmul(x, A.T)
    y_np = base_np + scaling * np.matmul(mid_np, B.T)

    np.testing.assert_allclose(mid_ov, mid_np, rtol=1e-2, atol=1e-2)
    np.testing.assert_allclose(y_ov, y_np, rtol=1e-2, atol=1e-2)
    cos_sim = float(np.dot(y_ov.flatten(), y_np.flatten()) / (np.linalg.norm(y_ov) * np.linalg.norm(y_np)))
    assert cos_sim > 0.999


def test_build_lora_backward_openvino_model():
    """Verify Adjoint backward OpenVINO model compiles on silicon and matches math ground truth."""
    batch_size = 2
    in_features = 64
    out_features = 64
    rank = 4
    scaling = 2.0

    model = build_lora_backward_openvino_model(
        batch_size=batch_size,
        in_features=in_features,
        out_features=out_features,
        rank=rank,
        scaling=scaling,
    )
    assert model.get_friendly_name() == "MicroLoRABackward"
    assert len(model.inputs) == 4
    assert len(model.outputs) == 2

    engine = LunarNPUEngine()
    compiled = engine.compile_model(model)
    req = compiled.create_infer_request()

    np.random.seed(1337)
    delta = (np.random.randn(batch_size, out_features) * 0.05).astype(np.float32)
    lora_mid = (np.random.randn(batch_size, rank) * 0.1).astype(np.float32)
    x = (np.random.randn(batch_size, in_features) * 0.1).astype(np.float32)
    B = (np.random.randn(out_features, rank) * 0.1).astype(np.float32)

    res = req.infer({"delta": delta, "lora_mid": lora_mid, "x": x, "B": B})
    grad_A_ov = res[compiled.output(0)]
    grad_B_ov = res[compiled.output(1)]

    # Mathematical Adjoint Forward GEMMs
    grad_B_np = scaling * np.matmul(delta.T, lora_mid)
    delta_B_np = np.matmul(delta, B)
    grad_A_np = scaling * np.matmul(delta_B_np.T, x)

    # Cosine similarity on physical accelerator
    cos_sim_A = float(np.dot(grad_A_ov.flatten(), grad_A_np.flatten()) / (np.linalg.norm(grad_A_ov) * np.linalg.norm(grad_A_np)))
    cos_sim_B = float(np.dot(grad_B_ov.flatten(), grad_B_np.flatten()) / (np.linalg.norm(grad_B_ov) * np.linalg.norm(grad_B_np)))

    assert cos_sim_A > 0.999
    assert cos_sim_B > 0.999


def test_sram_adamw_inplace_zero_dram_reallocation():
    """Verify SRAMAdamW updates memory strictly in-place without re-allocating array buffers."""
    opt = SRAMAdamW(lr=1e-3)
    A = np.ones((4, 32), dtype=np.float32)
    B = np.zeros((32, 4), dtype=np.float32)

    A_ptr = A.__array_interface__["data"][0]
    B_ptr = B.__array_interface__["data"][0]

    params = {"A": A, "B": B}
    grads = {
        "A": np.ones_like(A) * 0.01,
        "B": np.ones_like(B) * 0.02,
    }

    for _ in range(10):
        opt.step(params, grads)
        # Buffer addresses must remain strictly invariant
        assert A.__array_interface__["data"][0] == A_ptr
        assert B.__array_interface__["data"][0] == B_ptr

    assert opt.step_count == 10
    assert not np.allclose(A, 1.0)
    assert not np.allclose(B, 0.0)


def test_micro_lora_engine_silicon_train_step():
    """Verify MicroLoRAEngine executes train_step on NPU silicon with valid telemetry."""
    engine = LunarNPUEngine()
    lora = MicroLoRAEngine(in_features=64, out_features=64, rank=4, alpha=8.0, engine=engine)

    x = np.ones((2, 64), dtype=np.float32) * 0.05
    y_tgt = np.ones((2, 64), dtype=np.float32) * 0.15

    step_info = lora.train_step(x, y_tgt)
    assert step_info["step"] == 1
    assert step_info["loss"] >= 0.0
    assert step_info["latency_ms"] > 0.0
    assert step_info["backward_device"] in ["NPU", "GPU", "CPU"]
    assert step_info["grad_norm_B"] > 0.0


def test_micro_lora_train_on_session_logs():
    """Verify training on real user correction trajectories from session logs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        log_file = Path(tmpdir) / "test_session_audit.jsonl"
        entries = [
            {"command": "git commit -m 'test'", "verdict": "ALLOWED"},
            {"command": "python -m pytest", "verdict": "ALLOWED"},
            {"command": "rm -rf /", "verdict": "BLOCKED"},
            {"command": "lunar status", "verdict": "ALLOWED"},
            {"command": "mkfs.vfat /dev/sda", "verdict": "BLOCKED"},
        ]
        with open(log_file, "w", encoding="utf-8") as f:
            for e in entries:
                f.write(json.dumps(e) + "\n")

        lora = MicroLoRAEngine(in_features=64, out_features=64, rank=4)
        summary = lora.train_on_session_logs(log_path=log_file, steps_limit=5)

        assert summary["steps"] == 5
        assert "mean_step_latency_ms" in summary
        assert summary["mean_step_latency_ms"] > 0.0
        assert "backward_device" in summary
        assert summary["source"] == str(log_file)


def test_micro_lora_batch_shapes_caching():
    """Verify dynamic batch shapes are compiled and cached cleanly on engine."""
    lora = MicroLoRAEngine(in_features=32, out_features=32, rank=4)

    # 1D single vector
    x0 = np.random.randn(32).astype(np.float32)
    y0 = lora.forward(x0)
    assert y0.shape == (32,)

    # Batch size 1 (2D)
    x1 = np.random.randn(1, 32).astype(np.float32)
    y1 = lora.forward(x1)
    assert y1.shape == (1, 32)

    # Batch size 2
    x2 = np.random.randn(2, 32).astype(np.float32)
    y2 = lora.forward(x2)
    assert y2.shape == (2, 32)

    # Batch size 4
    x4 = np.random.randn(4, 32).astype(np.float32)
    y4 = lora.forward(x4)
    assert y4.shape == (4, 32)

    assert 1 in lora._compiled_forward
    assert 2 in lora._compiled_forward
    assert 4 in lora._compiled_forward
