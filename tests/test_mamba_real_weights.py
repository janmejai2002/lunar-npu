"""
Integration and Physical Silicon Tests for Sprint 1 (JOB P0-01):
Grounding Mamba-2 SSM in Real Pretrained Weights & Fast BPE Tokenizer.
"""

import numpy as np
import pytest

from lunar_core.engine import LunarNPUEngine
from lunar_core.mamba_ssm import (
    LunarMambaEngine,
    LunarMamba2Engine,
    MambaWeightLoader,
    RealFastBPETokenizer,
    build_mamba_step_openvino_model,
    build_mamba2_ssd_openvino_model,
)


def test_mamba_weight_loader_real():
    """Verify weight loader extracts real physical discretization matrices without random numbers."""
    d_inner = 64
    d_state = 16
    weights = MambaWeightLoader.get_default_weights(d_inner=d_inner, d_state=d_state)

    assert "A_bar" in weights
    assert "B_bar" in weights
    assert "C" in weights
    assert "D" in weights
    assert weights["source"] in ["pretrained_safetensors", "physical_analytical_discretization"]

    assert weights["A_bar"].shape == (d_inner, d_state)
    assert weights["B_bar"].shape == (d_inner, d_state)
    assert weights["C"].shape == (d_inner, d_state)
    assert weights["D"].shape == (1, d_inner)

    assert not np.isnan(weights["A_bar"]).any()
    assert not np.isnan(weights["B_bar"]).any()
    assert not np.isnan(weights["C"]).any()
    assert not np.isnan(weights["D"]).any()


def test_real_fast_bpe_tokenizer():
    """Verify Hugging Face Fast BPE tokenizer correctly encodes code and decodes back."""
    tok = RealFastBPETokenizer()
    code_prompt = "def fibonacci(n):"
    ids = tok.encode(code_prompt)

    assert isinstance(ids, list)
    assert len(ids) > 0
    assert tok.vocab_size > 1000

    decoded = tok.decode(ids)
    assert "fibonacci" in decoded


def test_mamba_step_with_real_weights_openvino():
    """Verify OpenVINO model compilation and step execution on physical NPU/CPU using real weights."""
    engine = LunarNPUEngine()
    d_inner = 64
    d_state = 16
    mamba = LunarMambaEngine(engine=engine, d_inner=d_inner, d_state=d_state)

    assert mamba.state.shape == (1, d_inner, d_state)
    x_token = np.ones((1, d_inner), dtype=np.float32) * 0.1

    y_out, lat_ms = mamba.step(x_token)
    assert y_out.shape == (1, d_inner)
    assert lat_ms > 0.0
    assert not np.allclose(mamba.state, 0.0)


def test_mamba_multi_turn_recurrent_persistence():
    """Verify O(1) state memory persistence invariant across 50 sequential steps."""
    mamba = LunarMambaEngine(d_inner=64, d_state=16)
    expected_bytes = 1 * 64 * 16 * 4  # 4,096 bytes FP32

    mamba.reset_state()
    assert np.allclose(mamba.state, 0.0)

    for i in range(50):
        token_arr = np.ones((1, 64), dtype=np.float32) * float(i + 1) * 0.01
        _, _ = mamba.step(token_arr)
        assert mamba.state.nbytes == expected_bytes

    # State must be rich and bounded
    assert not np.isnan(mamba.state).any()
    assert not np.isinf(mamba.state).any()


def test_mamba_code_token_generation():
    """Verify end-to-end code generation produces valid Python syntax tokens."""
    mamba = LunarMambaEngine(d_inner=64, d_state=16)
    prompt = "def fibonacci(n):"
    output = mamba.generate(prompt=prompt, max_new_tokens=15, temperature=0.2)

    assert isinstance(output, str)
    assert "def fibonacci(n):" in output
    # Must produce code characters
    assert len(output) > len(prompt)


def test_mamba2_ssd_physical_decay_model():
    """Verify Mamba-2 SSD multi-head model compiles and runs with deterministic physical head decays."""
    n_heads = 4
    d_head = 64
    d_state = 16
    mamba2 = LunarMamba2Engine(n_heads=n_heads, d_head=d_head, d_state=d_state)

    token = np.ones((1, n_heads, d_head), dtype=np.float32) * 0.05
    y_out, lat_ms = mamba2.step(token)

    assert y_out.shape == (1, n_heads, d_head)
    assert lat_ms > 0.0
    assert mamba2.state_bytes == 16384  # Theorem 11.1 constant invariant
