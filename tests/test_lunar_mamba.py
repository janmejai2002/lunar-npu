"""Unit and integration tests for LunarMambaEngine."""

import numpy as np
import pytest
from lunar_core.mamba_ssm import LunarMambaEngine, build_mamba_step_openvino_model


def test_build_mamba_model():
    model = build_mamba_step_openvino_model(d_inner=32, d_state=8)
    assert model is not None
    assert len(model.inputs) == 2
    assert len(model.outputs) == 2


def test_mamba_engine_step():
    d_inner = 32
    d_state = 8
    mamba = LunarMambaEngine(d_inner=d_inner, d_state=d_state)
    assert mamba.state.shape == (1, d_inner, d_state)

    x = np.random.randn(1, d_inner).astype(np.float32)
    y_out, lat = mamba.step(x)

    assert y_out.shape == (1, d_inner)
    assert lat > 0.0
    # Recurrent state should have updated from zero
    assert not np.allclose(mamba.state, 0.0)


def test_mamba_reset_state():
    mamba = LunarMambaEngine(d_inner=16, d_state=4)
    x = np.random.randn(1, 16).astype(np.float32)
    mamba.step(x)
    assert not np.allclose(mamba.state, 0.0)

    mamba.reset_state()
    assert np.allclose(mamba.state, 0.0)


def test_mamba_benchmark():
    mamba = LunarMambaEngine(d_inner=16, d_state=4)
    bench = mamba.benchmark(num_steps=20)
    assert bench["steps"] == 20
    assert bench["mean_step_latency_ms"] > 0.0
    assert bench["tokens_per_second"] > 0.0
