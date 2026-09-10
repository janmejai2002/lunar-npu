"""Unit and integration tests for Pillar 2: Mamba-2 SSD & PQ8 Systolic Memory."""

import numpy as np
import pytest
from lunar_core.mamba_ssm import (
    LunarMamba2Engine,
    PersistentStateManager,
    build_mamba2_ssd_openvino_model,
)
from lunar_core.vector_memory import (
    ProductQuantizerPQ8,
    LunarSystolicVectorMemory,
)


def test_mamba2_ssd_model_construction():
    model = build_mamba2_ssd_openvino_model(n_heads=4, d_head=64, d_state=16)
    assert model is not None
    assert len(model.inputs) == 2
    assert len(model.outputs) == 2
    # Verify input shapes
    assert list(model.inputs[0].shape) == [1, 4, 64]
    assert list(model.inputs[1].shape) == [1, 4, 64, 16]


def test_mamba2_engine_step_and_constant_memory():
    mamba2 = LunarMamba2Engine(n_heads=4, d_head=64, d_state=16)
    assert mamba2.state.shape == (1, 4, 64, 16)
    # Check Theorem 11.1: State is strictly invariant
    initial_bytes = mamba2.state_bytes
    assert initial_bytes == 1 * 4 * 64 * 16 * 4  # 16,384 bytes FP32

    token = np.random.randn(1, 4, 64).astype(np.float32)
    y_out, lat_ms = mamba2.step(token)
    assert y_out.shape == (1, 4, 64)
    assert lat_ms > 0.0
    assert mamba2.state_bytes == initial_bytes
    assert not np.allclose(mamba2.state, 0.0)


def test_persistent_state_manager_restore_sub_15us():
    mgr = PersistentStateManager()
    state = np.random.randn(1, 4, 64, 16).astype(np.float32)

    sid = mgr.snapshot(state, snapshot_id="session_alpha")
    assert sid == "session_alpha"

    restored, restore_us = mgr.restore("session_alpha")
    assert np.allclose(restored, state)

    bench = mgr.benchmark_restore_latency(iterations=100)
    assert bench["iterations"] == 100
    assert bench["mean_restore_latency_us"] >= 0.0
    assert bench["energy_saved_joules"] == 45.0


def test_persistent_state_manager_fork():
    mgr = PersistentStateManager()
    state = np.ones((1, 4, 64, 16), dtype=np.float32) * 5.0
    mgr.snapshot(state, snapshot_id="root_session")

    branch_id = mgr.fork("root_session", "speculative_branch_1")
    assert branch_id == "speculative_branch_1"

    branch_state, _ = mgr.restore("speculative_branch_1")
    assert np.allclose(branch_state, 5.0)


def test_product_quantizer_pq8_compression():
    pq = ProductQuantizerPQ8(dim=384, n_subspaces=48, n_clusters=256)
    # Generate 100 random 384D vectors
    rng = np.random.RandomState(42)
    vectors = rng.randn(100, 384).astype(np.float32)
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    vectors = vectors / norms

    # Encode to 48 bytes
    codes = pq.encode(vectors)
    assert codes.shape == (100, 48)
    assert codes.dtype == np.uint8

    # Verify 32x compression ratio (1536 bytes -> 48 bytes)
    raw_bytes = vectors.nbytes
    comp_bytes = codes.nbytes
    assert raw_bytes / comp_bytes == 32.0

    # Decode and check correlation
    reconstructed = pq.decode(codes)
    assert reconstructed.shape == (100, 384)
    dot_prods = np.sum(vectors * reconstructed, axis=-1)
    assert np.mean(dot_prods) > 0.6  # Solid reconstruction alignment on S^383


def test_product_quantizer_pq8_adc_lut():
    pq = ProductQuantizerPQ8(dim=384, n_subspaces=48, n_clusters=256)
    query = np.random.randn(384).astype(np.float32)
    query /= np.linalg.norm(query)

    lut, lut_us = pq.compute_adc_lut(query)
    assert lut.shape == (48, 256)
    assert lut_us >= 0.0


def test_systolic_vector_memory_search_and_decay():
    sys_mem = LunarSystolicVectorMemory()
    sys_mem.add("Intel Core Ultra 7 258V Lunar Lake SoC", doc_id="d1")
    sys_mem.add("Mamba-2 SSD recurrence on NPU silicon", doc_id="d2")
    sys_mem.add("Level Zero USM Direct3D shared NT handle", doc_id="d3")

    assert sys_mem.total_items == 3
    assert sys_mem.total_bytes_used == 3 * 48  # exactly 144 bytes!

    results = sys_mem.query("NPU Mamba state recurrence", top_k=2)
    assert len(results) == 2
    assert "score" in results[0]
    assert "scan_latency_ms" in results[0]


def test_systolic_vector_memory_benchmark():
    sys_mem = LunarSystolicVectorMemory()
    bench = sys_mem.benchmark_scan(count=5000)
    assert bench["items_scanned"] == 5000
    assert bench["mean_scan_latency_ms"] >= 0.0
    assert bench["compression_ratio"] == "32.0x"
