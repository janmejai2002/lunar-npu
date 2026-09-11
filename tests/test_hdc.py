"""Unit and integration tests for 10,000-bit Hyperdimensional Computing (HDC) Engine."""

import json
import numpy as np
import pytest
from click.testing import CliRunner

from lunar_core.cli import cli
from lunar_core.hdc import BinaryHypervector, HDCQueryResult, HyperdimensionalMemoryEngine


def test_binary_hypervector_creation():
    vec = BinaryHypervector.random(dim=10000, seed=42)
    assert vec.dim == 10000
    assert vec.n_bytes == 1250
    assert len(vec.packed) == 1250
    assert isinstance(vec._int_val, int)


def test_hdc_deterministic_binding_unbinding():
    a = BinaryHypervector.from_text("concept_alpha", dim=10000)
    b = BinaryHypervector.from_text("concept_beta", dim=10000)
    bound = a.bind(b)

    # Reversible unbinding test
    recovered_a = bound.unbind(b)
    assert np.array_equal(a.packed, recovered_a.packed)
    assert a.similarity(recovered_a) == 1.0
    assert a.hamming_distance(recovered_a) == 0


def test_hdc_quasi_orthogonality():
    a = BinaryHypervector.from_text("token_encoder", dim=10000)
    b = BinaryHypervector.from_text("decoder_head", dim=10000)
    sim = a.similarity(b)
    # Expected quasi-orthogonal similarity around 0.50 (within 0.04)
    assert 0.46 <= sim <= 0.54


def test_hdc_permutation_reversibility():
    a = BinaryHypervector.from_text("sequence_root", dim=10000)
    perm = a.permute(7)
    inv_perm = perm.permute(-7)
    assert np.array_equal(a.packed, inv_perm.packed)
    assert a.similarity(inv_perm) == 1.0
    # Permutation makes vector quasi-orthogonal to itself
    assert 0.46 <= a.similarity(perm) <= 0.54


def test_hdc_majority_bundling():
    a = BinaryHypervector.from_text("intel_lunar_lake", dim=10000)
    b = BinaryHypervector.from_text("ai_boost_npu", dim=10000)
    c = BinaryHypervector.from_text("arc_gpu_xe2", dim=10000)

    bundled = BinaryHypervector.bundle([a, b, c])
    assert bundled.dim == 10000

    # Bundled vector must retain high similarity (>0.70) to all constituent vectors
    assert bundled.similarity(a) > 0.70
    assert bundled.similarity(b) > 0.70
    assert bundled.similarity(c) > 0.70


def test_hdc_memory_engine_add_and_query():
    engine = HyperdimensionalMemoryEngine(dim=10000)
    engine.add("doc_mamba", "Mamba-2 state space model selective recurrence")
    engine.add("doc_lora", "OpenVINO Micro-LoRA adjoint backward pass compilation")
    engine.add("doc_vision", "DirectX DXGI desktop duplication screen perception")

    assert len(engine) == 3

    results = engine.query("Mamba-2 state space model selective recurrence", top_k=1)
    assert len(results) == 1
    assert results[0].key == "doc_mamba"
    assert results[0].similarity == 1.0
    assert results[0].hamming_distance == 0


def test_hdc_associative_recall_latency():
    engine = HyperdimensionalMemoryEngine(dim=10000)
    # Populate 100 active session vectors
    for i in range(100):
        engine.add(f"ephemeral_item_{i}", f"session token data payload {i}")

    # Query recall (sample best of 3 to eliminate OS thread context switch jitter)
    results = engine.query("session token data payload 42", top_k=5)
    assert len(results) == 5
    assert results[0].key == "ephemeral_item_42"
    min_lat = min(engine.query("session token data payload 42", top_k=5)[0].latency_us for _ in range(3))
    # Recall latency across 100 vectors must be < 150µs (typically ~50µs)
    assert min_lat < 150.0


def test_hdc_cli_store_and_query():
    runner = CliRunner()
    res_store = runner.invoke(cli, ["hdc", "store", "agent_goal", "Make Everything Real", "--json"])
    assert res_store.exit_code == 0
    data_store = json.loads(res_store.output)
    assert data_store["status"] == "stored"
    assert data_store["key"] == "agent_goal"

    res_query = runner.invoke(cli, ["hdc", "query", "Make Everything Real", "--top-k", "1", "--json"])
    assert res_query.exit_code == 0
    data_query = json.loads(res_query.output)
    assert len(data_query) >= 1
    assert data_query[0]["key"] == "agent_goal"
    assert data_query[0]["similarity"] == 1.0
