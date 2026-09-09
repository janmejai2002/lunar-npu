"""Unit and integration tests for LunarVectorMemory."""

import numpy as np
import pytest
from lunar_core.vector_memory import LunarVectorMemory


def test_vector_memory_embed():
    vmem = LunarVectorMemory(seq_len=32, embedding_dim=384)
    text = "Intel Lunar Lake NPU ambient vector intelligence"
    vec, lat = vmem.embed(text)

    assert isinstance(vec, np.ndarray)
    assert vec.ndim == 1
    assert vec.shape[0] in [384, 768]
    # Verify L2 normalization onto unit hypersphere
    norm = np.linalg.norm(vec)
    assert pytest.approx(norm, rel=1e-3) == 1.0
    assert lat >= 0.0


def test_vector_memory_search():
    vmem = LunarVectorMemory(seq_len=32, embedding_dim=384)
    vmem.clear()
    vmem.add_document("Quantum computing and superconducting qubits", metadata={"topic": "physics"}, doc_id="d1")
    vmem.add_document("Intel Lunar Lake neural processing unit 47 TOPS", metadata={"topic": "hardware"}, doc_id="d2")
    vmem.add_document("Mamba state-space recurrence step without KV-cache", metadata={"topic": "ai"}, doc_id="d3")

    results = vmem.query("Tell me about NPU hardware architecture", top_k=2)
    assert len(results) == 2
    assert results[0]["id"] in ["d1", "d2", "d3"]
    assert "score" in results[0]
    assert "query_latency_ms" in results[0]


def test_vector_memory_clear():
    vmem = LunarVectorMemory()
    vmem.add_document("Test sample")
    assert len(vmem.documents) > 0
    vmem.clear()
    assert len(vmem.documents) == 0
