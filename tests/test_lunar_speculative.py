"""Unit and integration tests for LunarSpeculativePipeline."""

import pytest
from lunar_core.speculative import LunarSpeculativePipeline


def test_speculative_draft_step():
    spec = LunarSpeculativePipeline(gamma=4, vocab_size=1000)
    prefix = [10, 20, 30]
    draft_tokens, lat = spec.draft_step(prefix)

    assert len(draft_tokens) == 4
    assert lat >= 0.0
    for tok in draft_tokens:
        assert isinstance(tok, (int, float))


def test_speculative_cycle_full():
    spec = LunarSpeculativePipeline(gamma=3, vocab_size=5000)
    prefix = [1, 2, 3]

    res = spec.speculative_cycle(prefix_tokens=prefix)
    assert res["prefix_len"] == 3
    assert len(res["draft_tokens"]) == 3
    assert res["accepted_count"] >= 1
    assert 0.0 <= res["acceptance_rate"] <= 1.0
    assert res["speedup_factor"] >= 1.0
    assert "total_latency_ms" in res
