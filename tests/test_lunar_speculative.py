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
    assert "total_latency_ms" in res

    # The previous assertion here was `res["speedup_factor"] >= 1.0`, which
    # could not fail: the value was computed as max(speedup, 1.0). The clamp is
    # gone, so assert the internally consistent relationship instead.
    assert res["speedup_factor"] == pytest.approx(
        res["baseline_latency_ms"] / res["total_latency_ms"], rel=1e-2
    ), "speedup_factor is not consistent with the latencies it is derived from"

    # Honest expectation while the draft model is untrained random weights:
    # acceptance is near zero, so speculation costs more than it saves.
    assert res["draft_model_trained"] is False
    assert res["speedup_factor"] < 1.0, (
        "Speculation appears to be winning with an untrained draft model, which "
        "would be surprising. If a real draft model was wired in, update this test."
    )


def test_speculative_real_target_verifier():
    from lunar_core.speculative import RealTargetVerifier
    verifier = RealTargetVerifier()
    if verifier.is_real:
        assert verifier.compiled_model is not None
        preds, lat = verifier.verify([750, 3974, 6860])
        assert len(preds) == 3
        assert lat > 0.0
    else:
        assert verifier.is_real is False


def test_speculative_cycle_with_real_target():
    spec = LunarSpeculativePipeline(gamma=2, vocab_size=151936, enable_real_target=True)
    res = spec.speculative_cycle(prefix_tokens=[750, 3974, 6860])
    assert res["prefix_len"] == 3
    assert len(res["draft_tokens"]) == 2
    assert res["accepted_count"] >= 1
    assert "target_engine" in res
    assert "is_real_target" in res

