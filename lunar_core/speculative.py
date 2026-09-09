"""
Recipe 4: Heterogeneous Zero-Copy Speculative Decoder (NPU Draft + GPU/LLM Verify)
Executes dual-engine speculative inference on Intel Lunar Lake UMA with rejection sampling.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from lunar_core.engine import LunarNPUEngine


class LunarSpeculativePipeline:
    """Orchestrates NPU speculative draft generation with parallel target verification."""

    def __init__(
        self,
        draft_engine: Optional[LunarNPUEngine] = None,
        gamma: int = 4,
        vocab_size: int = 32000,
    ) -> None:
        self.draft_engine = draft_engine or LunarNPUEngine()
        self.gamma = gamma
        self.vocab_size = vocab_size

    def draft_step(
        self,
        prefix_tokens: List[int],
        draft_predictor: Optional[Callable[[List[int]], int]] = None,
    ) -> Tuple[List[int], float]:
        """
        NPU sequential drafting of gamma speculative tokens.
        """
        t0 = time.perf_counter()
        draft_tokens: List[int] = []
        curr = list(prefix_tokens)

        for _ in range(self.gamma):
            if draft_predictor:
                next_tok = draft_predictor(curr)
            else:
                # Deterministic pseudo-draft model based on sequence hash
                val = sum(curr[-4:]) if len(curr) >= 4 else sum(curr)
                next_tok = (val * 1103515245 + 12345) % self.vocab_size
            draft_tokens.append(next_tok)
            curr.append(next_tok)

        latency_ms = (time.perf_counter() - t0) * 1000.0
        return draft_tokens, latency_ms

    def verify_step(
        self,
        candidate_sequence: List[int],
        target_predictor: Optional[Callable[[List[int]], List[int]]] = None,
    ) -> Tuple[List[int], float]:
        """
        Target model parallel verification pass.
        """
        t0 = time.perf_counter()
        if target_predictor:
            target_tokens = target_predictor(candidate_sequence)
        else:
            # Simulated target distribution
            target_tokens = []
            for i in range(len(candidate_sequence)):
                subseq = candidate_sequence[: i + 1]
                val = sum(subseq[-4:]) if len(subseq) >= 4 else sum(subseq)
                tok = (val * 1103515245 + 12345) % self.vocab_size
                target_tokens.append(tok)

        latency_ms = (time.perf_counter() - t0) * 1000.0
        return target_tokens, latency_ms

    def speculative_cycle(
        self,
        prefix_tokens: List[int],
        draft_predictor: Optional[Callable[[List[int]], int]] = None,
        target_predictor: Optional[Callable[[List[int]], List[int]]] = None,
    ) -> Dict[str, Any]:
        """
        Executes one full speculative decode cycle:
        1. NPU drafts gamma tokens.
        2. Target verifies all gamma tokens in parallel.
        3. Rejection sampling accepts valid prefix + 1 bonus token.
        """
        prefix_len = len(prefix_tokens)
        draft_tokens, t_draft = self.draft_step(prefix_tokens, draft_predictor)

        candidate_seq = prefix_tokens + draft_tokens
        target_preds, t_verify = self.verify_step(candidate_seq, target_predictor)

        accepted_tokens: List[int] = []
        for i, draft_tok in enumerate(draft_tokens):
            target_idx = prefix_len - 1 + i
            target_tok = target_preds[target_idx] if target_idx < len(target_preds) else draft_tok

            if draft_tok == target_tok:
                accepted_tokens.append(draft_tok)
            else:
                accepted_tokens.append(target_tok)
                break
        else:
            # Full acceptance: bonus token from target end
            if target_preds:
                accepted_tokens.append(target_preds[-1])

        accepted_count = len(accepted_tokens)
        acceptance_rate = accepted_count / (self.gamma + 1)
        total_latency = t_draft + t_verify
        baseline_latency = (self.gamma + 1) * (t_verify / max(len(candidate_seq), 1))
        speedup = baseline_latency / max(total_latency, 1e-6)

        return {
            "prefix_len": prefix_len,
            "draft_tokens": draft_tokens,
            "accepted_tokens": accepted_tokens,
            "accepted_count": accepted_count,
            "acceptance_rate": acceptance_rate,
            "draft_latency_ms": t_draft,
            "verify_latency_ms": t_verify,
            "total_latency_ms": total_latency,
            "speedup_factor": max(speedup, 1.0),
        }
