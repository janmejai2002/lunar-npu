"""
Pre-execution shell command guard.

Screens proposed agent shell commands against a deterministic blocklist before
they reach the OS.

REALITY STATUS (audited 2026-09-11):
  - Stage 1: Aho-Corasick substring blocklist over known-catastrophic commands.
  - Stage 2: Regex rules for spacing/flag-order variants Stage 1 cannot catch.
  - Stage 3: Lexical keyword heuristics.
  - Stage 4: An OpenVINO MLP that runs on the NPU. ITS WEIGHTS ARE UNTRAINED
    (np.random, seed 42). Measured output is the constant 0.0474 for every
    input tested, i.e. it has zero discriminative power. It is therefore
    ADVISORY ONLY and is excluded from the verdict. See HAZARD_MODEL_IS_TRAINED.

SCOPE AND LIMITS — read before relying on this:
  This is a deterministic blocklist, NOT a security boundary. It stops known
  literal spellings of catastrophic commands. It does not and cannot stop an
  adversary, obfuscation, base64/encoded payloads, indirection through a
  script file, or any destructive command not enumerated below. Treat it as a
  seatbelt against agent slips, never as a sandbox.
"""

from __future__ import annotations

import hashlib
import re
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import openvino as ov
import openvino.opset13 as ops

from lunar_core.engine import LunarNPUEngine


# Set to True only when `table`, W1 and W2 below are loaded from weights that
# were actually fit to a labelled corpus of shell commands. While this is False,
# the model's output MUST NOT influence any verdict.
HAZARD_MODEL_IS_TRAINED = False


def build_hazard_classifier_openvino(
    seq_len: int = 16,
    d_emb: int = 64,
    d_hid: int = 32,
    vocab_size: int = 8192,
) -> ov.Model:
    """
    Builds an UNTRAINED binary classifier graph as a pure OpenVINO model.
    Architecture: Token Gather -> Mean Pool -> Dense(d_hid) -> ReLU -> Dense(1) -> Sigmoid

    WARNING: W1 and W2 are np.random values that were never fit to data, and the
    output bias is -3.0. Measured behaviour is a near-constant 0.0474 regardless
    of input. This graph exercises the NPU compile/infer path; it does not
    classify anything. Do not gate safety decisions on its output until
    HAZARD_MODEL_IS_TRAINED is True.
    """
    input_ids = ops.parameter([1, seq_len], ov.Type.i64, name="input_ids")

    rng = np.random.RandomState(42)
    table = (rng.randn(vocab_size, d_emb) * 0.05).astype(np.float32)

    # Initialize calibrated weights: hazardous shell tokens produce positive hazard bias
    hazard_words = [
        "rm", "rf", "drop", "truncate", "system32", "windows", "format",
        "mkfs", "eval", "exec", "delete", "destroy", "kill", "wipe", "dd",
        "shutil", "rmtree", "invoke-expression", "iex", "force"
    ]
    for hw in hazard_words:
        h_idx = int(hashlib.md5(hw.encode("utf-8")).hexdigest()[:8], 16) % (vocab_size - 1) + 1
        table[h_idx] += 0.8

    safe_words = [
        "git", "status", "diff", "log", "pytest", "test", "npm", "cargo",
        "python", "echo", "cat", "ls", "dir", "build", "get-childitem",
        "codemap", "lunar"
    ]
    for sw in safe_words:
        s_idx = int(hashlib.md5(sw.encode("utf-8")).hexdigest()[:8], 16) % (vocab_size - 1) + 1
        table[s_idx] -= 0.6

    embed_const = ops.constant(table)
    gathered = ops.gather(embed_const, input_ids, ops.constant(0, dtype=np.int64))
    pooled = ops.reduce_mean(gathered, ops.constant(1, dtype=np.int64), keep_dims=False)

    # Dense layer 1: d_emb -> d_hid
    W1 = (rng.randn(d_emb, d_hid) * 0.02).astype(np.float32)
    b1 = np.zeros(d_hid, dtype=np.float32)
    h1 = ops.add(ops.matmul(pooled, ops.constant(W1), False, False), ops.constant(b1))
    h1_relu = ops.relu(h1)

    # Dense layer 2: d_hid -> 1
    W2 = (rng.randn(d_hid, 1) * 0.02).astype(np.float32)
    b2 = np.array([-3.0], dtype=np.float32)  # conservative baseline toward safe (< 0.05 prob)
    logit = ops.add(ops.matmul(h1_relu, ops.constant(W2), False, False), ops.constant(b2))

    prob = ops.sigmoid(logit)

    model = ov.Model([prob.output(0)], [input_ids], "SiliconHazardClassifier")
    return model


class SiliconCircuitBreaker:
    """Deterministic pre-execution command blocklist. Not a security boundary."""

    # Regex tier. Catches spacing and flag-order variants that the literal
    # substring blocklist misses (e.g. "rm  -rf  /", "rm -fr /").
    DANGEROUS_PATTERNS = [
        # Recursive delete of a root-ish path, any flag order/spacing.
        re.compile(r"\brm\b[^|;&\n]*\s-[a-z]*r[a-z]*f|\brm\b[^|;&\n]*\s-[a-z]*f[a-z]*r", re.IGNORECASE),
        re.compile(r"\brm\b[^|;&\n]*--recursive[^|;&\n]*--force", re.IGNORECASE),
        re.compile(r"--no-preserve-root", re.IGNORECASE),
        # PowerShell recursive delete of a system or home path.
        re.compile(r"Remove-Item[^|;&\n]*-Recurse", re.IGNORECASE),
        re.compile(r"\brd\s+/s\b|\brmdir\s+/s\b", re.IGNORECASE),
        re.compile(r"\bdel\b[^|;&\n]*/s\b", re.IGNORECASE),
        # Destructive SQL.
        re.compile(r"DROP\s+(?:DATABASE|TABLE|SCHEMA)", re.IGNORECASE),
        re.compile(r"TRUNCATE\s+TABLE", re.IGNORECASE),
        re.compile(r"DELETE\s+FROM\s+\w+\s*(?:;|$)", re.IGNORECASE),  # unqualified DELETE
        # Fork bomb.
        re.compile(r":\(\)\s*\{\s*:\s*\|\s*:\s*&\s*\}\s*;", re.IGNORECASE),
        # History-destroying git.
        re.compile(r"git\s+push[^|;&\n]*(?:--force\b|-f\b)", re.IGNORECASE),
        re.compile(r"git\s+reset\s+--hard", re.IGNORECASE),
        re.compile(r"git\s+clean\s+-[a-z]*[fd]", re.IGNORECASE),
        # Raw block device writes / filesystem creation.
        re.compile(r"\bdd\b[^|;&\n]*of=/dev/(?:sd[a-z]|nvme|disk)", re.IGNORECASE),
        re.compile(r"mkfs(?:\.[a-z0-9]+)?\s+/dev/", re.IGNORECASE),
        re.compile(r"\bformat\s+[c-z]:", re.IGNORECASE),
        re.compile(r"diskpart\b[^|;&\n]*clean", re.IGNORECASE),
        # Pipe-from-network straight into an interpreter.
        re.compile(r"(?:curl|wget|iwr|Invoke-WebRequest)[^|;&\n]*\|\s*(?:ba|z|k|)sh\b", re.IGNORECASE),
        re.compile(r"(?:curl|wget)[^|;&\n]*\|\s*(?:sudo\s+)?(?:python|perl|ruby|node)\b", re.IGNORECASE),
        re.compile(r"Invoke-Expression[^|;&\n]*(?:DownloadString|Invoke-WebRequest|New-Object\s+Net)", re.IGNORECASE),
        # Reverse shells.
        re.compile(r"/dev/tcp/\d", re.IGNORECASE),
        re.compile(r"\bnc\b[^|;&\n]*-e\s*/bin/(?:ba)?sh", re.IGNORECASE),
        # Python one-liners that delete trees.
        re.compile(r"rmtree\s*\(", re.IGNORECASE),
        re.compile(r"os\.remove|os\.unlink|os\.rmdir", re.IGNORECASE),
        # Credential exfiltration paths.
        re.compile(r"\.ssh/id_(?:rsa|ed25519)|\.aws/credentials|\.env\b[^|;&\n]*\|", re.IGNORECASE),
        # Privilege / defence changes.
        re.compile(r"Set-MpPreference[^|;&\n]*Disable", re.IGNORECASE),
        re.compile(r"\bchmod\s+(?:-R\s+)?777\s+/", re.IGNORECASE),
        re.compile(r"\bicacls\b[^|;&\n]*/grant[^|;&\n]*Everyone", re.IGNORECASE),
    ]

    SUSPICIOUS_KEYWORDS = [
        "eval(", "exec(", "base64.b64decode", "Invoke-Expression", "iex ",
        "FromBase64String", "-EncodedCommand", "-enc ", "bypass -nop",
        "certutil -urlcache", "bitsadmin /transfer",
    ]

    def __init__(
        self,
        engine: Optional[LunarNPUEngine] = None,
        hazard_threshold: float = 0.15,
        seq_len: int = 16,
    ) -> None:
        self.engine = engine or LunarNPUEngine()
        self.hazard_threshold = hazard_threshold
        self.seq_len = seq_len
        self.audit_log: List[Dict[str, Any]] = []

        # Compile real neural hazard classifier on NPU silicon
        classifier_model = build_hazard_classifier_openvino(seq_len=self.seq_len)
        self.classifier_compiled = self.engine.compile_model(classifier_model)
        self.classifier_req = self.classifier_compiled.create_infer_request()

    def _tokenize(self, command_str: str) -> np.ndarray:
        """Tokenize shell command into fixed-length token IDs via MD5 hash."""
        words = [w.strip(".,!?:;\"'()[]{}|&<>$;") for w in command_str.lower().split()]
        words = [w for w in words if w]
        ids = [int(hashlib.md5(w.encode("utf-8")).hexdigest()[:8], 16) % 8191 + 1 for w in words[:self.seq_len]]
        if len(ids) < self.seq_len:
            ids = ids + [0] * (self.seq_len - len(ids))
        return np.array([ids], dtype=np.int64)

    def audit(self, command_str: str) -> Dict[str, Any]:
        """Convenience alias for audit_command."""
        return self.audit_command(command_str)

    def audit_command(self, command_str: str) -> Dict[str, Any]:
        """
        Audit a command against the regex tier and keyword heuristics.
        The NPU model runs but does not affect the verdict while untrained.
        """
        t0 = time.perf_counter()
        clean_cmd = command_str.strip()

        # Step 1: Regex tier
        for pattern in self.DANGEROUS_PATTERNS:
            if pattern.search(clean_cmd):
                lat = (time.perf_counter() - t0) * 1000.0
                result = {
                    "verdict": "BLOCKED",
                    "hazard_probability": 1.0,
                    "reason": f"Violated deterministic safety rule: {pattern.pattern}",
                    "latency_ms": lat,
                    "command": clean_cmd,
                    "timestamp": time.time(),
                    "tier": "DFA_REGEX_GATE",
                }
                self.audit_log.append(result)
                return result

        # Step 2: Keyword heuristics
        heuristic_score = 0.0
        matched_heuristics = []
        for kw in self.SUSPICIOUS_KEYWORDS:
            if kw.lower() in clean_cmd.lower():
                heuristic_score += 0.35
                matched_heuristics.append(kw)

        # Step 3: NPU model inference — ADVISORY ONLY while untrained.
        raw_prob = self._neural_advisory(clean_cmd)

        # Verdict is driven by deterministic signals only. The untrained model
        # contributes nothing; folding its constant output in would only ever
        # raise the floor on every command equally.
        hazard_prob = float(min(heuristic_score, 1.0))
        if HAZARD_MODEL_IS_TRAINED and raw_prob is not None:
            hazard_prob = float(min(max(hazard_prob, raw_prob), 1.0))

        verdict = "BLOCKED" if hazard_prob >= self.hazard_threshold else "ALLOWED"
        reason = (
            f"Suspicious execution markers: {matched_heuristics}" if matched_heuristics
            else "No deterministic rule matched (NOT a safety guarantee)"
        )

        lat = (time.perf_counter() - t0) * 1000.0
        result = {
            "verdict": verdict,
            "hazard_probability": round(hazard_prob, 4),
            "reason": reason,
            "latency_ms": lat,
            "command": clean_cmd,
            "timestamp": time.time(),
            "tier": "HEURISTIC_KEYWORD" if matched_heuristics else "NO_RULE_MATCHED",
            "device": self.engine.device,
            "neural_advisory": raw_prob,
            "neural_model_trained": HAZARD_MODEL_IS_TRAINED,
        }
        self.audit_log.append(result)
        return result

    def _neural_advisory(self, clean_cmd: str) -> Optional[float]:
        """
        Run the NPU model and return its scalar output, or None on failure.
        The value is recorded for observability only; see HAZARD_MODEL_IS_TRAINED.
        """
        try:
            input_ids = self._tokenize(clean_cmd)
            self.classifier_req.set_tensor(
                self.classifier_compiled.inputs[0],
                ov.Tensor(input_ids),
            )
            self.classifier_req.infer()
            return float(self.classifier_req.get_output_tensor(0).data.flatten()[0])
        except Exception:
            return None

    def get_audit_summary(self) -> Dict[str, Any]:
        """Summarize audit history."""
        total = len(self.audit_log)
        blocked = sum(1 for a in self.audit_log if a["verdict"] == "BLOCKED")
        allowed = total - blocked
        mean_lat = np.mean([a["latency_ms"] for a in self.audit_log]) if total > 0 else 0.0
        return {
            "total_commands_audited": total,
            "allowed_count": allowed,
            "blocked_count": blocked,
            "mean_latency_ms": float(mean_lat),
        }


# ============================================================================
# PILLAR 4: DUAL-STAGE SILICON CIRCUIT BREAKER (AHO-CORASICK DFA + NPU NEURAL)
# ============================================================================

from collections import deque


class AhoCorasickNode:
    """Trie node with Aho-Corasick failure transitions."""
    def __init__(self) -> None:
        self.children: Dict[str, AhoCorasickNode] = {}
        self.fail: Optional[AhoCorasickNode] = None
        self.output: List[str] = []


class AhoCorasickDFA:
    """
    Stage 1 literal-substring blocklist using Aho-Corasick matching.

    Single O(len(text)) pass over the command looking for any enumerated
    catastrophic spelling. It cannot have false POSITIVES beyond the literal
    strings below, and it has a very high false-NEGATIVE rate by construction:
    anything not spelled exactly like an entry below passes. The regex tier in
    SiliconCircuitBreaker.DANGEROUS_PATTERNS exists to cover the common
    spacing/flag-order variants; neither tier is exhaustive.
    """

    DEFAULT_CATASTROPHIC_PATTERNS = [
        "rm -rf /",
        "rm -rf ~",
        "rm -rf *",
        "remove-item -recurse c:\\windows",
        "remove-item -recurse c:\\windows\\system32",
        "drop database",
        "drop table",
        "truncate table",
        ":(){ :|:& };:",
        "git push origin main --force",
        "git push --force",
        "git push -f",
        "dd if=/dev/zero of=/dev/sda",
        "dd if=/dev/zero of=/dev/nvme",
        "dd if= of=/dev/sd",
        "mkfs.ext4",
        "mkfs.",
        "format c:",
        "format c: /fs:",
        "invoke-expression (new-object net.webclient)",
        "shutil.rmtree('/')",
        "bash -i >& /dev/tcp/",
    ]

    def __init__(self, patterns: Optional[List[str]] = None) -> None:
        self.patterns = patterns or self.DEFAULT_CATASTROPHIC_PATTERNS
        self.root = AhoCorasickNode()
        self._build_trie()
        self._build_failure_links()

    def _build_trie(self) -> None:
        """Insert all catastrophic patterns into the trie."""
        for pattern in self.patterns:
            node = self.root
            clean_pat = pattern.lower().strip()
            for char in clean_pat:
                if char not in node.children:
                    node.children[char] = AhoCorasickNode()
                node = node.children[char]
            node.output.append(clean_pat)

    def _build_failure_links(self) -> None:
        """Construct Aho-Corasick failure links via BFS queue."""
        queue: deque[AhoCorasickNode] = deque()
        for char, child in self.root.children.items():
            child.fail = self.root
            queue.append(child)

        while queue:
            current = queue.popleft()
            for char, child in current.children.items():
                fallback = current.fail
                while fallback is not None and char not in fallback.children:
                    fallback = fallback.fail
                child.fail = fallback.children[char] if fallback is not None else self.root
                if child.fail.output:
                    child.output.extend(child.fail.output)
                queue.append(child)

    def scan(self, text: str) -> List[str]:
        """
        Single-pass O(len(text)) scan.
        Returns list of matched catastrophic pattern strings.
        """
        matched: List[str] = []
        node = self.root
        lower_text = text.lower()

        for char in lower_text:
            while node is not None and char not in node.children:
                node = node.fail
            if node is None:
                node = self.root
                continue
            node = node.children[char]
            if node.output:
                matched.extend(node.output)

        return matched


class DualStageSiliconCircuitBreaker(SiliconCircuitBreaker):
    """
    Pre-execution command guard combining a literal blocklist with regex rules.

    Stage 1: Aho-Corasick literal blocklist.
    Stage 2: Regex rules for spelling variants.
    Stage 3: Keyword heuristics.
    Stage 4: NPU model inference, ADVISORY ONLY (untrained; see module docstring).

    Deterministic blocklist, not a security boundary. See module docstring.
    """

    def __init__(
        self,
        engine: Optional[LunarNPUEngine] = None,
        hazard_threshold: float = 0.15,
        seq_len: int = 16,
    ) -> None:
        super().__init__(engine=engine, hazard_threshold=hazard_threshold, seq_len=seq_len)
        self.dfa = AhoCorasickDFA()

    def audit_command(self, command_str: str) -> Dict[str, Any]:
        """
        Audit a command through the blocklist, regex and heuristic tiers.
        The NPU model runs but does not affect the verdict while untrained.
        """
        t0 = time.perf_counter()
        clean_cmd = command_str.strip()

        # STAGE 1: Aho-Corasick literal blocklist
        matches = self.dfa.scan(clean_cmd)
        if matches:
            lat_ms = (time.perf_counter() - t0) * 1000.0
            lat_us = lat_ms * 1000.0
            result = {
                "verdict": "BLOCKED",
                "hazard_probability": 1.0,
                "reason": f"Catastrophic DFA violation: matched pattern '{matches[0]}'",
                "latency_ms": round(lat_ms, 4),
                "latency_us": round(lat_us, 2),
                "command": clean_cmd,
                "timestamp": time.time(),
                "tier": "DFA_AHOCORASICK_GATE",
                "matched_pattern": matches[0],
            }
            self.audit_log.append(result)
            return result

        # Also evaluate dangerous regex patterns for edge cases
        for pattern in self.DANGEROUS_PATTERNS:
            if pattern.search(clean_cmd):
                lat_ms = (time.perf_counter() - t0) * 1000.0
                result = {
                    "verdict": "BLOCKED",
                    "hazard_probability": 1.0,
                    "reason": f"Violated deterministic safety rule: {pattern.pattern}",
                    "latency_ms": round(lat_ms, 4),
                    "command": clean_cmd,
                    "timestamp": time.time(),
                    "tier": "DFA_REGEX_GATE",
                }
                self.audit_log.append(result)
                return result

        # STAGE 3: Keyword heuristics, then STAGE 4 advisory model
        heuristic_score = 0.0
        matched_heuristics = []
        for kw in self.SUSPICIOUS_KEYWORDS:
            if kw.lower() in clean_cmd.lower():
                heuristic_score += 0.35
                matched_heuristics.append(kw)

        raw_prob = self._neural_advisory(clean_cmd)

        hazard_prob = float(min(heuristic_score, 1.0))
        if HAZARD_MODEL_IS_TRAINED and raw_prob is not None:
            hazard_prob = float(min(max(hazard_prob, raw_prob), 1.0))

        verdict = "BLOCKED" if hazard_prob >= self.hazard_threshold else "ALLOWED"
        reason = (
            f"Suspicious execution markers: {matched_heuristics}" if matched_heuristics
            else "No deterministic rule matched (NOT a safety guarantee)"
        )

        lat_ms = (time.perf_counter() - t0) * 1000.0
        result = {
            "verdict": verdict,
            "hazard_probability": round(hazard_prob, 4),
            "reason": reason,
            "latency_ms": round(lat_ms, 3),
            "command": clean_cmd,
            "timestamp": time.time(),
            "tier": "HEURISTIC_KEYWORD" if matched_heuristics else "NO_RULE_MATCHED",
            "device": self.engine.device,
            "neural_advisory": raw_prob,
            "neural_model_trained": HAZARD_MODEL_IS_TRAINED,
        }
        self.audit_log.append(result)
        return result

