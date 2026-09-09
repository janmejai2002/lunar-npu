"""
Antigravity PreToolUse Hook: Silicon Circuit Breaker
====================================================
Intercepts proposed `run_command` tool calls and validates safety via:
1. Sub-0.15ms deterministic DFA regex gate
2. Physical NPU silicon neural hazard classifier

Contract (Antigravity Lifecycle Hook):
Input on stdin:  JSON with toolCall { name, args }
Output on stdout: JSON with { decision: "allow" | "deny", reason: "..." }
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

from lunar_core.circuit_breaker import SiliconCircuitBreaker

AUDIT_LOG_PATH = Path(".lunar_circuit_audit.jsonl")


def run_circuit_breaker_hook(payload: Dict[str, Any], cb_instance: Optional[SiliconCircuitBreaker] = None) -> Dict[str, Any]:
    """
    Evaluates a tool call payload against the Silicon Circuit Breaker.
    Returns hook decision dict matching Antigravity PreToolUse contract.
    """
    tool_call = payload.get("toolCall", {})
    tool_name = tool_call.get("name", "")
    args = tool_call.get("args", {})

    # Only run_command warrants shell command circuit breaking
    if tool_name != "run_command":
        return {
            "decision": "allow",
            "reason": f"Tool '{tool_name}' bypassed (non-shell operation)",
        }

    cmd = args.get("CommandLine", "").strip()
    if not cmd:
        return {
            "decision": "allow",
            "reason": "Empty command line bypassed",
        }

    try:
        breaker = cb_instance or SiliconCircuitBreaker()
        audit_res = breaker.audit_command(cmd)

        # Log audit entry to persistent workspace log
        try:
            with open(AUDIT_LOG_PATH, "a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "timestamp": time.time(),
                    "conversationId": payload.get("conversationId", ""),
                    "stepIdx": payload.get("stepIdx", 0),
                    "command": cmd,
                    "verdict": audit_res.get("verdict"),
                    "tier": audit_res.get("tier"),
                    "hazard_probability": audit_res.get("hazard_probability"),
                    "latency_ms": audit_res.get("latency_ms"),
                    "reason": audit_res.get("reason"),
                }) + "\n")
        except Exception:
            pass

        if audit_res.get("verdict") == "BLOCKED":
            return {
                "decision": "deny",
                "reason": (
                    f"[BLOCKED] Silicon Circuit Breaker BLOCKED command (Tier: {audit_res.get('tier')}, "
                    f"Latency: {audit_res.get('latency_ms', 0):.2f}ms): {audit_res.get('reason')}"
                ),
            }

        return {
            "decision": "allow",
            "reason": (
                f"[PASSED] Silicon Circuit Breaker PASSED (Tier: {audit_res.get('tier')}, "
                f"Latency: {audit_res.get('latency_ms', 0):.2f}ms)"
            ),
        }

    except Exception as e:
        # Fail safe to avoid breaking agent workflow on transient hook errors
        return {
            "decision": "allow",
            "reason": f"Circuit breaker hook internal exception: {e}",
        }


def main():
    try:
        raw_in = sys.stdin.read()
        if not raw_in.strip():
            sys.stdout.write(json.dumps({"decision": "allow", "reason": "No input payload"}))
            return

        payload = json.loads(raw_in)
        result = run_circuit_breaker_hook(payload)
        sys.stdout.write(json.dumps(result))
    except Exception as e:
        sys.stdout.write(json.dumps({"decision": "allow", "reason": f"Fallback allow on stdin parse error: {e}"}))


if __name__ == "__main__":
    main()
