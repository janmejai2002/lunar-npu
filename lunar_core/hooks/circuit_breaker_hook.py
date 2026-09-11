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

from lunar_core.circuit_breaker import SiliconCircuitBreaker, DualStageSiliconCircuitBreaker
from lunar_core.hooks.silicon_guard_pipe import send_guard_ipc_query, get_pipe_server

AUDIT_LOG_PATH = Path(".lunar_circuit_audit.jsonl")


def run_circuit_breaker_hook(payload: Dict[str, Any], cb_instance: Optional[SiliconCircuitBreaker] = None) -> Dict[str, Any]:
    """
    Evaluates a tool call payload against the Silicon Circuit Breaker via IPC or resident DFA.
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
        if cb_instance is not None:
            raw_audit = cb_instance.audit_command(cmd)
            is_blocked = raw_audit.get("verdict") == "BLOCKED"
            audit_res = {
                "decision": "DENY" if is_blocked else "ALLOW",
                "status_code": 403 if is_blocked else 200,
                "tier": raw_audit.get("tier"),
                "hazard_probability": raw_audit.get("hazard_probability"),
                "reason": raw_audit.get("reason"),
                "latency_ms": raw_audit.get("latency_ms", 0.0),
                "receipt": "LUNAR-BLOCKED-403" if is_blocked else "LUNAR-SEC-FASTPATH",
            }
        else:
            audit_res = send_guard_ipc_query(cmd)

        is_blocked = (
            audit_res.get("decision") == "DENY"
            or audit_res.get("verdict") == "BLOCKED"
            or audit_res.get("status_code") == 403
        )

        receipt = audit_res.get("receipt", "LUNAR-BLOCKED-403" if is_blocked else "LUNAR-SEC-PASSED")
        lat_ms = audit_res.get("latency_ms")
        if lat_ms is None:
            lat_ms = audit_res.get("latency_us", 1.0) / 1000.0

        # Log audit entry to persistent workspace log
        try:
            with open(AUDIT_LOG_PATH, "a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "timestamp": time.time(),
                    "conversationId": payload.get("conversationId", ""),
                    "stepIdx": payload.get("stepIdx", 0),
                    "command": cmd,
                    "verdict": "BLOCKED" if is_blocked else "ALLOWED",
                    "tier": audit_res.get("tier"),
                    "hazard_probability": audit_res.get("hazard_probability"),
                    "latency_ms": round(lat_ms, 3),
                    "receipt": receipt,
                    "reason": audit_res.get("reason"),
                }) + "\n")
        except Exception:
            pass

        if is_blocked:
            return {
                "decision": "deny",
                "reason": (
                    f"[BLOCKED] Silicon Circuit Breaker BLOCKED command (Tier: {audit_res.get('tier')}, "
                    f"Latency: {lat_ms:.2f}ms): {audit_res.get('reason')}"
                ),
            }

        return {
            "decision": "allow",
            "reason": (
                f"[PASSED] Silicon Circuit Breaker PASSED (Receipt: {receipt}, "
                f"Tier: {audit_res.get('tier')}, Latency: {lat_ms:.2f}ms)"
            ),
        }

    except Exception as e:
        # Fail safe to avoid breaking agent workflow on transient hook errors
        return {
            "decision": "allow",
            "reason": f"Circuit breaker hook internal exception: {e}",
        }


def evaluate_command(command_str: str) -> tuple[int, str]:
    """Convenience helper to evaluate a shell command directly."""
    payload = {
        "toolCall": {
            "name": "run_command",
            "args": {"CommandLine": command_str}
        }
    }
    result = run_circuit_breaker_hook(payload)
    status_code = result.get("status_code", 200)
    reason = result.get("reason", "")
    if result.get("decision") == "deny":
        status_code = 403
    return status_code, reason


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
