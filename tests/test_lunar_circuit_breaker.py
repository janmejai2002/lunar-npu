"""Unit and integration tests for SiliconCircuitBreaker."""

import pytest
from lunar_core.circuit_breaker import SiliconCircuitBreaker


def test_circuit_breaker_blocks_dangerous_commands():
    cb = SiliconCircuitBreaker()

    dangerous_cases = [
        "rm -rf /",
        "rm -rf ~",
        "Remove-Item -Recurse C:\\Windows\\System32",
        "DROP TABLE users;",
        "TRUNCATE TABLE accounts",
        ":(){ :|:& };:",
        "git push origin main --force",
        "dd if=/dev/zero of=/dev/sda",
    ]

    for cmd in dangerous_cases:
        audit = cb.audit_command(cmd)
        assert audit["verdict"] == "BLOCKED", f"Expected '{cmd}' to be BLOCKED, got {audit['verdict']}"
        assert audit["hazard_probability"] >= 0.15
        assert audit["latency_ms"] >= 0.0


def test_circuit_breaker_allows_safe_commands():
    cb = SiliconCircuitBreaker()

    safe_cases = [
        "git status",
        "python -m pytest tests/",
        "npm run build",
        "codemap status .",
        "python -m lunar_core status",
        "Get-ChildItem -Path .",
    ]

    for cmd in safe_cases:
        audit = cb.audit_command(cmd)
        assert audit["verdict"] == "ALLOWED", f"Expected '{cmd}' to be ALLOWED, got {audit['verdict']}"
        assert audit["hazard_probability"] < 0.15


def test_circuit_breaker_summary():
    cb = SiliconCircuitBreaker()
    cb.audit_command("git status")
    cb.audit_command("rm -rf /")

    summary = cb.get_audit_summary()
    assert summary["total_commands_audited"] == 2
    assert summary["allowed_count"] == 1
    assert summary["blocked_count"] == 1
    assert summary["mean_latency_ms"] >= 0.0
