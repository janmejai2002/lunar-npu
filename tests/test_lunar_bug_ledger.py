"""Unit tests for BugLedgerEngine."""

import pytest
from lunar_core.bug_ledger import BugLedgerEngine


def test_bug_ledger_lifecycle(tmp_path):
    ledger_file = tmp_path / "TEST_BUG_LEDGER.md"
    ledger = BugLedgerEngine(ledger_path=ledger_file)

    # Initial empty read
    entries = ledger.read_entries()
    assert len(entries) == 0

    # Log a new bug
    entry = ledger.log_bug(
        bug_id="BUG-101",
        severity="High",
        subsystem="NPU Compiler",
        summary="Dynamic dimension shapes fail systolic compilation",
        root_cause="OpenVINO NPU plugin requires static tensor bounds",
        specialist="Speculative NPU Architect",
        test="test_vector_memory_embed",
        status="OPEN"
    )
    assert entry["id"] == "BUG-101"
    assert entry["status"] == "OPEN"

    # Verify written to disk
    re_read = ledger.read_entries()
    assert len(re_read) == 1
    assert re_read[0]["id"] == "BUG-101"

    # Resolve bug
    resolved = ledger.resolve_bug("BUG-101", verification_result="exit_code=0")
    assert resolved is True

    # Verify updated status
    re_read2 = ledger.read_entries()
    assert re_read2[0]["status"] == "VERIFIED_FIX"
    assert "exit_code=0" in re_read2[0]["verification_test"]
