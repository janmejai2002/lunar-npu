"""Unit tests for Antigravity lifecycle hooks in lunar_core.hooks."""

import json
from pathlib import Path
import pytest

from lunar_core.hooks.circuit_breaker_hook import run_circuit_breaker_hook
from lunar_core.hooks.memory_indexer_hook import run_memory_indexer_hook, MEMORY_FILE_PATH
from lunar_core.vector_memory import LunarVectorMemory


def test_circuit_breaker_hook_blocks_dangerous_command():
    payload = {
        "toolCall": {
            "name": "run_command",
            "args": {"CommandLine": "rm -rf /"},
        },
        "stepIdx": 1,
        "conversationId": "test-hooks-conv",
    }
    decision = run_circuit_breaker_hook(payload)
    assert decision["decision"] == "deny"
    assert "BLOCKED" in decision["reason"]


def test_circuit_breaker_hook_allows_safe_command():
    payload = {
        "toolCall": {
            "name": "run_command",
            "args": {"CommandLine": "git status"},
        },
        "stepIdx": 2,
        "conversationId": "test-hooks-conv",
    }
    decision = run_circuit_breaker_hook(payload)
    assert decision["decision"] == "allow"
    assert "PASSED" in decision["reason"]


def test_circuit_breaker_hook_ignores_non_shell():
    payload = {
        "toolCall": {
            "name": "view_file",
            "args": {"AbsolutePath": "foo.py"},
        },
        "stepIdx": 3,
    }
    decision = run_circuit_breaker_hook(payload)
    assert decision["decision"] == "allow"


def test_memory_indexer_hook_persists_action(tmp_path):
    test_mem_file = tmp_path / "test_memory.json"
    mem = LunarVectorMemory()

    payload = {
        "toolCall": {
            "name": "run_command",
            "args": {"CommandLine": "python -m pytest tests/"},
        },
        "stepIdx": 4,
        "conversationId": "test-hooks-conv",
    }

    # Monkeypatch memory file for isolated test
    import lunar_core.hooks.memory_indexer_hook as mih
    orig_path = mih.MEMORY_FILE_PATH
    try:
        mih.MEMORY_FILE_PATH = test_mem_file
        res = run_memory_indexer_hook(payload, vmem_instance=mem)
        assert res == {}  # Antigravity PostToolUse contract expects empty dict
        assert test_mem_file.exists()

        # Verify persisted document can be loaded back
        mem2 = LunarVectorMemory()
        loaded_count = mem2.load_from_disk(test_mem_file)
        assert loaded_count >= 1
        assert "pytest" in mem2.documents[0]["text"]
    finally:
        mih.MEMORY_FILE_PATH = orig_path
