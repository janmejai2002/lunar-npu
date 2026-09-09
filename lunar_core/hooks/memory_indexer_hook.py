"""
Antigravity PostToolUse Hook: Local NPU Vector Memory Indexer
=============================================================
Intercepts tool completions and indexes actions/code decisions into persistent
hyperspherical vector memory (S^383) accelerated on Intel Lunar Lake NPU silicon.

Contract (Antigravity Lifecycle Hook):
Input on stdin:  JSON with toolCall, stepIdx, conversationId, error (if any)
Output on stdout: Empty JSON object `{}`
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

from lunar_core.vector_memory import LunarVectorMemory

MEMORY_FILE_PATH = Path(".lunar_workspace_memory.json")
_VMEM_INSTANCE: Optional[LunarVectorMemory] = None


def get_vmem_instance() -> LunarVectorMemory:
    global _VMEM_INSTANCE
    if _VMEM_INSTANCE is None:
        _VMEM_INSTANCE = LunarVectorMemory()
        if MEMORY_FILE_PATH.exists():
            try:
                _VMEM_INSTANCE.load_from_disk(MEMORY_FILE_PATH)
            except Exception:
                pass
    return _VMEM_INSTANCE


def run_memory_indexer_hook(payload: Dict[str, Any], vmem_instance: Optional[LunarVectorMemory] = None) -> Dict[str, Any]:
    """
    Indexes tool executions and code decisions into local NPU vector memory.
    Returns empty dict `{}` matching Antigravity PostToolUse contract.
    """
    tool_call = payload.get("toolCall", {})
    tool_name = tool_call.get("name", "")
    args = tool_call.get("args", {})
    step_idx = payload.get("stepIdx", 0)
    conv_id = payload.get("conversationId", "")
    error_msg = payload.get("error", "")

    # Extract semantic summary of the action
    text_to_embed = ""
    metadata = {
        "tool": tool_name,
        "stepIdx": step_idx,
        "conversationId": conv_id,
        "timestamp": time.time(),
        "has_error": bool(error_msg),
    }

    if tool_name == "run_command":
        cmd = args.get("CommandLine", "").strip()
        if not cmd:
            return {}
        text_to_embed = f"Command execution: {cmd}"
        metadata["command"] = cmd

    elif tool_name == "write_to_file":
        target = args.get("TargetFile", "")
        desc = args.get("Description", "")
        text_to_embed = f"Write file: {target}. Description: {desc}"
        metadata["target_file"] = target

    elif tool_name == "replace_file_content":
        target = args.get("TargetFile", "")
        inst = args.get("Instruction", "")
        text_to_embed = f"Edit file: {target}. Instruction: {inst}"
        metadata["target_file"] = target

    elif tool_name in ("view_file", "list_dir", "grep_search", "find_by_name"):
        # Skip excessive read-only queries to prevent memory saturation
        return {}
    else:
        text_to_embed = f"Agent action tool: {tool_name}"

    if not text_to_embed:
        return {}

    try:
        mem = vmem_instance or get_vmem_instance()
        doc_id = f"step_{conv_id[:8]}_{step_idx}_{int(time.time() * 1000) % 10000}"
        entry = mem.add_document(text_to_embed, metadata=metadata, doc_id=doc_id)
        # Persist updated memory to workspace file
        mem.save_to_disk(MEMORY_FILE_PATH)
    except Exception:
        pass

    # Antigravity PostToolUse expects an empty JSON object `{}`
    return {}


def main():
    try:
        raw_in = sys.stdin.read()
        if not raw_in.strip():
            sys.stdout.write(json.dumps({}))
            return

        payload = json.loads(raw_in)
        res = run_memory_indexer_hook(payload)
        sys.stdout.write(json.dumps(res))
    except Exception:
        sys.stdout.write(json.dumps({}))


if __name__ == "__main__":
    main()
