"""
tests/test_live_transmission.py
================================
Operational Transmission Layer End-to-End Integration Tests (v2.2.0).
Verifies:
1. Real AST code ingestion into S^383 vector memory and sub-4ms PQ8 search.
2. Windows Named-Pipe IPC (\\\\.\\pipe\\lunar_silicon_guard) safety checks and cryptographic receipts.
3. PreToolUse circuit breaker hook physically rejecting destructive commands with code 403.
4. Command Deck (Studio) API endpoints: /api/query, /api/tokens/stats, /api/memory/inspect, and /api/stream.
"""

import json
import threading
import time
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path
import pytest
import numpy as np

from lunar_core.indexer import CodeRepoIndexer, get_repo_indexer, CodeChunk
from lunar_core.vector_memory import LunarVectorMemory, ProductQuantizerPQ8
from lunar_core.hooks.silicon_guard_pipe import (
    SiliconGuardPipeServer,
    send_guard_ipc_query,
    get_pipe_server,
)
from lunar_core.hooks.circuit_breaker_hook import evaluate_command
from lunar_core.studio import LunarStudioHandler


class ReusableThreadingServer(ThreadingHTTPServer):
    allow_reuse_address = True
    daemon_threads = True

    def handle_error(self, request, client_address):
        # Gracefully ignore client disconnects during SSE teardown
        pass


# =========================================================================
# 1. Real Code AST Ingestion & S^383 Vector Memory Tests
# =========================================================================

def test_real_ast_code_indexer_crawls_and_extracts():
    """Verify CodeRepoIndexer extracts real AST symbols from the jolly-meitner codebase."""
    indexer = CodeRepoIndexer()
    this_repo = Path(__file__).parent.parent

    # Target specific core files for fast, deterministic extraction
    engine_file = this_repo / "lunar_core" / "engine.py"
    assert engine_file.exists(), f"Engine file must exist at {engine_file}"

    chunks = indexer.extract_file_chunks(engine_file, "jolly-meitner")
    assert len(chunks) > 0, "Must extract at least one AST symbol from engine.py"

    symbols = {c.symbol_name for c in chunks}
    # Should find real classes/functions in engine.py
    assert any("DualProfileGovernor" in s or "LunarNPUEngine" in s or "USMSharedRingBuffer" in s for s in symbols)

    # Verify chunk structure contains real code tokens
    for c in chunks:
        assert c.repo == "jolly-meitner"
        assert c.file_path.endswith("engine.py")
        assert c.start_line > 0
        assert len(c.code_snippet) > 0
        assert len(c.semantic_text) > 0


def test_real_code_indexing_and_pq8_quantization():
    """Verify S^383 projection, 32x PQ8 quantization (48 bytes), and code recall."""
    this_repo = Path(__file__).parent.parent
    indexer = CodeRepoIndexer()

    # Ingest real files from jolly-meitner
    stats = indexer.index_repositories(
        repo_paths=[this_repo],
        max_chunks=25,
        persist=False,
    )

    assert stats["status"] == "INDEXED"
    assert stats["chunks_indexed"] > 0
    assert stats["pq8_compressed_kb"] > 0
    assert stats["compression_ratio"] == "32x (48 bytes/symbol)"
    assert len(indexer.codes) == stats["chunks_indexed"]
    assert indexer.codes.shape[1] == 48  # 48 uint8 bytes per 384-D vector

    # Perform real code query
    t0 = time.perf_counter()
    results = indexer.query_code("DualProfileGovernor NPU power", top_k=3)
    latency_ms = (time.perf_counter() - t0) * 1000.0

    assert len(results) > 0
    assert latency_ms < 150.0  # Cold query including table init
    top = results[0]
    assert "symbol" in top
    assert "code" in top
    assert len(top["pq8_code_hex"]) == 96  # 48 bytes in hex format = 96 chars


def test_global_indexer_singleton_and_warm_sub_4ms_search():
    """Verify warm queries across indexed repos execute with sub-4ms ADC scan."""
    indexer = get_repo_indexer()
    assert len(indexer.chunks) > 0

    # Warmup query
    indexer.query_code("test", top_k=1)

    # Timed query
    t0 = time.perf_counter()
    res = indexer.query_code("circuit breaker DFA", top_k=2)
    lat_ms = (time.perf_counter() - t0) * 1000.0

    assert len(res) > 0
    # ADC scan itself is verified < 4.0ms
    assert res[0]["adc_scan_ms"] < 4.0
    assert lat_ms < 60.0  # Total Python invocation overhead included


# =========================================================================
# 2. Windows Named-Pipe IPC & Circuit Breaker Guard Tests
# =========================================================================

def test_silicon_guard_pipe_server_allows_benign_command():
    """Verify benign commands pass through named-pipe logic with 200 and HMAC receipt."""
    server = SiliconGuardPipeServer()

    req_json = json.dumps({"command": "git status"})
    resp_raw = server.handle_request_json(req_json)
    resp = json.loads(resp_raw)

    assert resp["decision"] == "ALLOW"
    assert resp["status_code"] == 200
    assert resp["receipt"].startswith("LUNAR-SEC-")
    assert "npu_timestamp" in resp
    # Benign commands pass DFA + Neural Gate (<25000µs accounting for cold start, well below 150ms spawn)
    assert resp["latency_us"] < 25000.0


def test_silicon_guard_pipe_server_blocks_destructive_command():
    """Verify catastrophic commands are physically blocked with 403."""
    server = SiliconGuardPipeServer()

    # Construct destructive command safely as string
    catastrophic_cmd = "rm" + " -rf /"
    req_json = json.dumps({"command": catastrophic_cmd})
    resp_raw = server.handle_request_json(req_json)
    resp = json.loads(resp_raw)

    assert resp["decision"] == "DENY"
    assert resp["status_code"] == 403
    assert resp["receipt"] == "LUNAR-BLOCKED-403"
    assert "DFA" in resp["tier"] or "NEURAL" in resp["tier"]


def test_named_pipe_client_roundtrip_ipc():
    """Verify send_guard_ipc_query communicates with the resident server."""
    srv = SiliconGuardPipeServer()
    srv.start_background()

    try:
        # Benign test
        res_benign = send_guard_ipc_query("python -m pytest tests/ -q", server_instance=srv)
        assert res_benign["decision"] == "ALLOW"
        assert res_benign["status_code"] == 200
        assert res_benign["receipt"].startswith("LUNAR-SEC-")

        # Dangerous test
        danger_cmd = "rm" + " -rf /"
        res_danger = send_guard_ipc_query(danger_cmd, server_instance=srv)
        assert res_danger["decision"] == "DENY"
        assert res_danger["status_code"] == 403
        assert res_danger["receipt"] == "LUNAR-BLOCKED-403"
    finally:
        srv.stop()


def test_pre_tool_circuit_breaker_hook_evaluation():
    """Verify circuit_breaker_hook.py evaluate_command returns correct rejection/allowance."""
    # Benign
    code, output = evaluate_command("echo 'Hello Lunar Lake'")
    assert code in (0, 200)
    assert "LUNAR-SEC-" in output or "PASSED" in output

    # Destructive
    danger_cmd = "rm" + " -rf /"
    code, output = evaluate_command(danger_cmd)
    assert code == 403
    assert "BLOCKED" in output


# =========================================================================
# 3. Studio Flight Recorder & Token Stats API Endpoints Tests
# =========================================================================

@pytest.fixture(scope="module")
def studio_server():
    """Spawn ephemeral Studio HTTP server for integration verification."""
    server = ReusableThreadingServer(("127.0.0.1", 0), LunarStudioHandler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    base_url = f"http://127.0.0.1:{port}"
    yield base_url
    server.shutdown()
    server.server_close()


def test_studio_real_code_query_endpoint(studio_server):
    """Verify GET /api/query returns real indexed code chunks and PQ8 codes."""
    req = urllib.request.Request(f"{studio_server}/api/query?q=DualProfileGovernor", headers={"Connection": "close"})
    with urllib.request.urlopen(req, timeout=10) as resp:
        assert resp.status == 200
        data = json.loads(resp.read().decode("utf-8"))

        assert isinstance(data, list)
        assert len(data) > 0

        first = data[0]
        assert "symbol" in first
        assert "file" in first
        assert "pq8_code_hex" in first


def test_studio_token_savings_ticker_endpoint(studio_server):
    """Verify GET /api/tokens/stats returns cumulative tokens and cache hit telemetry."""
    req = urllib.request.Request(f"{studio_server}/api/tokens/stats", headers={"Connection": "close"})
    with urllib.request.urlopen(req, timeout=10) as resp:
        assert resp.status == 200
        data = json.loads(resp.read().decode("utf-8"))

        assert "total_tokens_saved" in data
        assert "dollars_saved" in data
        assert "code_symbols_indexed" in data
        assert data["total_tokens_saved"] > 0


def test_studio_memory_inspector_endpoint(studio_server):
    """Verify GET /api/memory/inspect returns indexed symbols table."""
    req = urllib.request.Request(f"{studio_server}/api/memory/inspect", headers={"Connection": "close"})
    with urllib.request.urlopen(req, timeout=10) as resp:
        assert resp.status == 200
        data = json.loads(resp.read().decode("utf-8"))

        assert data["status"] == "ok"
        assert "total_indexed" in data
        assert "chunks" in data
        assert len(data["chunks"]) > 0
        first = data["chunks"][0]
        assert "symbol" in first
        assert "pq8_code_hex" in first


def test_studio_sse_stream_endpoint(studio_server):
    """Verify GET /api/stream opens an EventStream."""
    req = urllib.request.Request(f"{studio_server}/api/stream", headers={"Connection": "close"})
    with urllib.request.urlopen(req, timeout=10) as resp:
        assert resp.status == 200
        assert "text/event-stream" in resp.headers.get("Content-Type", "")
        # Read the initial event
        chunk = resp.read(256).decode("utf-8", errors="replace")
        assert len(chunk) > 0
        assert "event:" in chunk or "INIT" in chunk
