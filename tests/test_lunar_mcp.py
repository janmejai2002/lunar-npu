"""Unit tests for LunarMCPServer (Model Context Protocol)."""

import json
import pytest
from lunar_core.mcp_server import LunarMCPServer


def test_mcp_initialize():
    server = LunarMCPServer()
    req = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2024-11-05",
            "clientInfo": {"name": "test-client", "version": "1.0"},
        },
    }
    resp = server.handle_request(req)
    assert resp is not None
    assert resp["id"] == 1
    assert "result" in resp
    assert resp["result"]["serverInfo"]["name"] == "lunar-npu-mcp"
    assert "tools" in resp["result"]["capabilities"]


def test_mcp_tools_list():
    server = LunarMCPServer()
    req = {
        "jsonrpc": "2.0",
        "id": 2,
        "method": "tools/list",
    }
    resp = server.handle_request(req)
    assert resp is not None
    assert resp["id"] == 2
    tools = resp["result"]["tools"]
    tool_names = [t["name"] for t in tools]
    assert "lunar_status" in tool_names
    assert "lunar_mamba_step" in tool_names
    assert "lunar_vector_search" in tool_names
    assert "lunar_circuit_breaker_audit" in tool_names


def test_mcp_tool_call_status():
    server = LunarMCPServer()
    req = {
        "jsonrpc": "2.0",
        "id": 3,
        "method": "tools/call",
        "params": {
            "name": "lunar_status",
            "arguments": {},
        },
    }
    resp = server.handle_request(req)
    assert resp is not None
    assert resp["id"] == 3
    assert resp["result"]["isError"] is False
    content = resp["result"]["content"][0]["text"]
    data = json.loads(content)
    assert "device" in data
    assert "available_devices" in data


def test_mcp_tool_call_circuit_breaker():
    server = LunarMCPServer()
    # Test blocked malicious command
    req_blocked = {
        "jsonrpc": "2.0",
        "id": 4,
        "method": "tools/call",
        "params": {
            "name": "lunar_circuit_breaker_audit",
            "arguments": {"command": "rm -rf /"},
        },
    }
    resp_blocked = server.handle_request(req_blocked)
    content = json.loads(resp_blocked["result"]["content"][0]["text"])
    assert content["verdict"] == "BLOCKED"

    # Test allowed safe command
    req_allowed = {
        "jsonrpc": "2.0",
        "id": 5,
        "method": "tools/call",
        "params": {
            "name": "lunar_circuit_breaker_audit",
            "arguments": {"command": "git status"},
        },
    }
    resp_allowed = server.handle_request(req_allowed)
    content_allowed = json.loads(resp_allowed["result"]["content"][0]["text"])
    assert content_allowed["verdict"] == "ALLOWED"
