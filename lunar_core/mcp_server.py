"""
Lunar NPU Model Context Protocol (MCP) Server
==============================================
Agent-native Model Context Protocol (MCP) interface for Intel Lunar Lake
47 TOPS NPU, Mamba SSM recurrence, Vector Memory, and Silicon Circuit Breaker.

Enables seamless one-click integration into Cursor, Claude Desktop, Antigravity,
Cline, Windsurf, and custom autonomous agent runtimes.
"""

from __future__ import annotations

import sys
import json
import traceback
from typing import Any, Dict, List, Optional

from lunar_core.engine import LunarNPUEngine
from lunar_core.mamba_ssm import LunarMambaEngine
from lunar_core.vector_memory import LunarVectorMemory
from lunar_core.circuit_breaker import SiliconCircuitBreaker
from lunar_core.router import MicroRouter


TOOLS_DEFINITIONS = [
    {
        "name": "lunar_status",
        "description": "Inspect Intel Lunar Lake NPU hardware properties, physical NCE tiles, driver version, and peak INT8 TOPS.",
        "inputSchema": {
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
    },
    {
        "name": "lunar_mamba_step",
        "description": "Execute constant-memory Mamba SSM recurrence steps (O(1) hidden state) on Intel NPU silicon without KV-cache.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "steps": {
                    "type": "integer",
                    "description": "Number of recurrent steps to benchmark (default 50)",
                    "default": 50,
                }
            },
            "additionalProperties": False,
        },
    },
    {
        "name": "lunar_vector_search",
        "description": "Query dense edge vector memory on normalized unit hypersphere S^383 using sub-3ms cosine similarity.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Natural language query to search stored knowledge",
                },
                "top_k": {
                    "type": "integer",
                    "description": "Number of top results to return (default 3)",
                    "default": 3,
                },
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    },
    {
        "name": "lunar_circuit_breaker_audit",
        "description": "Deterministic 2.2µs Silicon Circuit Breaker safety gatekeeper. Audits shell commands or agent actions before execution.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The shell command, script, or syscall to audit for destructive hazards.",
                }
            },
            "required": ["command"],
            "additionalProperties": False,
        },
    },
    {
        "name": "lunar_add_memory",
        "description": "Embed and store a fact or document in on-device NPU vector memory.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "Document content to embed and store in S^383 memory.",
                },
                "metadata": {
                    "type": "object",
                    "description": "Optional key-value metadata tags.",
                },
            },
            "required": ["text"],
            "additionalProperties": False,
        },
    },
    {
        "name": "lunar_route_task",
        "description": "Classify task prompt and route to optimal agent archetype (CODER, ARCHITECT, TESTER_DEVOPS, RESEARCHER, SECURITY_AUDITOR) in <3ms on NPU.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "The task prompt or user request to classify and route.",
                },
                "temperature": {
                    "type": "number",
                    "description": "Temperature scaling for softmax confidence calibration (default 0.1).",
                    "default": 0.1,
                },
            },
            "required": ["prompt"],
            "additionalProperties": False,
        },
    },
]



class LunarMCPServer:
    """Stdio JSON-RPC 2.0 Model Context Protocol server."""

    def __init__(self) -> None:
        self.engine = LunarNPUEngine()
        self.mamba = LunarMambaEngine(engine=self.engine, d_inner=64, d_state=16)
        self.vmem = LunarVectorMemory(engine=self.engine, embedding_dim=384)
        self.cb = SiliconCircuitBreaker(engine=self.engine)
        self.router = MicroRouter(memory_engine=self.vmem)

        # Seed default architectural memory if empty
        if not self.vmem.documents:
            self.vmem.add_document("Intel Lunar Lake microarchitecture features 6 NCE physical tiles.")
            self.vmem.add_document("Mamba recurrence operates with zero dynamic memory allocation and constant O(1) state.")
            self.vmem.add_document("Silicon Circuit Breakers enforce deterministic kernel guardrails in 2.2 microseconds.")
            self.vmem.add_document("Lunar Lake on-package LPDDR5X-8533 enables zero-copy heterogeneous UMA sharing.")

    def handle_request(self, request: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        msg_id = request.get("id")
        method = request.get("method")
        params = request.get("params", {})

        # Handle notifications
        if msg_id is None:
            return None

        try:
            if method == "initialize":
                return {
                    "jsonrpc": "2.0",
                    "id": msg_id,
                    "result": {
                        "protocolVersion": "2024-11-05",
                        "capabilities": {
                            "tools": {}
                        },
                        "serverInfo": {
                            "name": "lunar-npu-mcp",
                            "version": "1.0.0",
                        },
                    },
                }

            elif method == "ping":
                return {"jsonrpc": "2.0", "id": msg_id, "result": {}}

            elif method == "tools/list":
                return {
                    "jsonrpc": "2.0",
                    "id": msg_id,
                    "result": {
                        "tools": TOOLS_DEFINITIONS
                    },
                }

            elif method == "tools/call":
                tool_name = params.get("name")
                tool_args = params.get("arguments", {})
                result_text = self._call_tool(tool_name, tool_args)
                return {
                    "jsonrpc": "2.0",
                    "id": msg_id,
                    "result": {
                        "content": [
                            {"type": "text", "text": result_text}
                        ],
                        "isError": False,
                    },
                }

            else:
                return {
                    "jsonrpc": "2.0",
                    "id": msg_id,
                    "error": {
                        "code": -32601,
                        "message": f"Method not found: {method}",
                    },
                }

        except Exception as e:
            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "error": {
                    "code": -32000,
                    "message": str(e),
                    "data": traceback.format_exc(),
                },
            }

    def _call_tool(self, name: str, args: Dict[str, Any]) -> str:
        if name == "lunar_status":
            info = self.engine.get_device_info()
            return json.dumps(info, indent=2)

        elif name == "lunar_mamba_step":
            steps = int(args.get("steps", 50))
            res = self.mamba.benchmark(num_steps=steps)
            return json.dumps(res, indent=2)

        elif name == "lunar_vector_search":
            query = args["query"]
            top_k = int(args.get("top_k", 3))
            results = self.vmem.query(query, top_k=top_k)
            return json.dumps(results, indent=2)

        elif name == "lunar_circuit_breaker_audit":
            cmd = args["command"]
            verdict = self.cb.audit(cmd)
            return json.dumps(verdict, indent=2)

        elif name == "lunar_add_memory":
            text = args["text"]
            meta = args.get("metadata", {})
            entry = self.vmem.add_document(text, metadata=meta)
            return json.dumps({
                "status": "STORED",
                "id": entry["id"],
                "latency_ms": entry["latency_ms"],
            }, indent=2)

        elif name == "lunar_route_task":
            prompt = args["prompt"]
            temp = float(args.get("temperature", 0.1))
            decision = self.router.route(prompt, temperature=temp)
            return json.dumps(decision.to_dict(), indent=2)

        else:
            raise ValueError(f"Unknown tool: {name}")


    def run_stdio(self) -> None:
        """Run standard I/O communication loop."""
        for raw_line in sys.stdin:
            line = raw_line.strip()
            if not line:
                continue

            # Header-based MCP message
            if line.startswith("Content-Length:"):
                length = int(line.split(":")[1].strip())
                sys.stdin.readline()
                content = sys.stdin.read(length)
                req = json.loads(content)
                resp = self.handle_request(req)
                if resp is not None:
                    sys.stdout.write(json.dumps(resp) + "\n")
                    sys.stdout.flush()
                continue

            # Line-delimited JSON message (skip any BOM or non-JSON prefix)
            idx = line.find("{")
            if idx == -1:
                continue
            line = line[idx:]
            req = json.loads(line)
            resp = self.handle_request(req)
            if resp is not None:
                sys.stdout.write(json.dumps(resp) + "\n")
                sys.stdout.flush()



def main():
    server = LunarMCPServer()
    server.run_stdio()


if __name__ == "__main__":
    main()
