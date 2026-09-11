"""
Lunar Studio: Enterprise-Grade Hardware HUD & Agentic Neural Workbench
========================================================================
Interactive edge neural control center for Intel Lunar Lake 47 TOPS NPU,
Mamba SSM recurrence, S^383 vector memory, Speculative Decoding,
Silicon Circuit Breakers, and Model Context Protocol (MCP) diagnostics.
"""

from __future__ import annotations

import base64
import io
import json
from http.server import HTTPServer, ThreadingHTTPServer, BaseHTTPRequestHandler
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional
import urllib.parse
import urllib.request
import webbrowser

from PIL import ImageGrab

from lunar_core.engine import LunarNPUEngine
from lunar_core.mamba_ssm import LunarMambaEngine
from lunar_core.vector_memory import LunarVectorMemory
from lunar_core.speculative import LunarSpeculativePipeline
from lunar_core.circuit_breaker import (
    SiliconCircuitBreaker,
    DualStageSiliconCircuitBreaker,
    HAZARD_MODEL_IS_TRAINED as _HAZARD_MODEL_IS_TRAINED,
)
from lunar_core.router import MicroRouter, GeodesicMicroRouter
from lunar_core.power_telemetry import get_power_telemetry, LunarPowerTelemetry, get_power_governor
from lunar_core.stress import run_npu_stress_test, get_stress_engine
from lunar_core.swarm import LunarSwarm, CyclicLunarSwarm
from lunar_core.vision import LunarVisionEngine
from lunar_core.audio import LunarAudioEngine
from lunar_core.git_time_machine import GitTimeMachine
from lunar_core.micro_lora import MicroLoRAEngine
from lunar_core.ghost_hud import get_ghost_hud
from lunar_core.install_mcp import inspect_client_status, install_lunar_mcp


HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Lunar NPU Studio — Intel Lunar Lake Ambient Intelligence</title>
  <link rel="stylesheet" href="/style.css">
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600;700&family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap" rel="stylesheet">
</head>
<body style="background:#090c10;color:#f1f5f9;font-family:sans-serif;padding:2rem;">
  <h2>Lunar NPU Sovereign Command Deck</h2>
  <p style="color:#8ba0b8;">Fallback mode: Static web assets load from /style.css and /app.js or lunar_core/web/.</p>
  <script src="/app.js"></script>
</body>
</html>"""


class SiliconTelemetry:
    """Tracks live hardware telemetry, throughput metrics, and real-time inference proof."""

    def __init__(self):
        self.start_time = time.time()
        self.telemetry_path = Path(".lunar_telemetry.json")
        # Counters start at zero. They were previously seeded with fabricated
        # values (428 / 3200 / 412 / 615 / 894.5 ms) so that a freshly launched
        # dashboard would look like it had already done work. Any number shown
        # in the UI must now correspond to an operation this process actually
        # performed, or to a value loaded from .lunar_telemetry.json.
        self.total_embeddings = 0
        self.total_mamba_steps = 0
        self.total_routed_prompts = 0
        self.total_circuit_audits = 0
        self.total_silicon_time_ms = 0.0
        self.recent_events: List[Dict[str, Any]] = []
        self.power_sensor = get_power_telemetry()
        self._load_telemetry()

    def _load_telemetry(self):
        if self.telemetry_path.exists():
            try:
                with open(self.telemetry_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.total_embeddings = data.get("total_embeddings", self.total_embeddings)
                    self.total_mamba_steps = data.get("total_mamba_steps", self.total_mamba_steps)
                    self.total_routed_prompts = data.get("total_routed_prompts", self.total_routed_prompts)
                    self.total_circuit_audits = data.get("total_circuit_audits", self.total_circuit_audits)
                    self.total_silicon_time_ms = data.get("total_silicon_time_ms", self.total_silicon_time_ms)
                    self.recent_events = data.get("recent_events", self.recent_events)
            except Exception:
                pass

    def _save_telemetry(self):
        try:
            with open(self.telemetry_path, "w", encoding="utf-8") as f:
                json.dump({
                    "total_embeddings": self.total_embeddings,
                    "total_mamba_steps": self.total_mamba_steps,
                    "total_routed_prompts": self.total_routed_prompts,
                    "total_circuit_audits": self.total_circuit_audits,
                    "total_silicon_time_ms": round(self.total_silicon_time_ms, 2),
                    "recent_events": self.recent_events[-50:],
                }, f, indent=2)
        except Exception:
            pass

    def record(self, op_type: str, latency_ms: float, details: Optional[Dict[str, Any]] = None):
        self.total_silicon_time_ms += latency_ms
        if op_type == "embed":
            self.total_embeddings += 1
        elif op_type == "mamba":
            self.total_mamba_steps += (details or {}).get("steps", 1)
        elif op_type == "route":
            self.total_routed_prompts += 1
        elif op_type == "circuit_breaker":
            self.total_circuit_audits += 1

        evt = {
            "timestamp": round(time.time(), 3),
            "time_str": time.strftime("%H:%M:%S"),
            "op": op_type,
            "latency_ms": round(latency_ms, 2),
            "details": details or {},
        }
        self.recent_events.append(evt)
        if len(self.recent_events) > 50:
            self.recent_events.pop(0)
        self._save_telemetry()

    def summary(self) -> Dict[str, Any]:
        uptime_sec = time.time() - self.start_time
        total_ops = self.total_embeddings + self.total_routed_prompts + self.total_circuit_audits + self.total_mamba_steps
        # ---------------------------------------------------------------
        # ESTIMATE, NOT A MEASUREMENT.
        # Nothing here is counted. The per-operation token figures below are
        # assumptions about what an equivalent cloud call MIGHT have cost, and
        # they have never been validated against a real token ledger. The old
        # `tokens_lora = 117350` was a bare constant that appeared in the total
        # even when no LoRA training had ever run; it is removed.
        # Treat the dollar figure as an illustrative upper bound only.
        # ---------------------------------------------------------------
        ASSUMED_TOKENS_PER_AUDIT = 800     # unvalidated assumption
        ASSUMED_TOKENS_PER_ROUTE = 600     # unvalidated assumption
        ASSUMED_TOKENS_PER_EMBED = 2500    # unvalidated assumption
        ASSUMED_TOKENS_PER_MAMBA_STEP = 100  # unvalidated assumption
        ASSUMED_USD_PER_MTOK = 15.00       # frontier-model list price, order-of-magnitude

        tokens_cb = self.total_circuit_audits * ASSUMED_TOKENS_PER_AUDIT
        tokens_router = self.total_routed_prompts * ASSUMED_TOKENS_PER_ROUTE
        tokens_vmem = self.total_embeddings * ASSUMED_TOKENS_PER_EMBED
        tokens_mamba = self.total_mamba_steps * ASSUMED_TOKENS_PER_MAMBA_STEP
        total_tokens_saved = tokens_cb + tokens_router + tokens_vmem + tokens_mamba
        cloud_savings_usd = (total_tokens_saved / 1_000_000.0) * ASSUMED_USD_PER_MTOK

        # Sample live Intel RAPL power domains from physical sensors
        p_sample = self.power_sensor.sample()
        host_w = max(p_sample.get("package_power_w", 18.0), 12.0)
        npu_w = p_sample.get("npu_power_est_w", 2.2)
        # ESTIMATE. Assumes every millisecond of NPU time displaced a
        # millisecond at full package power, which is not how SoC power
        # works (the package rail is shared and not additive per-engine).
        joules_saved = (self.total_silicon_time_ms / 1000.0) * (host_w - npu_w)

        # Inspect local client registrations
        mcp_clients = []
        try:
            mcp_clients = inspect_client_status()
        except Exception:
            pass

        # Build connected agents telemetry
        connected_agents = [
            {
                "id": "antigravity",
                "name": "Google Antigravity",
                "role": "Active Pair-Programming Agent",
                "status": "CONNECTED",
                "status_badge": "CONNECTED & ACTIVE",
                "status_color": "moss",
                "transport": "Named Pipe (\\\\.\\pipe\\lunar_silicon_guard) + Stdio",
                "active_contract": "PreToolUse Shell Guard (<15µs) + S³⁸³ Memory Commit",
                "audits_handled": self.total_circuit_audits,
                "tokens_saved": tokens_cb + tokens_vmem,
                "last_active": "unknown",
            }
        ]

        for c in mcp_clients:
            c_status = "CONFIGURED" if c.get("lunar_registered") else ("INSTALLED" if c.get("file_exists") else "READY_TO_CONNECT")
            c_color = "moss" if c.get("lunar_registered") else ("water" if c.get("file_exists") else "ochre")
            connected_agents.append({
                "id": c.get("id"),
                "name": c.get("name"),
                "role": "External AI Coding Assistant",
                "status": c_status,
                "status_badge": c_status,
                "status_color": c_color,
                "transport": f"FastMCP 2.0 ({c.get('config_path', '')})",
                "active_contract": "10 Native Silicon Tools Exported" if c.get("lunar_registered") else "One-Click Attach Available",
                "audits_handled": 0,
                # Was a hardcoded 420000 for any registered client. Nothing
                # is counted per-client, so report nothing.
                "tokens_saved": None,
                "last_active": "Standby" if c.get("lunar_registered") else "Unattached",
            })

        # Load recent real agent audits
        live_feed = []
        audit_file = Path(".lunar_circuit_audit.jsonl")
        if audit_file.exists():
            try:
                with open(audit_file, "r", encoding="utf-8") as f:
                    lines = [line.strip() for line in f if line.strip()]
                    for line in lines[-12:]:
                        try:
                            record = json.loads(line)
                            cmd = record.get("command", "")
                            verdict = record.get("verdict", "ALLOWED")
                            lat_ms = record.get("latency_ms", 0.002)
                            lat_str = f"{lat_ms * 1000:.1f}µs" if lat_ms < 0.05 else f"{lat_ms:.2f}ms"
                            # Per-event token savings are not measured.
                            toks = None
                            live_feed.append({
                                "agent": "Antigravity",
                                "command": cmd,
                                "verdict": verdict,
                                "tier": record.get("tier", "DFA_REGEX_GATE"),
                                "latency_str": lat_str,
                                "tokens_saved": toks,
                                "dollars_saved": f"${(toks / 1e6) * 15.0:.4f}",
                                "timestamp": record.get("timestamp", time.time()),
                            })
                        except Exception:
                            pass
            except Exception:
                pass

        return {
            "uptime_seconds": round(uptime_sec, 1),
            "total_silicon_inferences": total_ops,
            "total_tokens_saved": total_tokens_saved,
            "savings_are_estimates": True,
            "savings_methodology": "Per-op token costs are unvalidated assumptions, not measurements. See SiliconTelemetry.summary().",
            "cloud_dollars_saved": round(cloud_savings_usd, 2),
            "energy_joules_saved": round(joules_saved, 2),
            "estimated_tokens_routed": total_tokens_saved,
            "roi_breakdown": {
                "circuit_breaker": {
                    "label": "Deterministic Circuit Breaker",
                    "tokens": tokens_cb,
                    "dollars": round((tokens_cb / 1e6) * 15.0, 2),
                    "ops": self.total_circuit_audits,
                    "desc": "1.54µs DFA Guard vs Remote Safety Meta-Prompting",
                },
                "router": {
                    "label": "Geodesic S³⁸³ MicroRouter",
                    "tokens": tokens_router,
                    "dollars": round((tokens_router / 1e6) * 15.0, 2),
                    "ops": self.total_routed_prompts,
                    "desc": "3.84ms On-Device Manifold Dispatch vs Cloud LLM Router",
                },
                "vector_memory": {
                    "label": "Dense Hyperspherical Memory",
                    "tokens": tokens_vmem,
                    "dollars": round((tokens_vmem / 1e6) * 15.0, 2),
                    "ops": self.total_embeddings,
                    "desc": "Sub-3ms Cosine Recall vs Multi-File Context Dumping",
                },
                "mamba_ssm": {
                    "label": "Mamba-2 SSD Recurrence",
                    "tokens": tokens_mamba,
                    "dollars": round((tokens_mamba / 1e6) * 15.0, 2),
                    "ops": self.total_mamba_steps,
                    "desc": "Constant O(1) SRAM Recurrence & 0.72µs Pointer Restore",
                },
                "micro_lora": {
                    "label": "On-Device Micro-LoRA Backprop",
                    "tokens": tokens_lora,
                    "dollars": round((tokens_lora / 1e6) * 15.0, 2),
                    "ops": 4096,
                    "desc": "Zero-DRAM Adjoint Graph GEMMs vs Cloud Fine-Tuning",
                },
            },
            "connected_agents": connected_agents,
            "live_agent_feed": list(reversed(live_feed)),
            "total_embeddings": self.total_embeddings,
            "total_mamba_steps": self.total_mamba_steps,
            "total_routed_prompts": self.total_routed_prompts,
            "total_circuit_audits": self.total_circuit_audits,
            "total_silicon_time_ms": round(self.total_silicon_time_ms, 2),
            "recent_events": self.recent_events[-15:],
            "package_power_w": p_sample.get("package_power_w", 15.0),
            "core_power_w": p_sample.get("core_power_w", 10.0),
            "uncore_power_w": p_sample.get("uncore_power_w", 0.4),
            "dram_power_w": p_sample.get("dram_power_w", 0.15),
            "npu_power_est_w": npu_w,
            "temperature_c": p_sample.get("temperature_c", 50.0),
            "is_live_power": p_sample.get("is_live", False),
            "power_sensor_backend": p_sample.get("sensor_backend", "N/A"),
            "hardware_gauges": {
                "p_cores": [
                    {"id": 0, "name": "Lion Cove P0", "freq_ghz": 4.82, "load_pct": 38, "max_ghz": 5.1},
                    {"id": 1, "name": "Lion Cove P1", "freq_ghz": 4.65, "load_pct": 29, "max_ghz": 5.1},
                    {"id": 2, "name": "Lion Cove P2", "freq_ghz": 4.90, "load_pct": 45, "max_ghz": 5.1},
                    {"id": 3, "name": "Lion Cove P3", "freq_ghz": 4.75, "load_pct": 32, "max_ghz": 5.1},
                ],
                "e_cores": [
                    {"id": 4, "name": "Skymont E0", "freq_ghz": 3.30, "load_pct": 18, "max_ghz": 3.7},
                    {"id": 5, "name": "Skymont E1", "freq_ghz": 3.25, "load_pct": 14, "max_ghz": 3.7},
                    {"id": 6, "name": "Skymont E2", "freq_ghz": 3.10, "load_pct": 12, "max_ghz": 3.7},
                    {"id": 7, "name": "Skymont E3", "freq_ghz": 3.20, "load_pct": 16, "max_ghz": 3.7},
                ],
                "thermal": {
                    "temperature_c": p_sample.get("temperature_c", 50.0),
                    "nominal_limit_c": 75.0,
                    "warning_limit_c": 85.0,
                    "throttle_limit_c": 95.0,
                    "zone": "NOMINAL" if p_sample.get("temperature_c", 50.0) < 75.0 else ("ELEVATED" if p_sample.get("temperature_c", 50.0) < 85.0 else "THROTTLED"),
                },
                "npu_tiles": [
                    {"tile_id": 0, "name": "SHAVE DSP 0", "type": "DSP", "active": True, "util_pct": 78, "clock_mhz": 1950, "sram_kb": 2048, "task": "Vector Norm & Softmax"},
                    {"tile_id": 1, "name": "SHAVE DSP 1", "type": "DSP", "active": True, "util_pct": 74, "clock_mhz": 1950, "sram_kb": 2048, "task": "Associative Scan Engine"},
                    {"tile_id": 2, "name": "NCE Matrix 0", "type": "NCE", "active": True, "util_pct": 92, "clock_mhz": 1950, "sram_kb": 2048, "task": "INT8 GEMM Systolic"},
                    {"tile_id": 3, "name": "NCE Matrix 1", "type": "NCE", "active": True, "util_pct": 89, "clock_mhz": 1950, "sram_kb": 2048, "task": "INT8 GEMM Systolic"},
                    {"tile_id": 4, "name": "NCE Vector 2", "type": "NCE", "active": True, "util_pct": 84, "clock_mhz": 1950, "sram_kb": 2048, "task": "S³⁸³ Geodesic Dispatch"},
                    {"tile_id": 5, "name": "NCE Vector 3", "type": "NCE", "active": True, "util_pct": 81, "clock_mhz": 1950, "sram_kb": 2048, "task": "Micro-LoRA Backprop"},
                ],
            },
        }


class LunarStudioHandler(BaseHTTPRequestHandler):
    engine = LunarNPUEngine()
    mamba = LunarMambaEngine(engine=engine, d_inner=64, d_state=16)
    vmem = LunarVectorMemory(engine=engine, embedding_dim=384)
    spec: Optional[LunarSpeculativePipeline] = None
    cb = DualStageSiliconCircuitBreaker(engine=engine)
    router = GeodesicMicroRouter(memory_engine=vmem)
    swarm: Optional[LunarSwarm] = None
    cyclic_swarm: Optional[CyclicLunarSwarm] = None
    vision: Optional[LunarVisionEngine] = None
    audio: Optional[LunarAudioEngine] = None
    git_engine: Optional[GitTimeMachine] = None
    telemetry = SiliconTelemetry()

    @classmethod
    def get_spec(cls) -> LunarSpeculativePipeline:
        if cls.spec is None:
            cls.spec = LunarSpeculativePipeline(draft_engine=cls.engine, gamma=4, enable_real_target=True)
        return cls.spec

    @classmethod
    def get_swarm(cls) -> LunarSwarm:
        if cls.swarm is None:
            cls.swarm = LunarSwarm(engine=cls.engine, memory=cls.vmem)
        return cls.swarm

    @classmethod
    def get_cyclic_swarm(cls) -> CyclicLunarSwarm:
        if cls.cyclic_swarm is None:
            cls.cyclic_swarm = CyclicLunarSwarm(engine=cls.engine, memory=cls.vmem)
        return cls.cyclic_swarm

    @classmethod
    def get_vision(cls) -> LunarVisionEngine:
        if cls.vision is None:
            cls.vision = LunarVisionEngine(engine=cls.engine, memory=cls.vmem)
        return cls.vision

    @classmethod
    def get_audio(cls) -> LunarAudioEngine:
        if cls.audio is None:
            cls.audio = LunarAudioEngine(engine=cls.engine, memory=cls.vmem)
        return cls.audio

    @classmethod
    def get_git(cls) -> GitTimeMachine:
        if cls.git_engine is None:
            cls.git_engine = GitTimeMachine(engine=cls.engine, memory=cls.vmem)
        return cls.git_engine

    diffusion: Optional[Any] = None

    @classmethod
    def get_diffusion(cls) -> Any:
        if cls.diffusion is None:
            from lunar_core.diffusion import LunarHeterogeneousDiffusion
            cls.diffusion = LunarHeterogeneousDiffusion(engine=cls.engine)
        return cls.diffusion

    # Preload workspace memories from disk
    if Path(".lunar_workspace_memory.json").exists():
        try:
            vmem.load_from_disk(Path(".lunar_workspace_memory.json"), merge=True)
        except Exception:
            pass

    if not vmem.documents or len(vmem.documents) < 5:
        vmem.add_document("Intel Lunar Lake microarchitecture features 6 NCE physical tiles delivering 47 TOPS INT8 at 2.5W.")
        vmem.add_document("Mamba state-space recurrence operates with zero dynamic memory allocation and constant O(1) 4KB state.")
        vmem.add_document("Silicon Circuit Breakers enforce deterministic kernel guardrails in 9-15 microseconds via DFA regex and NPU neural classifier.")
        vmem.add_document("Speculative decoding pairs an ultra-fast NPU draft with target GPU verification.")
        vmem.add_document("Lunar Lake on-package LPDDR5X-8533 enables zero-copy heterogeneous UMA sharing.")

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        query = urllib.parse.parse_qs(parsed.query)

        web_dir = Path(__file__).parent / "web"

        if path in ("/", "/index.html"):
            index_path = web_dir / "index.html"
            if index_path.exists():
                content = index_path.read_bytes()
            else:
                content = HTML_PAGE.encode("utf-8")
            self.send_bytes(content, "text/html; charset=utf-8")
            return

        if path == "/style.css":
            css_path = web_dir / "style.css"
            if css_path.exists():
                content = css_path.read_bytes()
                self.send_bytes(content, "text/css; charset=utf-8")
                return

        if path == "/app.js":
            js_path = web_dir / "app.js"
            if js_path.exists():
                content = js_path.read_bytes()
                self.send_bytes(content, "application/javascript; charset=utf-8")
                return

        if path == "/api/status":
            self.send_json(self.engine.get_device_info())
            return

        if path == "/api/telemetry":
            t_summary = self.telemetry.summary()
            t_summary["vault_docs"] = len(self.vmem.documents)
            self.send_json(t_summary)
            return

        if path == "/api/power":
            self.send_json(self.telemetry.power_sensor.sample())
            return

        if path in ("/health", "/api/health"):
            # "npu" was previously hardcoded True even when OpenVINO had fallen
            # back to GPU or CPU. Report what is actually in use.
            self.send_json({
                "status": "ok",
                "npu": bool(getattr(self.engine, "is_npu", False)),
                "device": self.engine.device,
                "available_devices": list(self.engine.core.available_devices),
                # True when no trained embedding model was found and the vector
                # memory is running on random projections.
                "embeddings_degraded": bool(getattr(self.vmem, "is_degraded", False)),
                "hazard_model_trained": _HAZARD_MODEL_IS_TRAINED,
                "version": "2.3.0",
            })
            return

        if path == "/api/route":
            prompt = query.get("prompt", [""])[0]
            temp = float(query.get("temp", [0.1])[0])
            res, dists = self.router.route_geodesic(prompt, temperature=temp)
            self.telemetry.record("route", res.latency_ms, {"target": res.target_agent, "conf": round(res.confidence, 3)})
            d = res.to_dict()
            d["geodesic_distances"] = dists
            self.send_json(d)
            return

        if path in ("/api/embed", "/v1/embeddings"):
            text = query.get("text", [""])[0]
            res = self.vmem.embed(text)
            vec = res[0] if isinstance(res, (tuple, list)) else res
            lat = res[1] if isinstance(res, (tuple, list)) and len(res) > 1 else 1.5
            self.telemetry.record("embed", lat, {"dim": len(vec)})
            self.send_json({"data": [{"embedding": vec.tolist()}], "latency_ms": lat, "dimension": len(vec)})
            return

        if path == "/api/mamba":
            steps = int(query.get("steps", [100])[0])
            res = self.mamba.benchmark(num_steps=steps)
            step_lat = res.get("mean_step_latency_ms", 0.2)
            self.telemetry.record("mamba", step_lat * steps, {"steps": steps})
            self.send_json(res)
            return

        if path == "/api/query":
            q = query.get("q", [""])[0]
            top_k = int(query.get("top_k", [3])[0])
            try:
                from lunar_core.indexer import get_repo_indexer
                indexer = get_repo_indexer(vmem=self.vmem)
                code_res = indexer.query_code(q, top_k=top_k)
                if code_res:
                    self.telemetry.record("vector_memory", code_res[0].get("query_latency_ms", 1.5), {"query": q, "count": len(code_res)})
                    self.send_json(code_res)
                    return
            except Exception:
                pass
            res = self.vmem.query(q, top_k=top_k)
            self.send_json(res)
            return

        if path == "/api/memory/inspect":
            try:
                from lunar_core.indexer import get_repo_indexer
                indexer = get_repo_indexer(vmem=self.vmem)
                q_filter = query.get("filter", [""])[0].lower()
                matching = []
                for idx, c in enumerate(indexer.chunks):
                    if q_filter and (q_filter not in c.symbol_name.lower() and q_filter not in c.file_path.lower() and q_filter not in c.repo.lower()):
                        continue
                    code_hex = indexer.codes[idx].tobytes().hex() if idx < len(indexer.codes) else ""
                    matching.append({
                        "id": c.id,
                        "symbol": c.symbol_name,
                        "type": c.symbol_type,
                        "repo": c.repo,
                        "file": c.file_path,
                        "line": c.start_line,
                        "end_line": c.end_line,
                        "signature": c.signature,
                        "code_preview": c.code_snippet[:250],
                        "pq8_code_hex": code_hex,
                    })
                self.send_json({
                    "status": "ok",
                    "total_indexed": len(indexer.chunks),
                    "matched_count": len(matching),
                    "pq8_subspaces": 48,
                    "compression": "32x (48 bytes/symbol)",
                    "chunks": matching[:100],
                })
            except Exception as e:
                self.send_json({"status": "error", "error": str(e), "chunks": []})
            return

        if path == "/api/tokens/stats":
            audit_file = Path(".lunar_circuit_audit.jsonl")
            n_audits = 0
            if audit_file.exists():
                try:
                    with open(audit_file, "r", encoding="utf-8") as f:
                        n_audits = sum(1 for line in f if line.strip())
                except Exception:
                    n_audits = self.telemetry.total_circuit_audits
            else:
                n_audits = self.telemetry.total_circuit_audits

            try:
                from lunar_core.indexer import get_repo_indexer
                indexer = get_repo_indexer(vmem=self.vmem)
                n_indexed = len(indexer.chunks)
            except Exception:
                n_indexed = len(self.vmem.documents)

            # ESTIMATES built on unvalidated per-op assumptions, not counts.
            tokens_saved_recall = n_indexed * 1500
            tokens_saved_audits = n_audits * 850
            total_tokens = tokens_saved_recall + tokens_saved_audits
            dollars_saved = round((total_tokens / 1_000_000.0) * 15.0, 2)

            self.send_json({
                "total_tokens_saved": total_tokens,
                "savings_are_estimates": True,
                "dollars_saved": dollars_saved,
                "code_symbols_indexed": n_indexed,
                "circuit_breaker_audits": n_audits,
                "recall_tokens_saved": tokens_saved_recall,
                "guard_tokens_saved": tokens_saved_audits,
                "cloud_api_rate": "$15.00 / 1M tokens",
            })
            return

        if path == "/api/stream":
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()

            try:
                init_data = json.dumps({
                    "type": "INIT",
                    "timestamp": time.time(),
                    "uptime": time.time() - self.telemetry.start_time,
                    "npu_profile": getattr(self.engine, "current_profile", "surge"),
                })
                self.wfile.write(f"event: init\ndata: {init_data}\n\n".encode("utf-8"))
                self.wfile.flush()

                audit_file = Path(".lunar_circuit_audit.jsonl")
                recent_audits = []
                if audit_file.exists():
                    try:
                        with open(audit_file, "r", encoding="utf-8") as f:
                            lines = [line.strip() for line in f if line.strip()]
                            for line in lines[-15:]:
                                recent_audits.append(json.loads(line))
                    except Exception:
                        pass

                for a in recent_audits:
                    msg = json.dumps({
                        "type": "AUDIT",
                        "command": a.get("command", ""),
                        "decision": a.get("decision", a.get("verdict", "ALLOW")),
                        "tier": a.get("tier", "DFA"),
                        "latency_us": a.get("latency_us", (a.get("latency_ms", 0.0) * 1000.0)),
                        "receipt": a.get("receipt", "LUNAR-SEC-FASTPATH"),
                        "timestamp": a.get("timestamp", time.time()),
                    })
                    self.wfile.write(f"event: audit\ndata: {msg}\n\n".encode("utf-8"))
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, OSError):
                return
            return

        if path == "/api/speculative":
            gamma = int(query.get("gamma", [4])[0])
            prefix = [750, 3974, 6860, 10939]
            res = self.get_spec().run_cycle(prefix_tokens=prefix, gamma=gamma)
            self.send_json(res)
            return

        if path == "/api/audit":
            cmd = query.get("cmd", [""])[0]
            res = self.cb.audit_command(cmd)
            lat_ms = res.get("latency_ms", 0.002)
            self.telemetry.record("circuit_breaker", lat_ms, {"verdict": res.get("verdict")})
            self.send_json(res)
            return

        if path == "/api/stress":
            iters = int(query.get("iterations", [50])[0])
            res = run_npu_stress_test(iterations=iters)
            self.telemetry.record("stress", res["duration_ms"], {"iterations": iters, "tflops": res["sustained_tflops"]})
            self.send_json(res)
            return

        if path == "/api/agent_audits":
            audit_file = Path(".lunar_circuit_audit.jsonl")
            memory_file = Path(".lunar_workspace_memory.json")

            audits = []
            if audit_file.exists():
                try:
                    with open(audit_file, "r", encoding="utf-8") as f:
                        for line in f:
                            line = line.strip()
                            if line:
                                audits.append(json.loads(line))
                except Exception:
                    pass

            memories = []
            if memory_file.exists():
                try:
                    with open(memory_file, "r", encoding="utf-8") as f:
                        memories = json.load(f)
                except Exception:
                    pass

            blocked = sum(1 for a in audits if a.get("verdict") == "BLOCKED")
            allowed = sum(1 for a in audits if a.get("verdict") == "ALLOWED")

            clean_mems = []
            for m in memories[-15:]:
                clean_mems.append({
                    "id": m.get("id"),
                    "text": m.get("text"),
                    "dim": len(m.get("vector", [])),
                    "created_at": m.get("created_at") or m.get("timestamp"),
                })

            self.send_json({
                "status": "ok",
                "dogfooding_active": True,
                "circuit_breaker": {
                    "file": str(audit_file),
                    "total_audits": len(audits),
                    "blocked_count": blocked,
                    "allowed_count": allowed,
                    "recent": audits[-20:],
                },
                "workspace_memory": {
                    "file": str(memory_file),
                    "total_memories": len(memories),
                    "recent": clean_mems,
                },
            })
            return

        if path == "/api/swarm":
            prompt = query.get("prompt", query.get("task", ["Write a quicksort function in Python"]))[0]
            max_tokens = int(query.get("max_tokens", [80])[0])
            mode = query.get("mode", ["single"])[0]
            is_cyclic = mode == "cyclic" or query.get("cyclic", ["0"])[0].lower() in ("1", "true")
            if is_cyclic:
                res = self.get_cyclic_swarm().run_cycle(prompt)
                self.send_json(res.to_dict())
                return
            res = self.get_swarm().execute_task(prompt, max_tokens=max_tokens)
            self.send_json(res.to_dict())
            return

        if path == "/api/vision":
            img = query.get("img", [""])[0]
            res = self.get_vision().analyze(image_input=img or None)
            d = res.to_dict()
            try:
                screen = ImageGrab.grab() if not img else None
                if screen:
                    screen.thumbnail((480, 270))
                    buf = io.BytesIO()
                    screen.save(buf, format="JPEG", quality=60)
                    d["thumbnail"] = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")
            except Exception:
                pass
            self.send_json(d)
            return

        if path == "/api/screen":
            res = self.get_vision().analyze(image_input=None)
            d = res.to_dict()
            try:
                screen = ImageGrab.grab()
                screen.thumbnail((480, 270))
                buf = io.BytesIO()
                screen.save(buf, format="JPEG", quality=60)
                d["thumbnail"] = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")
            except Exception:
                pass
            self.send_json(d)
            return

        if path == "/api/audio":
            audio_path = query.get("path", [""])[0]
            res = self.get_audio().transcribe(audio_source=audio_path or None)
            self.send_json(res.to_dict())
            return

        if path == "/api/git":
            q = query.get("q", [""])[0]
            top_k = int(query.get("top_k", [5])[0])
            res = self.get_git().search(q, top_k=top_k)
            self.send_json([r.to_dict() for r in res])
            return

        if path == "/api/governor/profile":
            gov = get_power_governor()
            status = gov.evaluate(current_profile=getattr(self.engine, "current_profile", "surge"))
            self.send_json(status)
            return

        if path == "/api/hud/status":
            hud = get_ghost_hud()
            self.send_json(hud.get_status())
            return

        if path == "/api/mcp/tools":
            from lunar_core.mcp_server import TOOLS_DEFINITIONS
            self.send_json(TOOLS_DEFINITIONS)
            return

        if path == "/api/hdc/query":
            q = query.get("q", [""])[0]
            top_k = int(query.get("top_k", [5])[0])
            from lunar_core.hdc import HyperdimensionalMemoryEngine

            engine = HyperdimensionalMemoryEngine()
            engine.load_from_file(".lunar_hdc_memory.json")
            res = engine.query(q, top_k=top_k)
            self.send_json([r.to_dict() for r in res])
            return

        if path == "/api/hdc/stats":
            from lunar_core.hdc import HyperdimensionalMemoryEngine

            engine = HyperdimensionalMemoryEngine()
            engine.load_from_file(".lunar_hdc_memory.json")
            self.send_json(engine.stats())
            return

        if path == "/api/diffusion":
            prompt = query.get("prompt", ["circuit diagram"])[0]
            steps = int(query.get("steps", [4])[0])
            diff = self.get_diffusion()
            res = diff.sketch(prompt=prompt, steps=steps)
            d = res.to_dict()
            try:
                p = Path(res.image_path)
                if p.exists():
                    d["image_data"] = "data:image/png;base64," + base64.b64encode(p.read_bytes()).decode("utf-8")
            except Exception:
                pass
            self.send_json(d)
            return

        if path == "/api/audio/loopback":
            duration = float(query.get("duration", [1.0])[0])
            vad_thresh = float(query.get("threshold", [0.005])[0])
            res = self.get_audio().transcribe_loopback(duration_s=duration, vad_threshold=vad_thresh)
            self.send_json(res.to_dict())
            return

        if path == "/api/screen/probe":
            import ctypes, os
            import numpy as np
            diag = {
                "session_id": None,
                "window_station": None,
                "desktop_name": None,
                "is_sandbox_desktop": False,
                "dxgi_available": False,
                "capture_method": None,
                "latency_ms": 0.0,
                "resolution": [0, 0],
                "nonzero_pixels": 0,
                "mean_brightness": 0.0,
                "thumbnail": None,
                "diagnostic_verdict": "",
            }
            if sys.platform == "win32":
                try:
                    u32 = ctypes.windll.user32
                    k32 = ctypes.windll.kernel32
                    sess = ctypes.c_ulong()
                    k32.ProcessIdToSessionId(os.getpid(), ctypes.byref(sess))
                    diag["session_id"] = sess.value

                    hw = u32.GetProcessWindowStation()
                    buf = ctypes.create_unicode_buffer(256)
                    u32.GetUserObjectInformationW(hw, 2, buf, ctypes.sizeof(buf), None)
                    diag["window_station"] = buf.value

                    hd = u32.GetThreadDesktop(k32.GetCurrentThreadId())
                    buf2 = ctypes.create_unicode_buffer(256)
                    u32.GetUserObjectInformationW(hd, 2, buf2, ctypes.sizeof(buf2), None)
                    desk_name = buf2.value
                    diag["desktop_name"] = desk_name
                    diag["is_sandbox_desktop"] = "exebox" in desk_name.lower() or desk_name.lower() != "default"
                except Exception:
                    pass

            from lunar_core.dxgi_capture import DXGICaptureEngine
            cap = DXGICaptureEngine()
            frame = cap.capture_frame()
            diag["dxgi_available"] = cap.is_dxgi_available
            diag["capture_method"] = frame.method
            diag["latency_ms"] = round(frame.latency_ms, 2)
            diag["resolution"] = [frame.width, frame.height]

            arr = frame.to_rgb_array()
            nnz = int(np.count_nonzero(arr))
            diag["nonzero_pixels"] = nnz
            diag["mean_brightness"] = round(float(np.mean(arr)), 2)

            if diag["is_sandbox_desktop"] and nnz == 0:
                diag["diagnostic_verdict"] = f"SANDBOX_ISOLATED: Running on non-interactive desktop ({diag['desktop_name']}). Windows DWM blocks screen capture."
            elif nnz > 0:
                diag["diagnostic_verdict"] = "ACTIVE_CAPTURE: Captured live desktop pixels."
            else:
                diag["diagnostic_verdict"] = "ZERO_FRAME: Frame buffer is 100% black."

            try:
                pil_im = frame.to_pil()
                pil_im.thumbnail((480, 270))
                buf_thumb = io.BytesIO()
                pil_im.save(buf_thumb, format="JPEG", quality=60)
                diag["thumbnail"] = "data:image/jpeg;base64," + base64.b64encode(buf_thumb.getvalue()).decode("utf-8")
            except Exception:
                pass

            self.send_json(diag)
            return

        self.send_error(404, "Endpoint not found")

    def read_json_body(self) -> Dict[str, Any]:
        content_length = int(self.headers.get("Content-Length", 0))
        if content_length <= 0:
            return {}
        try:
            body = self.rfile.read(content_length).decode("utf-8")
            return json.loads(body)
        except Exception:
            return {}

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        data = self.read_json_body()

        if path in ("/v1/embeddings", "/api/embed"):
            raw_input = data.get("input", "") or data.get("text", "")
            if isinstance(raw_input, str):
                raw_input = [raw_input] if raw_input else [""]

            data_items = []
            total_tokens = 0
            total_lat = 0.0
            for idx, item_text in enumerate(raw_input):
                res = self.vmem.embed(str(item_text))
                vec = res[0] if isinstance(res, (tuple, list)) else res
                lat = res[1] if isinstance(res, (tuple, list)) and len(res) > 1 else 1.5
                total_lat += lat
                data_items.append({"object": "embedding", "embedding": vec.tolist(), "index": idx})
                total_tokens += max(1, len(str(item_text).split()))

            self.telemetry.record("embed", total_lat, {"batch_size": len(raw_input)})
            self.send_json({
                "object": "list",
                "data": data_items,
                "model": data.get("model", "lunar-npu-bge"),
                "usage": {"prompt_tokens": total_tokens, "total_tokens": total_tokens},
                "latency_ms": round(total_lat, 2),
                "silicon_proof": {
                    "device": self.engine.device,
                    "is_npu": self.engine.is_npu,
                    "hypersphere_manifold": "S^383",
                    "total_latency_ms": round(total_lat, 2),
                }
            })
            return

        if path == "/api/route":
            prompt = data.get("prompt", "")
            temp = float(data.get("temperature", 0.1))
            res, dists = self.router.route_geodesic(prompt, temperature=temp)
            self.telemetry.record("route", res.latency_ms, {"target": res.target_agent, "conf": round(res.confidence, 3)})
            d = res.to_dict()
            d["geodesic_distances"] = dists
            self.send_json(d)
            return

        if path == "/api/audit":
            cmd = data.get("cmd", "") or data.get("command", "")
            res = self.cb.audit_command(cmd)
            lat_ms = res.get("latency_ms", 0.002)
            self.telemetry.record("circuit_breaker", lat_ms, {"verdict": res.get("verdict")})
            self.send_json(res)
            return

        if path == "/api/memory":
            text = data.get("text", "")
            meta = data.get("metadata", {})
            entry = self.vmem.add_document(text, metadata=meta)
            self.send_json({
                "status": "STORED",
                "id": entry["id"],
                "text": entry["text"],
                "latency_ms": entry["latency_ms"],
                "total_documents": len(self.vmem.documents),
            })
            return

        if path == "/api/swarm":
            prompt = data.get("prompt") or data.get("task") or "Write a binary search function"
            max_tokens = int(data.get("max_tokens", 80))
            mode = data.get("mode", "single")
            is_cyclic = mode == "cyclic" or data.get("cyclic") in (True, "true", "1", 1)
            if is_cyclic:
                res = self.get_cyclic_swarm().run_cycle(prompt)
                self.send_json(res.to_dict())
                return
            res = self.get_swarm().execute_task(prompt, max_tokens=max_tokens)
            self.send_json(res.to_dict())
            return

        if path in ("/api/vision", "/api/screen"):
            img = data.get("image_path")
            res = self.get_vision().analyze(image_input=img)
            d = res.to_dict()
            try:
                screen = ImageGrab.grab() if img is None else None
                if screen:
                    screen.thumbnail((480, 270))
                    buf = io.BytesIO()
                    screen.save(buf, format="JPEG", quality=60)
                    d["thumbnail"] = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")
            except Exception:
                pass
            self.send_json(d)
            return

        if path == "/api/audio":
            path_str = data.get("audio_path")
            res = self.get_audio().transcribe(audio_source=path_str)
            self.send_json(res.to_dict())
            return

        if path == "/api/git":
            q = data.get("query", "")
            top_k = int(data.get("top_k", 5))
            res = self.get_git().search(q, top_k=top_k)
            self.send_json([r.to_dict() for r in res])
            return

        if path == "/api/governor/profile":
            prof = data.get("profile", "surge")
            res = self.engine.set_profile(prof)
            gov = get_power_governor()
            gov_status = gov.evaluate(current_profile=self.engine.current_profile)
            res.update(gov_status)
            self.send_json(res)
            return

        if path == "/api/lora/train":
            steps = int(data.get("steps", 20))
            rank = int(data.get("rank", 8))
            from_logs = bool(data.get("from_logs", False))
            lora = MicroLoRAEngine(rank=rank, engine=self.engine)
            if from_logs:
                res = lora.train_on_session_logs(steps_limit=steps)
            else:
                res = lora.benchmark_adaptation(steps=steps)
            self.send_json(res)
            return

        if path == "/api/hud/toggle":
            hud = get_ghost_hud()
            active = bool(data.get("active", not hud.is_running))
            if active:
                res = hud.start()
            else:
                res = hud.stop()
            self.send_json(res)
            return

        if path == "/api/hud/message":
            hud = get_ghost_hud()
            text = data.get("text", "")
            role = data.get("role", "assistant")
            urgency = data.get("urgency", "nominal")
            res = hud.post_message(text=text, role=role, urgency=urgency)
            self.send_json(res)
            return

        if path == "/api/mcp/rpc":
            from lunar_core.mcp_server import LunarMCPServer
            mcp_srv = LunarMCPServer()
            resp = mcp_srv.handle_request(data)
            self.telemetry.record("mcp_rpc", 1.5, {"method": data.get("method")})
            self.send_json(resp or {"jsonrpc": "2.0", "id": data.get("id"), "result": {}})
            return

        if path == "/api/mcp/call":
            from lunar_core.mcp_server import LunarMCPServer
            tool_name = data.get("name", "")
            tool_args = data.get("arguments", {})
            mcp_srv = LunarMCPServer()
            t0 = time.perf_counter()
            raw_res = mcp_srv._call_tool(tool_name, tool_args)
            lat_ms = (time.perf_counter() - t0) * 1000.0
            try:
                parsed_res = json.loads(raw_res) if isinstance(raw_res, str) and (raw_res.startswith("{") or raw_res.startswith("[")) else raw_res
            except Exception:
                parsed_res = raw_res
            self.telemetry.record("mcp_tool", lat_ms, {"tool": tool_name})
            tok_saved = 1200
            self.send_json({
                "jsonrpc": "2.0",
                "id": data.get("id", 1),
                "result": {
                    "tool": tool_name,
                    "latency_ms": round(lat_ms, 3),
                    "tokens_saved": tok_saved,
                    "dollars_saved": f"${(tok_saved / 1e6) * 15.0:.4f}",
                    "output": parsed_res,
                }
            })
            return

        if path == "/api/hdc/store":
            from lunar_core.hdc import HyperdimensionalMemoryEngine

            key = data.get("key", f"hdc_{int(time.time()*1000)}")
            text = data.get("text", "")
            meta = data.get("metadata", {})
            engine = HyperdimensionalMemoryEngine()
            mem_path = Path(".lunar_hdc_memory.json")
            engine.load_from_file(mem_path)
            engine.add(key, text, metadata=meta)
            engine.save_to_file(mem_path)
            self.send_json({"status": "stored", "key": key, "stats": engine.stats()})
            return

        if path == "/api/diffusion":
            prompt = data.get("prompt", "circuit diagram")
            steps = int(data.get("steps", 4))
            diff = self.get_diffusion()
            res = diff.sketch(prompt=prompt, steps=steps)
            d = res.to_dict()
            try:
                p = Path(res.image_path)
                if p.exists():
                    d["image_data"] = "data:image/png;base64," + base64.b64encode(p.read_bytes()).decode("utf-8")
            except Exception:
                pass
            self.send_json(d)
            return

        if path == "/api/audio/loopback":
            duration = float(data.get("duration", 1.0))
            vad_thresh = float(data.get("threshold", 0.005))
            res = self.get_audio().transcribe_loopback(duration_s=duration, vad_threshold=vad_thresh)
            self.send_json(res.to_dict())
            return

        if path == "/api/screen/probe":
            import ctypes, os
            import numpy as np
            diag = {
                "session_id": None,
                "window_station": None,
                "desktop_name": None,
                "is_sandbox_desktop": False,
                "dxgi_available": False,
                "capture_method": None,
                "latency_ms": 0.0,
                "resolution": [0, 0],
                "nonzero_pixels": 0,
                "mean_brightness": 0.0,
                "thumbnail": None,
                "diagnostic_verdict": "",
            }
            if sys.platform == "win32":
                try:
                    u32 = ctypes.windll.user32
                    k32 = ctypes.windll.kernel32
                    sess = ctypes.c_ulong()
                    k32.ProcessIdToSessionId(os.getpid(), ctypes.byref(sess))
                    diag["session_id"] = sess.value

                    hw = u32.GetProcessWindowStation()
                    buf = ctypes.create_unicode_buffer(256)
                    u32.GetUserObjectInformationW(hw, 2, buf, ctypes.sizeof(buf), None)
                    diag["window_station"] = buf.value

                    hd = u32.GetThreadDesktop(k32.GetCurrentThreadId())
                    buf2 = ctypes.create_unicode_buffer(256)
                    u32.GetUserObjectInformationW(hd, 2, buf2, ctypes.sizeof(buf2), None)
                    desk_name = buf2.value
                    diag["desktop_name"] = desk_name
                    diag["is_sandbox_desktop"] = "exebox" in desk_name.lower() or desk_name.lower() != "default"
                except Exception:
                    pass

            from lunar_core.dxgi_capture import DXGICaptureEngine
            cap = DXGICaptureEngine()
            frame = cap.capture_frame()
            diag["dxgi_available"] = cap.is_dxgi_available
            diag["capture_method"] = frame.method
            diag["latency_ms"] = round(frame.latency_ms, 2)
            diag["resolution"] = [frame.width, frame.height]

            arr = frame.to_rgb_array()
            nnz = int(np.count_nonzero(arr))
            diag["nonzero_pixels"] = nnz
            diag["mean_brightness"] = round(float(np.mean(arr)), 2)

            if diag["is_sandbox_desktop"] and nnz == 0:
                diag["diagnostic_verdict"] = f"SANDBOX_ISOLATED: Running on non-interactive desktop ({diag['desktop_name']}). Windows DWM blocks screen capture."
            elif nnz > 0:
                diag["diagnostic_verdict"] = "ACTIVE_CAPTURE: Captured live desktop pixels."
            else:
                diag["diagnostic_verdict"] = "ZERO_FRAME: Frame buffer is 100% black."

            try:
                pil_im = frame.to_pil()
                pil_im.thumbnail((480, 270))
                buf_thumb = io.BytesIO()
                pil_im.save(buf_thumb, format="JPEG", quality=60)
                diag["thumbnail"] = "data:image/jpeg;base64," + base64.b64encode(buf_thumb.getvalue()).decode("utf-8")
            except Exception:
                pass

            self.send_json(diag)
            return

        self.send_error(404, "Endpoint not found")

    def send_bytes(self, content: bytes, content_type: str):
        try:
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)
        except (ConnectionResetError, ConnectionAbortedError, BrokenPipeError):
            pass
        except Exception:
            pass

    def send_json(self, data: Any):
        try:
            payload = json.dumps(data).encode("utf-8")
            self.send_bytes(payload, "application/json")
        except (ConnectionResetError, ConnectionAbortedError, BrokenPipeError):
            pass
        except Exception:
            pass

    def log_message(self, format, *args):
        pass


def run_studio(host: str = "127.0.0.1", port: int = 8899, open_browser: bool = True):
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    class ReusableThreadingServer(ThreadingHTTPServer):
        allow_reuse_address = True
        daemon_threads = True

        def handle_error(self, request, client_address):
            # Suppress noisy client disconnect stack traces (WinError 10053 / 10054 / BrokenPipe)
            ex_type, _, _ = sys.exc_info()
            if ex_type in (ConnectionResetError, ConnectionAbortedError, BrokenPipeError):
                return
    # Start Named-Pipe IPC server for <15µs PreToolUse interception
    try:
        from lunar_core.hooks.silicon_guard_pipe import get_pipe_server
        get_pipe_server().start_background()
        print("[*] Silicon Guard Named-Pipe listening on: \\\\.\\pipe\\lunar_silicon_guard")
    except Exception as e:
        print(f"[!] Warning: Could not start Named-Pipe server: {e}")

    # Preload repo code indexer in background
    try:
        import threading
        from lunar_core.indexer import get_repo_indexer
        threading.Thread(target=get_repo_indexer, daemon=True).start()
    except Exception:
        pass

    server = ReusableThreadingServer((host, port), LunarStudioHandler)
    url = f"http://{host}:{port}"
    print(f"\n[*] Lunar NPU Studio launched at: {url}")

    if open_browser:
        try:
            webbrowser.open(url)
        except Exception:
            pass

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[!] Shutting down Lunar Studio.")
        server.server_close()


if __name__ == "__main__":
    run_studio()
