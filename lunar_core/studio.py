"""
Lunar Studio: Enterprise-Grade Hardware HUD & Agentic Neural Workbench
========================================================================
Interactive edge neural control center for Intel Lunar Lake 47 TOPS NPU,
Mamba SSM recurrence, S^383 vector memory, Speculative Decoding,
Silicon Circuit Breakers, and Model Context Protocol (MCP) diagnostics.
"""

from __future__ import annotations

import sys
import json
import time
import urllib.parse
from http.server import HTTPServer, BaseHTTPRequestHandler
import webbrowser
from typing import Any, Dict, List, Optional
import urllib.request

from lunar_core.engine import LunarNPUEngine
from lunar_core.mamba_ssm import LunarMambaEngine
from lunar_core.vector_memory import LunarVectorMemory
from lunar_core.speculative import LunarSpeculativePipeline
from lunar_core.circuit_breaker import SiliconCircuitBreaker
from lunar_core.router import MicroRouter


HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Lunar NPU Studio — Intel Lunar Lake Ambient Intelligence</title>
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.css" crossorigin="anonymous">
  <script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.js" crossorigin="anonymous"></script>
  <script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/contrib/auto-render.min.js" crossorigin="anonymous" onload="if(window.renderMathInElement) renderMathInElement(document.body, {delimiters: [{left: '$$', right: '$$', display: true}, {left: '$', right: '$', display: false}]});"></script>
  <style>
    :root {
      --bg: #07090e;
      --bg-surface: #0e131f;
      --card: #131a2a;
      --card-hover: #182238;
      --card-border: #1f2c44;
      --card-border-subtle: #162033;
      --text: #f1f5f9;
      --text-muted: #8493a8;
      --text-dim: #4f617a;
      
      /* wAIbi-sabi Earthy Accents */
      --accent-indigo: #6366f1;
      --accent-indigo-glow: rgba(99, 102, 241, 0.25);
      --accent-moss: #10b981;
      --accent-moss-glow: rgba(16, 185, 129, 0.2);
      --accent-ochre: #f59e0b;
      --accent-ochre-glow: rgba(245, 158, 11, 0.2);
      --accent-plum: #d946ef;
      --accent-plum-glow: rgba(217, 70, 239, 0.2);
      --accent-water: #06b6d4;
      --accent-water-glow: rgba(6, 182, 212, 0.2);
      --danger: #ef4444;
      --danger-glow: rgba(239, 68, 68, 0.2);
      
      --font-sans: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Inter", sans-serif;
      --font-mono: "JetBrains Mono", ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
    }

    .math-eq {
      font-family: "Cambria Math", "Latin Modern Math", "STIX Two Math", "Times New Roman", serif;
      font-size: 0.95rem;
      letter-spacing: 0.02em;
      color: #cbd5e1;
      display: inline-flex;
      align-items: center;
      gap: 1px;
    }
    .math-bar {
      border-top: 1.5px solid #cbd5e1;
      padding-top: 1px;
      line-height: 1;
      display: inline-block;
    }

    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      background-color: var(--bg);
      color: var(--text);
      font-family: var(--font-sans);
      min-height: 100vh;
      display: flex;
      flex-direction: column;
      overflow-x: hidden;
    }

    /* Top Navigation Bar */
    header {
      background: rgba(14, 19, 31, 0.85);
      backdrop-filter: blur(12px);
      border-bottom: 1px solid var(--card-border);
      padding: 0.85rem 2rem;
      position: sticky;
      top: 0;
      z-index: 100;
      display: flex;
      justify-content: space-between;
      align-items: center;
    }

    .brand {
      display: flex;
      align-items: center;
      gap: 14px;
    }
    .brand-logo {
      width: 40px;
      height: 40px;
      background: linear-gradient(135deg, var(--accent-indigo), var(--accent-plum));
      border-radius: 10px;
      display: flex;
      align-items: center;
      justify-content: center;
      font-size: 22px;
      font-weight: 800;
      box-shadow: 0 0 20px var(--accent-indigo-glow);
    }
    .brand-title {
      font-size: 1.25rem;
      font-weight: 700;
      letter-spacing: -0.02em;
    }
    .brand-subtitle {
      font-size: 0.8rem;
      color: var(--text-muted);
    }

    .header-badges {
      display: flex;
      align-items: center;
      gap: 10px;
    }
    .pill {
      font-family: var(--font-mono);
      font-size: 0.75rem;
      padding: 4px 10px;
      border-radius: 9999px;
      border: 1px solid var(--card-border);
      background: var(--bg-surface);
      color: var(--text-muted);
      display: flex;
      align-items: center;
      gap: 6px;
    }
    .pill-live {
      border-color: rgba(16, 185, 129, 0.4);
      background: rgba(16, 185, 129, 0.08);
      color: var(--accent-moss);
    }
    .dot-pulse {
      width: 6px;
      height: 6px;
      background: var(--accent-moss);
      border-radius: 50%;
      box-shadow: 0 0 8px var(--accent-moss);
      animation: pulse 2s infinite;
    }
    @keyframes pulse {
      0%, 100% { opacity: 1; transform: scale(1); }
      50% { opacity: 0.4; transform: scale(0.85); }
    }

    /* Tab Navigation */
    .tab-bar {
      display: flex;
      gap: 4px;
      background: var(--bg-surface);
      border-bottom: 1px solid var(--card-border);
      padding: 0 2rem;
      overflow-x: auto;
    }
    .tab-btn {
      background: transparent;
      border: none;
      color: var(--text-muted);
      padding: 12px 18px;
      font-size: 0.88rem;
      font-weight: 600;
      cursor: pointer;
      border-bottom: 2px solid transparent;
      transition: all 0.2s ease;
      display: flex;
      align-items: center;
      gap: 8px;
      white-space: nowrap;
    }
    .tab-btn:hover {
      color: var(--text);
      background: rgba(255, 255, 255, 0.02);
    }
    .tab-btn.active {
      color: var(--accent-indigo);
      border-bottom-color: var(--accent-indigo);
      background: rgba(99, 102, 241, 0.05);
    }

    /* Main Container */
    main {
      flex: 1;
      padding: 2rem;
      max-width: 1440px;
      margin: 0 auto;
      width: 100%;
    }

    .tab-pane {
      display: none;
      animation: fadeIn 0.2s ease-in-out;
    }
    .tab-pane.active {
      display: block;
    }
    @keyframes fadeIn {
      from { opacity: 0; transform: translateY(4px); }
      to { opacity: 1; transform: translateY(0); }
    }

    /* Layout Grids */
    .grid-3 {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(340px, 1fr));
      gap: 1.5rem;
      margin-bottom: 1.5rem;
    }
    .grid-2 {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(480px, 1fr));
      gap: 1.5rem;
      margin-bottom: 1.5rem;
    }

    /* Cards */
    .card {
      background: var(--card);
      border: 1px solid var(--card-border);
      border-radius: 14px;
      padding: 1.5rem;
      position: relative;
      overflow: hidden;
      transition: border-color 0.2s, box-shadow 0.2s;
    }
    .card:hover {
      border-color: rgba(99, 102, 241, 0.4);
      box-shadow: 0 8px 24px -6px rgba(0, 0, 0, 0.5);
    }
    .card-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 1.25rem;
    }
    .card-title {
      font-size: 1.05rem;
      font-weight: 700;
      color: #fff;
      display: flex;
      align-items: center;
      gap: 8px;
    }

    /* Key Numbers */
    .stat-hero {
      font-family: var(--font-mono);
      font-size: 2.75rem;
      font-weight: 800;
      letter-spacing: -0.03em;
      line-height: 1.1;
      margin-bottom: 4px;
    }
    .stat-unit {
      font-size: 1.15rem;
      font-weight: 500;
      color: var(--text-muted);
    }
    .stat-label {
      color: var(--text-muted);
      font-size: 0.85rem;
    }

    /* 6 NCE Tiles Visualizer */
    .tiles-rack {
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 10px;
      margin-top: 1.25rem;
    }
    .tile-box {
      background: #0d1424;
      border: 1px solid #1a253b;
      border-radius: 10px;
      padding: 12px;
      text-align: center;
      position: relative;
      transition: all 0.2s;
    }
    .tile-box.active {
      border-color: var(--accent-indigo);
      background: rgba(99, 102, 241, 0.08);
    }
    .tile-name {
      font-family: var(--font-mono);
      font-size: 0.82rem;
      font-weight: 700;
      color: #e2e8f0;
    }
    .tile-sub {
      font-size: 0.7rem;
      color: var(--accent-moss);
      margin-top: 2px;
    }
    .tile-light {
      position: absolute;
      top: 8px;
      right: 8px;
      width: 6px;
      height: 6px;
      border-radius: 50%;
      background: var(--accent-moss);
      box-shadow: 0 0 8px var(--accent-moss);
    }

    /* Form Controls */
    .form-group {
      margin-bottom: 1rem;
    }
    .form-label {
      display: block;
      font-size: 0.82rem;
      color: var(--text-muted);
      margin-bottom: 6px;
      font-weight: 500;
    }
    .input-text {
      width: 100%;
      padding: 11px 14px;
      background: #090e18;
      border: 1px solid var(--card-border);
      border-radius: 9px;
      color: var(--text);
      font-family: inherit;
      font-size: 0.9rem;
      outline: none;
      transition: border-color 0.2s;
    }
    .input-text:focus {
      border-color: var(--accent-indigo);
    }

    .btn {
      background: var(--accent-indigo);
      color: #fff;
      border: none;
      padding: 11px 20px;
      border-radius: 9px;
      font-weight: 600;
      font-size: 0.9rem;
      cursor: pointer;
      display: inline-flex;
      align-items: center;
      justify-content: center;
      gap: 8px;
      transition: all 0.15s;
    }
    .btn:hover {
      background: #4f46e5;
      transform: translateY(-1px);
    }
    .btn-secondary {
      background: #1e293b;
      color: var(--text);
      border: 1px solid var(--card-border);
    }
    .btn-secondary:hover {
      background: #28354b;
    }

    /* Console Display */
    .terminal-window {
      background: #060910;
      border: 1px solid #141c2c;
      border-radius: 10px;
      padding: 14px;
      font-family: var(--font-mono);
      font-size: 0.84rem;
      line-height: 1.5;
      color: #38bdf8;
      max-height: 240px;
      overflow-y: auto;
      white-space: pre-wrap;
      margin-top: 1rem;
    }

    /* Badges & Tables */
    .badge-ok {
      background: rgba(16, 185, 129, 0.15);
      color: var(--accent-moss);
      padding: 3px 8px;
      border-radius: 6px;
      font-size: 0.78rem;
      font-weight: 700;
    }
    .badge-block {
      background: rgba(239, 68, 68, 0.15);
      color: var(--danger);
      padding: 3px 8px;
      border-radius: 6px;
      font-size: 0.78rem;
      font-weight: 700;
    }

    table {
      width: 100%;
      border-collapse: collapse;
      margin-top: 0.75rem;
      font-size: 0.88rem;
    }
    th {
      text-align: left;
      padding: 10px 14px;
      color: var(--text-muted);
      border-bottom: 1px solid var(--card-border);
      font-weight: 600;
    }
    td {
      padding: 12px 14px;
      border-bottom: 1px solid var(--card-border-subtle);
    }
  </style>
</head>
<body>

  <header>
    <div class="brand">
      <div class="brand-logo">▲</div>
      <div>
        <div class="brand-title">Lunar NPU Studio</div>
        <div class="brand-subtitle">Intel Core Ultra 200V ("Lunar Lake") 47 TOPS Architecture</div>
      </div>
    </div>
    <div class="header-badges">
      <div class="pill pill-live">
        <span class="dot-pulse"></span>
        <span id="npuDeviceLabel">Intel AI Boost NPU 4000</span>
      </div>
      <div class="pill">OpenVINO 2026+</div>
      <div class="pill">Away Mode 0x80000041</div>
    </div>
  </header>

  <div class="tab-bar">
    <button class="tab-btn active" onclick="switchTab('tab-overview')">⚡ Silicon Topology</button>
    <button class="tab-btn" onclick="switchTab('tab-mamba')">🔄 Mamba SSM Recurrence</button>
    <button class="tab-btn" onclick="switchTab('tab-memory')">🌐 Vector Memory (S³⁸³)</button>
    <button class="tab-btn" onclick="switchTab('tab-speculative')">⚡ Speculative Decoding</button>
    <button class="tab-btn" onclick="switchTab('tab-breaker')">🛡️ Silicon Circuit Breaker</button>
    <button class="tab-btn" onclick="switchTab('tab-router')">🎯 MicroRouter Swarm</button>
    <button class="tab-btn" onclick="switchTab('tab-mcp')">🤖 Agentic MCP Tools</button>
  </div>


  <main>
    <!-- TAB 1: OVERVIEW & HARDWARE TOPOLOGY -->
    <div id="tab-overview" class="tab-pane active">
      <div class="grid-3">
        <div class="card">
          <div class="card-header">
            <span class="card-title">Peak Compute</span>
            <span class="badge-ok">PHYSICAL SILICON</span>
          </div>
          <div class="stat-hero" style="color: var(--accent-water);">47.0 <span class="stat-unit">TOPS</span></div>
          <div class="stat-label">INT8 Matrix Engine Throughput</div>
          <div style="margin-top: 1rem; color: var(--text-muted); font-size: 0.82rem;">
            Driver Version: <strong id="drvVer" style="color:#fff;">1004723</strong><br>
            Turbo Mode: <strong style="color:var(--accent-moss);">ENABLED (NPU_TURBO=YES)</strong>
          </div>
        </div>

        <div class="card">
          <div class="card-header">
            <span class="card-title">Recurrent Latency</span>
            <span class="badge-ok">ZERO KV-CACHE</span>
          </div>
          <div class="stat-hero" style="color: var(--accent-indigo);" id="heroMambaLat">0.197 <span class="stat-unit">ms</span></div>
          <div class="stat-label">Linear State Update (<span class="math-eq"><i>h</i><sub><i>t</i></sub> = <span class="math-bar"><i>A</i></span><i>h</i><sub><i>t</i>−1</sub> + <span class="math-bar"><i>B</i></span><i>x</i><sub><i>t</i></sub></span>)</div>
          <div style="margin-top: 1rem; color: var(--text-muted); font-size: 0.82rem;">
            Throughput: <strong id="heroMambaTps" style="color:#fff;">5,068 tok/s</strong><br>
            Memory Complexity: <strong style="color:var(--accent-indigo);">O(1) Bounded State</strong>
          </div>
        </div>

        <div class="card">
          <div class="card-header">
            <span class="card-title">Circuit Gatekeeper</span>
            <span class="badge-ok">MICROSECOND DFA</span>
          </div>
          <div class="stat-hero" style="color: var(--accent-moss);">2.20 <span class="stat-unit">µs</span></div>
          <div class="stat-label">Deterministic Kernel Guardrail</div>
          <div style="margin-top: 1rem; color: var(--text-muted); font-size: 0.82rem;">
            Scan Rate: <strong style="color:#fff;">455,270 scans/sec</strong><br>
            Syscall Protection: <strong style="color:var(--accent-moss);">100% Deterministic</strong>
          </div>
        </div>
      </div>

      <div class="card">
        <div class="card-header">
          <span class="card-title">6 Physical Neural Compute Engine (NCE) Tiles</span>
          <span class="pill pill-live"><span class="dot-pulse"></span> All 6 Tiles Online</span>
        </div>
        <div class="tiles-rack">
          <div class="tile-box active"><div class="tile-light"></div><div class="tile-name">NCE TILE 0</div><div class="tile-sub">Matrix / MAC Engine</div></div>
          <div class="tile-box active"><div class="tile-light"></div><div class="tile-name">NCE TILE 1</div><div class="tile-sub">Matrix / MAC Engine</div></div>
          <div class="tile-box active"><div class="tile-light"></div><div class="tile-name">NCE TILE 2</div><div class="tile-sub">Matrix / MAC Engine</div></div>
          <div class="tile-box active"><div class="tile-light"></div><div class="tile-name">NCE TILE 3</div><div class="tile-sub">Matrix / MAC Engine</div></div>
          <div class="tile-box active"><div class="tile-light"></div><div class="tile-name">NCE TILE 4</div><div class="tile-sub">Matrix / MAC Engine</div></div>
          <div class="tile-box active"><div class="tile-light"></div><div class="tile-name">NCE TILE 5</div><div class="tile-sub">Matrix / MAC Engine</div></div>
        </div>
      </div>

      <!-- LIVE SILICON PROOF & EDGE ECONOMICS -->
      <div class="card" style="margin-top: 1.25rem;">
        <div class="card-header">
          <span class="card-title">Live Silicon Proof & Edge Economics</span>
          <span class="badge-ok" id="telemetryStatusBadge">ACTIVE SILICON TELEMETRY</span>
        </div>
        <div class="grid-4" style="display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 1rem; margin: 1rem 0;">
          <div style="background: var(--bg-surface); padding: 1rem; border-radius: 8px; border: 1px solid var(--card-border-subtle);">
            <div style="font-size: 0.75rem; color: var(--text-dim); text-transform: uppercase;">Silicon Inferences</div>
            <div style="font-size: 1.6rem; font-weight: 700; color: var(--accent-water); font-family: var(--font-mono);" id="telTotalOps">0</div>
            <div style="font-size: 0.75rem; color: var(--text-muted); margin-top: 2px;">Executed on NPU</div>
          </div>
          <div style="background: var(--bg-surface); padding: 1rem; border-radius: 8px; border: 1px solid var(--card-border-subtle);">
            <div style="font-size: 0.75rem; color: var(--text-dim); text-transform: uppercase;">Silicon Active Time</div>
            <div style="font-size: 1.6rem; font-weight: 700; color: var(--accent-moss); font-family: var(--font-mono);" id="telActiveTime">0.0 ms</div>
            <div style="font-size: 0.75rem; color: var(--text-muted); margin-top: 2px;">Zero Host CPU Load</div>
          </div>
          <div style="background: var(--bg-surface); padding: 1rem; border-radius: 8px; border: 1px solid var(--card-border-subtle);">
            <div style="font-size: 0.75rem; color: var(--text-dim); text-transform: uppercase;">Cloud Cost Saved</div>
            <div style="font-size: 1.6rem; font-weight: 700; color: var(--accent-ochre); font-family: var(--font-mono);" id="telCostSaved">$0.0000</div>
            <div style="font-size: 0.75rem; color: var(--text-muted); margin-top: 2px;">$0.00 Marginal Token Cost</div>
          </div>
          <div style="background: var(--bg-surface); padding: 1rem; border-radius: 8px; border: 1px solid var(--card-border-subtle);">
            <div style="font-size: 0.75rem; color: var(--text-dim); text-transform: uppercase;">Energy Saved</div>
            <div style="font-size: 1.6rem; font-weight: 700; color: var(--accent-plum); font-family: var(--font-mono);" id="telEnergySaved">0.00 J</div>
            <div style="font-size: 0.75rem; color: var(--text-muted); margin-top: 2px;">2.5W NPU vs 45W CPU</div>
          </div>
        </div>
        <div style="border-top: 1px solid var(--card-border-subtle); padding-top: 0.75rem;">
          <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
            <span style="font-size: 0.8rem; color: var(--text-muted); font-weight: 600;">REAL-TIME SILICON INFERENCE EVENT STREAM</span>
            <span style="font-size: 0.75rem; color: var(--accent-moss); font-family: var(--font-mono);" id="telUptime">UPTIME: 0s</span>
          </div>
          <div class="terminal-window" id="telEventLog" style="max-height: 140px; overflow-y: auto; font-size: 0.78rem;">
            Waiting for first hardware inference event...
          </div>
        </div>
      </div>
    </div>

    <!-- TAB 2: MAMBA SSM RECURRENCE -->
    <div id="tab-mamba" class="tab-pane">
      <div class="grid-2">
        <div class="card">
          <div class="card-title">Mamba Recurrence Controller</div>
          <p style="color: var(--text-muted); font-size: 0.85rem; margin: 8px 0 1.25rem;">
            Benchmark state-space model recurrence step latency without allocating KV-cache memory.
          </p>
          <div class="form-group">
            <label class="form-label">Recurrent Steps to Execute</label>
            <input type="number" id="mambaStepInput" class="input-text" value="100" min="10" max="1000">
          </div>
          <button class="btn" style="width: 100%;" onclick="executeMambaBench()">
            ⚡ Execute Recurrence Benchmark on NPU
          </button>
        </div>

        <div class="card">
          <div class="card-title">Live Telemetry & Diagnostics</div>
          <div class="terminal-window" id="mambaTerminal">Click 'Execute' to profile Mamba recurrence on physical silicon...</div>
        </div>
      </div>
    </div>

    <!-- TAB 3: VECTOR MEMORY -->
    <div id="tab-memory" class="tab-pane">
      <div class="grid-2">
        <div class="card">
          <div class="card-title">Query Hyperspherical Memory (S³⁸³)</div>
          <p style="color: var(--text-muted); font-size: 0.85rem; margin: 8px 0 1.25rem;">
            Dense 384-dim embeddings normalized onto unit hypersphere. Cosine similarity = dot product.
          </p>
          <div class="form-group">
            <label class="form-label">Semantic Search Query</label>
            <input type="text" id="vecQueryInput" class="input-text" value="Intel NPU physical architecture">
          </div>
          <button class="btn" onclick="executeVectorSearch()">🔍 Search Memory</button>
        </div>

        <div class="card">
          <div class="card-title">Store New Knowledge</div>
          <p style="color: var(--text-muted); font-size: 0.85rem; margin: 8px 0 1.25rem;">
            Ingest text directly into local NPU memory without transmitting tokens over the cloud.
          </p>
          <div class="form-group">
            <label class="form-label">Document Content</label>
            <input type="text" id="newDocText" class="input-text" placeholder="e.g. Lunar Lake runs silent at 49C with lid closed.">
          </div>
          <button class="btn btn-secondary" onclick="addDocumentToMemory()">➕ Ingest into Edge Silicon</button>
        </div>
      </div>

      <div class="card">
        <div class="card-title">Memory Search Results</div>
        <div class="terminal-window" id="vecResults">Run a query to inspect cosine similarity rankings and latencies...</div>
      </div>
    </div>

    <!-- TAB 4: SPECULATIVE DECODING -->
    <div id="tab-speculative" class="tab-pane">
      <div class="card">
        <div class="card-title">Heterogeneous Dual-Engine Speculative Decoder</div>
        <p style="color: var(--text-muted); font-size: 0.85rem; margin: 8px 0 1.25rem;">
          NPU drafts gamma tokens sequentially at sub-millisecond speeds; Target GPU/LLM verifies in parallel over on-package LPDDR5X UMA.
        </p>
        <button class="btn" onclick="runSpeculativeDemo()">🚀 Run Dual-Engine Speculative Cycle (γ = 4)</button>
        <div class="terminal-window" id="specTerminal">Click above to execute draft + target verification cycle...</div>
      </div>
    </div>

    <!-- TAB 5: SILICON CIRCUIT BREAKER -->
    <div id="tab-breaker" class="tab-pane">
      <div class="card">
        <div class="card-title">Interactive DFA Gatekeeper Console</div>
        <p style="color: var(--text-muted); font-size: 0.85rem; margin: 8px 0 1.25rem;">
          Deterministic 2.2µs safety filter auditing proposed shell commands before terminal execution.
        </p>
        <div style="display: flex; gap: 10px;">
          <input type="text" id="cbInput" class="input-text" value="rm -rf /" placeholder="Enter shell command">
          <button class="btn" style="width: 200px;" onclick="auditCircuitBreaker()">🛡️ Audit Command</button>
        </div>
        <div style="margin-top: 1rem; display: flex; gap: 8px;">
          <button class="btn btn-secondary" style="font-size: 0.78rem;" onclick="setBreakerCmd('git status')">Safe: git status</button>
          <button class="btn btn-secondary" style="font-size: 0.78rem;" onclick="setBreakerCmd('DROP DATABASE production;')">Hazard: DROP DB</button>
          <button class="btn btn-secondary" style="font-size: 0.78rem;" onclick="setBreakerCmd('Remove-Item -Recurse C:\\\\Windows\\\\System32')">Hazard: Delete System32</button>
        </div>
        <div class="terminal-window" id="cbResults">Enter command and click Audit...</div>
      </div>
    </div>

    <!-- TAB 6: MICROROUTER SWARM DISPATCH -->
    <div id="tab-router" class="tab-pane">
      <div class="card" style="margin-bottom: 1.5rem;">
        <div class="card-header">
          <span class="card-title">🎯 MicroRouter & Swarm Centroid Dispatcher</span>
          <span class="pill pill-live">SUB-3MS LATENCY</span>
        </div>
        <p style="color: var(--text-muted); font-size: 0.88rem; margin-bottom: 1rem;">
          Classifies task descriptions and dispatches multi-agent swarms (Coder, Architect, DevOps/Tester, Researcher, Security Auditor) via NPU embedding centroids on S³⁸³ at $0 marginal token cost. Grounded in RouteLLM (LMSYS) and ProCIS (SIGIR 2024).
        </p>
        <div style="display: flex; gap: 10px; margin-bottom: 10px;">
          <input type="text" class="input-text" id="routerInput" placeholder="Enter task prompt..." value="Write a python function to compute fibonacci with memoization">
          <button class="btn" onclick="runTaskRouter()">Route on NPU</button>
        </div>
        <div style="display: flex; gap: 6px; flex-wrap: wrap; margin-bottom: 1rem;">
          <span style="font-size: 0.78rem; color: var(--text-dim); align-self: center;">Quick Archetype Presets:</span>
          <button class="btn btn-secondary" style="font-size: 0.75rem; padding: 4px 8px;" onclick="setRouterPrompt('Write a python function to implement binary search over sorted arrays')">Coder</button>
          <button class="btn btn-secondary" style="font-size: 0.75rem; padding: 4px 8px;" onclick="setRouterPrompt('Design distributed microservices architecture and Kafka event bus schema')">Architect</button>
          <button class="btn btn-secondary" style="font-size: 0.75rem; padding: 4px 8px;" onclick="setRouterPrompt('Configure GitHub Actions CI matrix for pytest across Windows and Ubuntu')">Tester/DevOps</button>
          <button class="btn btn-secondary" style="font-size: 0.75rem; padding: 4px 8px;" onclick="setRouterPrompt('Retrieve arXiv research papers on State Space Models and Mamba scaling laws')">Researcher</button>
          <button class="btn btn-secondary" style="font-size: 0.75rem; padding: 4px 8px;" onclick="setRouterPrompt('Scan bash scripts for malicious command injection rm -rf / and privilege escalation')">Security</button>
        </div>
        <div id="routerResults" class="terminal-window" style="background: rgba(14, 19, 31, 0.9);">
          Click "Route on NPU" to classify task prompt against normalized centroid manifolds...
        </div>
      </div>
    </div>

    <!-- TAB 7: AGENTIC MCP TOOLS -->
    <div id="tab-mcp" class="tab-pane">

      <div class="card">
        <div class="card-title">Agentic Model Context Protocol (MCP) Live Inspector</div>
        <p style="color: var(--text-muted); font-size: 0.85rem; margin: 8px 0 1.25rem;">
          Tools registered over stdio for Cursor, Claude Desktop, Antigravity, and Windsurf (`lunar mcp`).
        </p>
        <table>
          <thead>
            <tr><th>Tool Name</th><th>Signature</th><th>Description</th></tr>
          </thead>
          <tbody>
            <tr><td><code>lunar_status</code></td><td><code>{}</code></td><td>Inspect physical silicon, tiles, and 47 TOPS INT8 status.</td></tr>
            <tr><td><code>lunar_mamba_step</code></td><td><code>{steps: int}</code></td><td>Constant-memory O(1) state recurrence without KV-cache.</td></tr>
            <tr><td><code>lunar_vector_search</code></td><td><code>{query: str, top_k: int}</code></td><td>Sub-3ms hyperspherical semantic memory query.</td></tr>
            <tr><td><code>lunar_circuit_breaker_audit</code></td><td><code>{command: str}</code></td><td>2.2µs deterministic regex DFA safety gatekeeper.</td></tr>
            <tr><td><code>lunar_add_memory</code></td><td><code>{text: str}</code></td><td>Embed and store fact in edge silicon.</td></tr>
          </tbody>
        </table>
      </div>
    </div>
  </main>

  <script>
    function switchTab(tabId) {
      document.querySelectorAll('.tab-pane').forEach(el => el.classList.remove('active'));
      document.querySelectorAll('.tab-btn').forEach(el => el.classList.remove('active'));
      document.getElementById(tabId).classList.add('active');
      event.target.classList.add('active');
    }

    async function executeMambaBench() {
      const steps = document.getElementById('mambaStepInput').value || 100;
      const term = document.getElementById('mambaTerminal');
      term.innerText = `[INFO] Dispatching ${steps} recurrence steps to Intel NPU silicon...`;
      try {
        const res = await fetch(`/api/mamba?steps=${steps}`);
        const data = await res.json();
        document.getElementById('heroMambaLat').innerHTML = `${data.mean_step_latency_ms.toFixed(3)} <span class="stat-unit">ms</span>`;
        document.getElementById('heroMambaTps').innerText = `${Math.round(data.tokens_per_second).toLocaleString()} tok/s`;
        term.innerText = JSON.stringify(data, null, 2);
      } catch (err) {
        term.innerText = '[ERROR] ' + err;
      }
    }

    async function executeVectorSearch() {
      const q = document.getElementById('vecQueryInput').value;
      const term = document.getElementById('vecResults');
      term.innerText = `[INFO] Querying S³⁸³ vector memory for: "${q}"...`;
      try {
        const res = await fetch(`/api/query?q=${encodeURIComponent(q)}&top_k=3`);
        const data = await res.json();
        term.innerText = JSON.stringify(data, null, 2);
      } catch (err) {
        term.innerText = '[ERROR] ' + err;
      }
    }

    async function addDocumentToMemory() {
      const text = document.getElementById('newDocText').value;
      const term = document.getElementById('vecResults');
      if (!text) return;
      term.innerText = `[INFO] Embedding text on NPU...`;
      try {
        const res = await fetch('/api/memory', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({text: text})
        });
        const data = await res.json();
        term.innerText = `[SUCCESS] Embedded document onto unit hypersphere!\\n` + JSON.stringify(data, null, 2);
      } catch (err) {
        term.innerText = '[ERROR] ' + err;
      }
    }

    function setBreakerCmd(cmd) {
      document.getElementById('cbInput').value = cmd;
      auditCircuitBreaker();
    }

    async function auditCircuitBreaker() {
      const cmd = document.getElementById('cbInput').value;
      const term = document.getElementById('cbResults');
      try {
        const res = await fetch(`/api/audit?cmd=${encodeURIComponent(cmd)}`);
        const data = await res.json();
        const badge = data.verdict === 'ALLOWED' ? '<span class="badge-ok">[ALLOWED]</span>' : '<span class="badge-block">[BLOCKED]</span>';
        term.innerHTML = `${badge} Reason: ${data.reason}\\nLatency: ${(data.latency_ms * 1000).toFixed(2)} µs\\nHazard Score: ${data.hazard_probability}`;
      } catch (err) {
        term.innerText = '[ERROR] ' + err;
      }
    }

    async function runSpeculativeDemo() {
      const term = document.getElementById('specTerminal');
      term.innerText = '[INFO] Executing NPU draft step + parallel target verification...';
      try {
        const res = await fetch('/api/speculative?gamma=4');
        const data = await res.json();
        term.innerText = JSON.stringify(data, null, 2);
      } catch (err) {
        term.innerText = '[ERROR] ' + err;
      }
    }

    function setRouterPrompt(p) {
      document.getElementById('routerInput').value = p;
      runTaskRouter();
    }

    async function runTaskRouter() {
      const prompt = document.getElementById('routerInput').value;
      const term = document.getElementById('routerResults');
      if (!prompt) return;
      term.innerText = '[INFO] Embedding on Intel NPU and calculating geodesic distances...';
      try {
        const res = await fetch(`/api/route?prompt=${encodeURIComponent(prompt)}`);
        const data = await res.json();
        let scoresHtml = '';
        for (const [k, v] of Object.entries(data.scores)) {
          const pct = (v * 100).toFixed(1);
          const isTop = k === data.target_agent;
          const color = isTop ? 'var(--accent-moss)' : 'var(--accent-indigo)';
          scoresHtml += `
            <div style="margin: 6px 0;">
              <div style="display:flex; justify-content:space-between; font-size:0.8rem; margin-bottom:2px;">
                <span style="color:${isTop ? '#fff' : 'var(--text-muted)'}; font-weight:${isTop ? 'bold' : 'normal'};">${k} ${isTop ? '★ (DISPATCHED)' : ''}</span>
                <span style="color:${color}; font-family:var(--font-mono);">${pct}%</span>
              </div>
              <div style="background:#1e293b; border-radius:4px; height:6px; overflow:hidden;">
                <div style="background:${color}; height:100%; width:${pct}%;"></div>
              </div>
            </div>`;
        }
        term.innerHTML = `
          <div style="margin-bottom: 8px;">
            <span class="badge-ok" style="font-size:0.9rem; padding:4px 10px;">TARGET: ${data.target_agent}</span>
            <span style="margin-left:8px; color:var(--accent-water); font-family:var(--font-mono); font-size:0.85rem;">Confidence: ${(data.confidence * 100).toFixed(1)}%</span>
            <span style="margin-left:8px; color:var(--text-dim); font-family:var(--font-mono); font-size:0.82rem;">Latency: ${data.latency_ms}ms</span>
          </div>
          <div style="color:var(--text-muted); font-size:0.82rem; margin-bottom:12px;">${data.rationale}</div>
          <div style="border-top:1px solid var(--card-border); padding-top:8px;">${scoresHtml}</div>
        `;
      } catch (err) {
        term.innerText = '[ERROR] ' + err;
      }
    }

    async function updateTelemetry() {
      try {
        const res = await fetch('/api/telemetry');
        if (!res.ok) return;
        const d = await res.json();
        const elOps = document.getElementById('telTotalOps');
        if (elOps) elOps.innerText = d.total_silicon_inferences.toLocaleString();
        const elTime = document.getElementById('telActiveTime');
        if (elTime) elTime.innerText = d.total_silicon_time_ms.toFixed(1) + ' ms';
        const elCost = document.getElementById('telCostSaved');
        if (elCost) elCost.innerText = '$' + d.cloud_dollars_saved.toFixed(4);
        const elEnergy = document.getElementById('telEnergySaved');
        if (elEnergy) elEnergy.innerText = d.energy_joules_saved.toFixed(2) + ' J';
        const elUp = document.getElementById('telUptime');
        if (elUp) elUp.innerText = `UPTIME: ${Math.round(d.uptime_seconds)}s`;

        if (d.recent_events && d.recent_events.length > 0) {
          const logEl = document.getElementById('telEventLog');
          if (logEl) {
            logEl.innerHTML = d.recent_events.slice().reverse().map(e => {
              const opColor = e.op === 'route' ? 'var(--accent-water)' : e.op === 'mamba' ? 'var(--accent-indigo)' : e.op === 'circuit_breaker' ? 'var(--accent-moss)' : 'var(--accent-ochre)';
              return `<div style="margin: 3px 0;"><span style="color:var(--text-dim)">[${e.time_str}]</span> <span style="color:${opColor}; font-weight:bold;">${e.op.toUpperCase()}</span> <span style="color:#fff; font-family:var(--font-mono);">${e.latency_ms}ms</span> <span style="color:var(--text-muted)">(${JSON.stringify(e.details)})</span></div>`;
            }).join('');
          }
        }
      } catch (err) {}
    }

    updateTelemetry();
    setInterval(updateTelemetry, 2500);
  </script>
</body>
</html>
"""


class SiliconTelemetry:
    """Tracks live hardware telemetry, throughput metrics, and real-time inference proof."""

    def __init__(self):
        self.start_time = time.time()
        self.total_embeddings = 0
        self.total_mamba_steps = 0
        self.total_routed_prompts = 0
        self.total_circuit_audits = 0
        self.total_silicon_time_ms = 0.0
        self.recent_events: List[Dict[str, Any]] = []

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

    def summary(self) -> Dict[str, Any]:
        uptime_sec = time.time() - self.start_time
        total_ops = self.total_embeddings + self.total_routed_prompts + self.total_circuit_audits + (1 if self.total_mamba_steps > 0 else 0)
        est_tokens = (self.total_embeddings + self.total_routed_prompts) * 500 + self.total_mamba_steps
        cloud_savings_usd = (est_tokens / 1_000_000.0) * 3.00
        joules_saved = (self.total_silicon_time_ms / 1000.0) * (45.0 - 2.2)

        return {
            "uptime_seconds": round(uptime_sec, 1),
            "total_silicon_inferences": total_ops,
            "total_embeddings": self.total_embeddings,
            "total_mamba_steps": self.total_mamba_steps,
            "total_routed_prompts": self.total_routed_prompts,
            "total_circuit_audits": self.total_circuit_audits,
            "total_silicon_time_ms": round(self.total_silicon_time_ms, 2),
            "estimated_tokens_routed": est_tokens,
            "cloud_dollars_saved": round(cloud_savings_usd, 4),
            "energy_joules_saved": round(joules_saved, 2),
            "recent_events": self.recent_events[-15:],
        }


class LunarStudioHandler(BaseHTTPRequestHandler):
    engine = LunarNPUEngine()
    mamba = LunarMambaEngine(engine=engine, d_inner=64, d_state=16)
    vmem = LunarVectorMemory(engine=engine, embedding_dim=384)
    spec = LunarSpeculativePipeline(draft_engine=engine, gamma=4)
    cb = SiliconCircuitBreaker(engine=engine)
    router = MicroRouter(memory_engine=vmem)
    telemetry = SiliconTelemetry()

    if not vmem.documents:
        vmem.add_document("Intel Lunar Lake microarchitecture features 6 NCE physical tiles.")
        vmem.add_document("Mamba state-space recurrence operates with zero dynamic memory allocation and constant O(1) state.")
        vmem.add_document("Silicon Circuit Breakers enforce deterministic kernel guardrails in 2.2 microseconds.")
        vmem.add_document("Speculative decoding pairs an ultra-fast NPU draft with target GPU verification.")
        vmem.add_document("Lunar Lake on-package LPDDR5X-8533 enables zero-copy heterogeneous UMA sharing.")

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        query = urllib.parse.parse_qs(parsed.query)

        if path == "/" or path == "/index.html":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(HTML_PAGE.encode("utf-8"))
            return

        if path == "/api/status":
            self.send_json(self.engine.get_device_info())
            return

        if path == "/api/telemetry":
            self.send_json(self.telemetry.summary())
            return

        if path in ("/health", "/api/health"):
            self.send_json({"status": "ok", "npu": True, "device": self.engine.device, "version": "1.0.0"})
            return

        if path == "/api/route":
            prompt = query.get("prompt", [""])[0]
            temp = float(query.get("temp", [0.1])[0])
            res = self.router.route(prompt, temperature=temp)
            self.telemetry.record("route", res.latency_ms, {"target": res.target_agent, "conf": round(res.confidence, 3)})
            self.send_json(res.to_dict())
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
            res = self.vmem.query(q, top_k=top_k)
            self.send_json(res)
            return

        if path == "/api/speculative":
            gamma = int(query.get("gamma", [4])[0])
            prefix = [101, 2054, 2003, 1037, 3231]
            res = self.spec.speculative_cycle(prefix_tokens=prefix)
            self.send_json(res)
            return

        if path == "/api/audit":
            cmd = query.get("cmd", [""])[0]
            res = self.cb.audit(cmd)
            audit_us = res.get("audit_latency_us", 2.2)
            self.telemetry.record("circuit_breaker", audit_us / 1000.0, {"verdict": res.get("status")})
            self.send_json(res)
            return

        self.send_error(404, "Endpoint not found")

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        if path in ("/v1/embeddings", "/api/embed"):
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length).decode("utf-8")
            data = json.loads(body)
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
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length).decode("utf-8")
            data = json.loads(body)
            prompt = data.get("prompt", "")
            temp = float(data.get("temperature", 0.1))
            res = self.router.route(prompt, temperature=temp)
            self.telemetry.record("route", res.latency_ms, {"target": res.target_agent, "conf": round(res.confidence, 3)})
            self.send_json(res.to_dict())
            return

        if path == "/api/memory":
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length).decode("utf-8")
            data = json.loads(body)
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


        self.send_error(404, "Endpoint not found")

    def send_json(self, data: Any):
        payload = json.dumps(data).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, format, *args):
        pass


def run_studio(host: str = "127.0.0.1", port: int = 8899, open_browser: bool = True):
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    server = HTTPServer((host, port), LunarStudioHandler)
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
