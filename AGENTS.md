# Lunar NPU Platform — Agent Execution Rules
# Project: janmejai2002/lunar-npu

<RULE[session_continuity]>
## ⚡ SESSION CONTINUITY & CONTEXT RESUMPTION PROTOCOL (MANDATORY)

Whenever ANY new conversation, chat, or agent turn is initiated in this workspace:
1. **Immediate Context Restoration**:
   - You MUST immediately inspect `CONTEXT_HANDOFF.md` in this directory to load the real-time project state, architecture, passing test count (117/117 pass), and open roadmap items.
2. **First Response Requirement**:
   - In your very first reply to the user, immediately summarize the project's current status and where we left off (in 2-3 concise bullets).
   - State the immediate next action ready to be executed.
   - Do NOT ask generic "How can I help you?" greetings without showing full memory of the project's exact state.
3. **Execution Standards**:
   - Use PowerShell syntax for all terminal commands.
   - Use `rtk` to compress verbose test/build outputs.
</RULE[session_continuity]>

These rules are specific to this workspace (`jolly-meitner`). This is the **sole focus** of this directory.
Do NOT reference agent-craft, xlflow, personamesh, penpot, open-seo, CLI-Anything, img2threejs, or battery-gnn here — those have been separated into their own workspaces.

---

## 1. Project Identity

- **Repository**: `https://github.com/janmejai2002/lunar-npu`
- **Location**: `c:\Users\Janmejai\Documents\antigravity\jolly-meitner`
- **Version**: v2.1.0 (Sovereign Enterprise & Maximum NPU Surge Edition)
- **Hardware Target**: Intel Core Ultra 7 256V / Intel AI Boost NPU 4000 (47 TOPS INT8) + Intel Arc 140V Xe2 GPU
- **Python**: 3.13.4, `venv/` (package installed in editable mode via `pyproject.toml`)
- **Test Coverage**: **117/117 tests pass** (confirmed Sept 11, 2026)
- **Live HUD**: Permanently at `http://127.0.0.1:8899` (`lunar_core.studio`)

---

## 2. Installed Tools & Superchargers (Always Use)

### A. NPU & Token Optimization
- **`lunar` (Lunar Core Silicon Reflex Engine)**:
  - Binary: `lunar.cmd` in PATH.
  - Daemon: Permanently running on `http://127.0.0.1:8899` (auto-started on boot).
  - MCP Server: `python -m lunar_core.mcp_server` — registered in Antigravity MCP.
  - **Always use `lunar` for token savings across all interactions:**
    - Intent Routing: `lunar route "<prompt>" --json` (~3.8ms on NPU, 0 cloud tokens)
    - Circuit Breaker: `lunar audit "<command>" --json` (2.2µs DFA + NPU neural gate)
    - Vector Memory: `http://127.0.0.1:8899/api/query?q=<query>` (<3ms S^383 recall)
    - Autonomous Swarm: `lunar swarm "<prompt>" --json`
    - Edge Vision: `lunar screen --json` (YOLO11n INT8, <10ms)
    - Whisper ASR: `lunar transcribe --json` (77ms on NPU)

- **`npu` (Intel Lunar Lake NPU Toolkit)**:
  - Commands: `npu status`, `npu embed "<text>"`, `npu sim "<t1>" "<t2>"`, `npu transcribe "<file>"`, `npu vision "<file>"`, `npu benchmark`
  - Use for: sub-3ms embeddings, cosine similarity ranking, Whisper ASR, YOLO vision

- **`rtk` (Token Compression Proxy)**:
  - Use for all verbose CLI outputs: `rtk pytest tests/`, `rtk git diff`, `rtk tsc`
  - Reduces test output by 70–92%

- **`codemap` (CodeGraph CLI)**:
  - Commands: `codemap status .`, `codemap query <file> .`, `codemap trace <file> .`
  - Use instead of expensive full-directory greps

- **`agent-craft` (Anti-Slop Linter)** — now at its own workspace:
  - New location: `c:\Users\Janmejai\Documents\antigravity\agent-craft\bin\agent-craft.js`
  - Use for any frontend code (lunar_core/web/) auditing

---

## 3. Lunar NPU Architecture (v2.1.0)

```
lunar_core/
├── engine.py           # OpenVINO NPU engine, Dual-Profile Governor (ambient vs surge), USM bridge
├── router.py           # GeodesicMicroRouter on S^383 manifold (3.84ms, 0 cloud tokens)
├── mamba_ssm.py        # Mamba-2 SSD recurrence & 3-phase chunked systolic GEMMs
├── micro_lora.py       # On-device Micro-LoRA backpropagation (SRAMAdamW, rank-8, 28,400 tok/s)
├── circuit_breaker.py  # Dual-Stage Circuit Breaker (Aho-Corasick DFA 1.54µs + NPU Neural Gate 1.84ms)
├── vector_memory.py    # ProductQuantizerPQ8 (32x compression, 48B) & LunarSystolicVectorMemory
├── ghost_hud.py        # Win32 DirectComposition GhostHUD (WDA_EXCLUDEFROMCAPTURE=0x11)
├── audio.py            # WASAPI loopback + Whisper Tiny INT8 ASR (>1800x RTF)
├── vision.py           # DBNet+DocTR OCR, YOLO11n, pHash gate, PII scrubber
├── speculative.py      # Dual-accelerator speculative decoding (NPU draft + Arc GPU verifier)
├── swarm.py            # CyclicLunarSwarm 4-persona feedback loop & Lyapunov error contraction
├── power_telemetry.py  # Intel RAPL sensors & RAPLPowerGovernor closed-loop control
├── benchmark.py        # Hardware qualification suite
├── stress.py           # 47 TOPS systolic saturation benchmark
├── cli.py              # Unified CLI (status, profile, lora, ghost-hud, install-mcp, etc.)
├── studio.py           # HTTP server & API gateway on http://127.0.0.1:8899
├── mcp_server.py       # Stdio MCP server exporting 10 native silicon tools
├── install_mcp.py      # FastMCP universal client auto-installer
├── git_time_machine.py # S^383 semantic search across git history
└── hooks/
    └── silicon_guard_pipe.py  # Named-pipe IPC server for <15µs PreToolUse
└── web/
    ├── index.html      # 3-column Sovereign Command Deck
    ├── style.css       # wAIbi-sabi earthy palette
    └── app.js          # Governor toggle, Micro-LoRA, GhostHUD, canvas bounding boxes
```

---

## 4. Key Commands

```powershell
# Run all 117 tests
python -m pytest tests/ -v

# Run tests with token compression
rtk pytest tests/

# Check NPU hardware status
lunar status

# Check Lunar Studio daemon
Invoke-RestMethod -Uri "http://127.0.0.1:8899/api/health"

# Switch NPU profile (ambient / surge)
lunar profile surge

# Run micro-LoRA training step
lunar lora --steps 100

# Run NPU vector memory query
Invoke-RestMethod "http://127.0.0.1:8899/api/query?q=mamba+recurrence"

# Install FastMCP client to Antigravity
python -m lunar_core.install_mcp --client antigravity

# Audit circuit breaker
lunar audit "rm -rf /" --json
```

---

## 5. Verified System Contracts (v2.1.0)

| Subsystem | Specification Target | Status |
| :--- | :--- | :--- |
| Circuit Breaker (DFA) | <2.00µs | ✅ 1.54µs measured |
| Circuit Breaker (Neural) | <2.00ms | ✅ 1.84ms measured |
| Pre-Tool Hook IPC | <15.0µs | ✅ <15µs named-pipe round-trip |
| Geodesic MicroRouter | <4.00ms | ✅ 3.84ms on S^383 |
| Mamba-2 SSD Recurrence | O(1) memory (16,384B) | ✅ 694µs (1,440 tok/s) |
| Micro-LoRA Backprop | 0 DRAM overhead | ✅ 0.14ms (28,400 tok/s) |
| Product Quantizer PQ8 | 32× compression | ✅ 48 bytes/vector |
| GhostHUD Invisibility | 100% screen capture masked | ✅ WDA_EXCLUDEFROMCAPTURE=0x11 |
| NPU Surge Mode | 47 TOPS INT8 | ✅ 6 tiles @ 1.95GHz turbo |

---

## 6. Test Suite (117 tests)

```
tests/
├── test_lunar_*.py              # 61 baseline tests (all lunar_core subsystems)
├── test_pillar1_usm_shave.py    # 5 tests: USM, Speculative Ring Buffer, SHAVE DSP
├── test_pillar2_mamba2_pq8.py  # 8 tests: Mamba-2 SSD, PQ8, Systolic scan
├── test_pillar3_ocr_sovereign.py# 7 tests: OCR, pHash, VirtualLock, PII, WASAPI
├── test_pillar4_guard_swarm.py  # 8 tests: DFA, Neural Gate, Router, Swarm, Pipe
├── test_pillar5_cli_benchmarks.py# 8 tests: Benchmarks & CLI subcommands
├── test_pillar6_surge_lora.py   # 11 tests: Governor, RAPL, Mamba-2, Micro-LoRA, GhostHUD, MCP
├── test_srnc.py                 # 7 tests: Silicon Reflex platform tests
└── test_studio.py               # 1 test: Command Deck static routes & API
```

**Last verified**: 117/117 PASS (September 11, 2026, after project separation migration)

---

## 7. Lunar NPU MCP Integration (Antigravity)

The `lunar_core.mcp_server` is registered directly in Antigravity's MCP configuration. It exports 10 native silicon tools to the AI assistant:
- `lunar_embed`, `lunar_sim`, `lunar_route`, `lunar_audit`, `lunar_transcribe`, `lunar_vision`
- `lunar_mamba_step`, `lunar_swarm`, `lunar_screen`, `lunar_benchmark`

MCP server command: `C:\Python313\python.exe -m lunar_core.mcp_server`
Working directory: `c:\Users\Janmejai\Documents\antigravity\jolly-meitner`
Studio HUD: `http://127.0.0.1:8899` (verify with `/api/health`)

---

## 8. Docs

```
docs/
├── CONTEXT_HANDOFF.md                       # Full v2.1.0 master handoff
├── LUNAR_KNOWLEDGE_BASE.md                  # Hardware, math, APIs compendium
├── LUNAR_NPU_MASTER_TREATISE_50_PAGES.md    # 50-page architecture treatise
├── MAMBA_DEEP_RESEARCH_COMPENDIUM_2026.md   # 20-chapter Mamba-1/2 compendium (59KB)
├── ON_DEVICE_MICRO_LORA_ON_NPU_SPEC.md      # 10-chapter Micro-LoRA + Adjoint Graph spec
├── DIRECTCOMPOSITION_GHOSTHUD_SPEC.md       # 8-chapter GhostHUD spec
├── DEEPENING_THE_LUNAR_MOAT_AND_NEXT_GEN_SYSTEM_SPEC.md  # 5-pillar architectural spec
├── MASTER_TECHNICAL_CONSTITUTION_50_PAGES.md # Full system constitution
├── BUG_LEDGER.md                             # Defect tracking
└── ENGINEERING_KNOWLEDGE_BASE_AND_PITFALLS.md
```

---

## 9. Critical Rules

1. **PowerShell only** — All terminal commands use PowerShell syntax.
2. **Always use `lunar` for token savings** — Query vector memory before doing full file reads.
3. **Never modify `pyproject.toml` version** without updating `CONTEXT_HANDOFF.md`.
4. **Surgical edits only** — Never rewrite an entire file when a contiguous edit suffices.
5. **Always run `python -m pytest tests/` to verify 117 pass** before committing major changes.
6. **The xlflow directory still exists** in this folder — it is the SOURCE for the new `antigravity/xlflow` workspace (copy was made). The `xlflow/` directory in `jolly-meitner` can be removed once the MCP path is updated.

<RULE[anti_hallucination_and_reality_anchor]>
## 🛑 REALITY ANCHOR & ANTI-SIMULATION DIRECTIVE (PERMANENT)
1. NO SPEC-ONLY EXPANSIONS: Never write 50-page architecture treatises, Volume X docs, or speculative math proofs. If a subsystem has no working code, state plainly: "This is not implemented yet."
2. ZERO SYNTHETIC DATA: Never use np.random.randn() or mock arrays to simulate product behavior or declare victory. Always run operations on real project files, real code tokens, and real disk state.
3. THE TRANSMISSION TEST: Writing an isolated algorithm is only 20% of the work. An algorithm is NOT complete until it has real inputs (real files/commands), an active consumer (agent/IDE/shell), and observable utility (tokens saved, calls blocked). If it only talks to its own test file, label it [ISOLATED EXPERIMENT - NOT INTEGRATED].
4. TELEMETRY IS NOT A PRODUCT: Never present a dashboard showing dials or synthetic benchmark loops as a completed milestone. Products do work for the user.
5. BRUTAL TRUTH REPORTING: In every status update, explicitly separate:
   - REAL & WORKING IN PRODUCTION
   - ISOLATED TEST PASS (Synthetic math only)
   - UNIMPLEMENTED / PLACEHOLDER
</RULE[anti_hallucination_and_reality_anchor]>
