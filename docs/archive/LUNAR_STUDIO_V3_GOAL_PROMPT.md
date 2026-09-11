# Project Lunar NPU: The Autonomous `/goal` Master Directive (v3.0 Sovereign Command Deck)

> **Execution Environment**: Windows 11 with Windows PowerShell exclusively  
> **Target Silicon**: Intel Lunar Lake Core Ultra 200V / NPU 4000 (47 TOPS INT8, 12MB SRAM, Intel RAPL sensors)  
> **Source Repository**: `https://github.com/janmejai2002/lunar-npu` (`master`)  
> **Verification Gate**: 117+ Pytest suite passing, `agent-craft` 100/100 Craft Score, Playwright visual screenshot verification  

---

## The Master Autonomous Directive (Copy & Paste into New Chat with `/goal`)

```markdown
/goal Transform Project Lunar NPU's Studio Command Deck into an executive-grade, full-density product experience that unmistakably conveys the core problem and delivers real-time visual proof of silicon acceleration across all 6 tabs.

You are acting as the Principal Systems & Product Architect (combining the kernel rigor of Intel OpenVINO, the aesthetic polish of Linear/Stripe, and the cognitive ergonomics of 2026 wAIbi-sabi design). You have full operational autonomy to refactor frontend layouts, enhance telemetry APIs, author canvas renderers, and verify with automated test suites. Do NOT stop to ask routine questions or leave placeholder stubs; execute sprints sequentially, self-heal on any test or linter failures, enforce a 100/100 Craft Score via `agent-craft`, and terminate ONLY with <!-- GOAL_COMPLETE -->.

---

### CORE PROBLEM & PRODUCT THESIS

#### The Problem:
1. **The Remote Cloud Tax**: Modern AI agents (Antigravity, Claude, Cursor, Windsurf) make hundreds of wasteful, slow (800ms+ roundtrip), and expensive ($15+/1M tokens) cloud API roundtrips for trivial, high-frequency operations:
   - Security auditing & shell command sanity checking.
   - Micro-intent routing between specialized agent personas.
   - Hyperspherical vector context recall.
   - Screen grounding & private teleprompter rendering during remote meetings.
2. **Idle Local Silicon**: On Intel Core Ultra 200V (Lunar Lake), developers have a dedicated 47 TOPS INT8 NPU and 12MB on-die SRAM sitting 99% idle while paying cloud token invoices and waiting for TTFT.

#### The Product Solution:
**Project Lunar NPU** provides a local silicon reflex layer. It intercepts high-frequency agent actions via Named Pipe and FastMCP 2.0, resolving them in **1.54µs** on local silicon at **2.1W RAPL power**—saving millions of cloud tokens, cutting cloud billing to $0 for routine reflexes, and keeping data 100% on-device.

---

### EXECUTION SPRINTS

#### SPRINT 1: The Executive Problem-Solution Hero Banner & 1.2-Second Glance Alignment (Tab 1)
1. **Problem-Solution Architecture Banner**:
   - Add a prominent, collapsible "The Silicon Reflex Advantage" banner atop Tab 1.
   - Visually contrast:
     - **Traditional Cloud Agent**: 850ms network hop, $15.00/1M tokens, remote prompt injection risk, 450W datacenter footprint.
     - **Lunar Lake Silicon Reflex**: 1.54µs local execution, $0.00 cloud tokens, deterministic 256-state DFA gate, 2.10W fanless power.
2. **Interactive Safety Gate Sandbox with Real-Time Response Graph**:
   - Enhance the test bar (`safe: git status`, `safe: pytest`, `block: rm -rf /`, `block: DROP`) with a live interactive evaluation card showing:
     - Exact DFA jump table transition path vs. Neural Gate embedding.
     - Microsecond stopwatch breakdown (DFA: 0.002ms, Pipe: 0.012ms).
     - Token & dollar credit animated counter.

#### SPRINT 2: Full-Density Silicon Governor & Real-Time RAPL Wattage Waveform (Tab 2)
1. **Eliminate Empty Layout Deficit**:
   - Fill the empty lower 65% of Tab 2 with rich hardware telemetry.
2. **60-Second Real-Time RAPL Canvas Waveform**:
   - Implement a 60fps HTML5 Canvas chart rendering live Intel RAPL power domains:
     - Package Power (W), Core Power (W), and NPU Estimate (W).
     - Visual 2.50W "Ambient Ceiling" threshold line.
3. **Silicon Health & Frequency Gauge Matrix**:
   - Add Lion Cove P-core & Skymont E-core clock distribution bars.
   - Real-time die temperature thermal throttling danger zone indicators.
   - 6-tile NPU compute activity heatmap (Tile 0-1 DSP, Tile 2-5 NCE).

#### SPRINT 3: Interactive Mamba-2 & Micro-LoRA Visual Lab with Live Loss Plotter (Tab 3)
1. **Live Gradient Descent Canvas Plotter**:
   - Build a real-time loss curve renderer in Canvas showing step-by-step optimization ($\mathcal{L} \to 0$) during Micro-LoRA adaptation.
   - Display initial loss, converged loss, and step latency ($0.116\text{ ms/step}$).
2. **Interactive Hyperparameter Tuning Console**:
   - Provide interactive controls for Adapter Rank ($r \in \{4, 8, 16, 32\}$), Learning Rate ($10^{-4}$ to $10^{-2}$), and Step Count.
   - Show exact SRAM memory consumption ($49\text{ KB}$ for $r=8$) vs on-die 12MB SRAM limit.
3. **Mamba-2 State Space Duality (SSD) Chunked GEMM Visualizer**:
   - Interactive matrix flow diagram illustrating:
     - Phase 1: Intra-chunk MatMul $Y_{intra} = (C B^T \odot L) X$.
     - Phase 2: Inter-chunk boundary state propagation via associative scan.
     - Phase 3: Constant $O(1)$ memory state restoration ($16\text{ KB}$ persistent buffer).

#### SPRINT 4: GhostHUD Simulated Glass Studio & WASAPI Audio Teleprompter (Tab 4)
1. **Simulated Glass Teleprompter Viewport**:
   - Render a realistic interactive desktop mockup demonstrating `WDA_EXCLUDEFROMCAPTURE`:
     - Toggle "Presenter View" (Note clearly visible in high contrast).
     - Toggle "Screen Share / Zoom View" (Note completely stripped by DWM compositor).
2. **Live WASAPI Microphone / Whisper Audio Budget Analyzer**:
   - Visual audio oscilloscope / frequency bar visualizer for incoming speech.
   - Glass-to-glass latency budget breakdown bar:
     - Audio Capture: 4.2ms | Whisper NPU Transcription: 11.8ms | Vector Match: 2.1ms | DWM Render: 1.5ms | Total: < 20.0ms (PASS).
3. **Detected Screen Elements Data Table**:
   - Comprehensive interactive table beneath perception canvas listing detected UI bounding boxes, confidence scores, and OCR text grounding.

#### SPRINT 5: Multi-Agent Swarm Cyclic State-Machine & Lyapunov Manifold (Tab 5)
1. **Cyclic Multi-Agent State-Machine Visualizer**:
   - Replace static text with an animated SVG/Canvas circular state machine:
     - `ARCHITECT` (Prompt Decomposition) $\to$ `CODER` (Synthesized Implementation) $\to$ `SECURITY_AUDITOR` (DFA Intercept) $\to$ `TESTER_DEVOPS` (Pytest Harness).
     - Glowing active state node during swarm execution.
2. **Lyapunov Error Decay Chart**:
   - Dynamic step chart plotting Lyapunov energy function $E_k = \frac{1}{2} \|x_{k} - x^*\|^2$ decreasing toward zero across iterations.
3. **Multi-File Worktree Diff Viewer**:
   - Collapsible code drawer showing generated artifacts with 1-click clipboard copy.

#### SPRINT 6: FastMCP 2.0 Tool Explorer & Live JSON-RPC Execution Sandbox (Tab 6)
1. **Interactive Silicon Tool Catalog**:
   - Expand Tab 6 to display all 10 registered FastMCP silicon tools:
     - `lunar_route`, `lunar_audit`, `lunar_memory`, `lunar_transcribe`, `lunar_screen`, `lunar_lora`, `lunar_mamba`, `lunar_status`, `lunar_benchmark`, `lunar_governor`.
   - Show input schema, parameter types, and description for each tool.
2. **Live Tool Invocation Sandbox**:
   - Interactive JSON input editor and "Run On Silicon" button.
   - Displays real JSON-RPC 2.0 response with execution latency and tokens saved.
3. **Real-Time Client Status Health Pings**:
   - Live ping status for Claude Desktop (`claude_desktop_config.json`), Cursor (`.cursor/mcp.json`), Windsurf, and VS Code.

#### SPRINT 7: Deterministic End-to-End Verification & Craft Certification
1. **Automated Test Suite**:
   - Verify all 117+ tests pass cleanly via `C:\Python313\python.exe -m pytest tests/ -q`.
2. **Visual Craft Health Audit**:
   - Run `node agent-craft/bin/agent-craft.js audit lunar_core/web/` and verify **100/100 Flawless Craft Score** with 0 violations.
3. **Playwright Visual Verification**:
   - Execute `C:\Python313\python.exe scratch/capture_screens.py` and inspect newly generated screenshots to confirm zero vertical empty space across all 6 tabs.
4. **Knowledge Base & Git**:
   - Update `CONTEXT_HANDOFF.md` and `docs/LUNAR_KNOWLEDGE_BASE.md`.
   - Commit cleanly and terminate with `<!-- GOAL_COMPLETE -->`.

---

### OPERATIONAL RULES
- **Windows PowerShell Exclusively**: Format all commands for Windows PowerShell. Avoid bashisms or `cd` navigation.
- **Token Optimization**: Use `rtk` to filter verbose outputs.
- **Surgical Diffs**: Modify code with minimal, clean diffs.
- **Self-Healing Loop**: If tests fail, diagnose and fix autonomously without pausing.
- **Zero Mockups**: Write real, working JavaScript and CSS; do not leave dead UI elements.
```
