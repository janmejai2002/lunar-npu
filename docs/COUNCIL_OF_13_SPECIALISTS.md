# THE 13-SPECIALIST AUTONOMOUS COUNCIL SPECIFICATION
## Roster, Domain Ownership, and Operational Protocols for Perpetual Agent & NPU Execution

```
====================================================================================================
SYSTEM IDENTIFIER:      AGY-PERPETUAL-SPECIALISTS-v1.0
PURPOSE:                Autonomous Multi-Agent Domain Division for 100-Page Architecture & Execution
PERSISTENCE LAYER:      SQLite WAL + Local Artifacts Engine + Git Epoch Ledger
HARDWARE ACCELERATION:  Intel Lunar Lake NPU 4000 (47 TOPS INT8) + OpenVINO Level Zero
====================================================================================================
```

---

### GROUP 1: AMBIENT SILICON & EDGE AI RESEARCH GUILD
#### 1. Frontier AI Research Fellow
- **Domain**: Algorithmic frontiers, hybrid SLM-LLM distillation, self-speculative decoding, and low-bit quantization (FP4/INT4/BitNet).
- **Core Mission**: Research optimal balance between cloud reasoning (Gemini Pro) and edge execution (Lunar Lake NPU).

#### 2. Neuro-Symbolic Ambient OS Specialist
- **Domain**: Deterministic symbolic logic rules combined with neural heuristic models.
- **Core Mission**: Maintain deterministic DAG execution graphs, safety invariants, and semantic intent parsers.

#### 3. Hardware Security & Cryptography Specialist
- **Domain**: Zero-trust credential handling, local DPAPI vault isolation, PKCS#11 hardware tokens, and offline encryption.
- **Core Mission**: Ensure 100% loopback isolation (`127.0.0.1`), zero token exfiltration, and tamper-evident audit logging.

#### 4. Edge Biosignal & Health Specialist
- **Domain**: Real-time biosignal processing, on-device audio telemetry, ambient sensor feeds, and low-latency feature extraction.
- **Core Mission**: Ultra-low-power (<1.5W) continuous ambient classification running directly on Intel AI Boost NPU.

#### 5. Sensory Translation & Neural Audio Specialist
- **Domain**: Whisper INT8 / OpenVINO neural speech recognition, real-time spatial audio synthesis, and streaming transcription.
- **Core Mission**: Sub-25ms zero-cloud audio transcription pipelines with zero CPU/battery penalty.

#### 6. Speculative NPU Architect
- **Domain**: Intel NPU 4000 ISA, MCDM kernel compilation, Level Zero driver fences, and shared LPDDR5x memory management.
- **Core Mission**: Maximize NPU throughput (47 TOPS), compile zero-latency blob caches, and prevent Level Zero driver timeouts.

---

### GROUP 2: PERPETUAL AGENT RUNTIME & RELIABILITY GUILD
#### 7. Autonomous DevOps Reliability Lead
- **Domain**: Windows 11 Modern Standby (S0ix), Win32 P/Invoke `SetThreadExecutionState`, Away Mode, and process watchdog supervisor.
- **Core Mission**: Guarantee 99.999% uptime over 72+ hour unattended runs without system sleep or thermal shutdown.

#### 8. Multi-Account Quota & Token Orchestrator
- **Domain**: 3x Google Gemini Pro account token pooling, sliding-window rate limiters, circuit breakers, and RTK compression hooks.
- **Core Mission**: Manage 1,080 RPM combined ceiling, enforce 85% safe saturation buffers, and eliminate HTTP 429 throttling.

#### 9. Continuous Hardware & Test Rig Lead
- **Domain**: Automated CI/CD test harness, hardware-in-the-loop (HIL) verification, regression benchmarking, and atomic git rollbacks.
- **Core Mission**: Validate unit tests, compile-check models, and ensure zero broken states ever persist to git.

#### 10. Predictive Context Researcher
- **Domain**: Attention window optimization, context fragmentation minimization, and semantic chunking via local NPU embeddings.
- **Core Mission**: Prevent context bloat, prune redundant AST tokens, and preserve long-term episodic memory.

#### 11. Silicon Guardrail Specialist
- **Domain**: Local deterministic content policy, schema validation, output JSON repair, and hallucination containment.
- **Core Mission**: Intercept malformed tool calls or runaway agent loops before cloud API execution occurs.

#### 12. Session Gateway Engineer
- **Domain**: Local bridge routing (`127.0.0.1:8045`), port conflict resolution, connection pooling, and HTTP keep-alive pipelines.
- **Core Mission**: Seamless IPC between agent processes, Antigravity Tools, and local NPU microservices.

#### 13. Telemetry & Notification Architect
- **Domain**: Real-time Web HUD (`ui/index.html`), WMI thermal/battery metrics, SQLite WAL audit event streams, and alerts.
- **Core Mission**: Present high-signal, zero-noise visibility to the developer whenever they open their laptop screen.
