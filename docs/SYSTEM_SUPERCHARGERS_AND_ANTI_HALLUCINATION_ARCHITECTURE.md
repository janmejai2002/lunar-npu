# System Superchargers, Token Optimization & Anti-Hallucination Architecture

This document provides a comprehensive operational inventory of all installed tools, hardware accelerators, proxy multiplexers, token compression mechanisms, vulnerability analyses, and the deterministic anti-hallucination protocols governing autonomous perpetual execution for the **Lunar Project**.

---

## 1. Installed Tools & Superchargers Inventory

The agent environment is equipped with a multi-layered optimization and execution stack operating across hardware, system binaries, proxy services, and progressive skills.

### A. Code Intelligence & Structural Graph Navigation
| Tool | Binary & Path | Version / Engine | Operational Role & Token Savings |
| :--- | :--- | :--- | :--- |
| **`codemap`** | `C:\Users\Janmejai\AppData\Roaming\npm\codemap.ps1` | v0.1.0 (Bun Runtime) | Pre-indexed AST Code Knowledge Graph. Provides Control Flow Graphs (`cfg`), Data Flow Graphs (`dfg`), Program Dependence Graphs (`pdg`), Program Slicing (`slice`), and Git Co-Change Coupling (`coupling`). **Replaces expensive multi-megabyte grep queries with targeted structural queries (saving 80–95% tokens on code navigation).** |
| **`graphify`** | `graphify_memory/` | SQLite + Semantic Graph | Persistent symbol relationship graph mapping dependencies, module contracts, and call hierarchy across turns. |

### B. Token Compression & Output Optimization
| Tool | Binary & Path | Version / Engine | Operational Role & Token Savings |
| :--- | :--- | :--- | :--- |
| **`rtk`** | `C:\Users\Janmejai\AppData\Roaming\npm\rtk.exe` | v0.48.0 (Native Rust) | Ultra-high performance CLI proxy. Intercepts verbose outputs from `git diff`, `pytest`, `tsc`, `cargo`, `pip`, `npm`, `pnpm`, and `ruff`, removing boilerplates and condensing diffs/errors into compact summaries. **Cuts context consumption by 60–90%.** |
| **`context-mode`** | MCP Server (`mcp_config.json`) | Sandboxed FTS5 Index | Sandboxes noisy tool outputs and intermediate execution traces, indexing project files in SQLite FTS5. **Reduces context injection overhead by up to 98%.** |

### C. Hardware-Accelerated Local Inference (Zero Cloud Tokens)
| Tool | Binary & Path | Hardware Target | Capabilities & Latency |
| :--- | :--- | :--- | :--- |
| **`npu`** | `C:\Users\Janmejai\AppData\Roaming\npm\npu.cmd` | **Intel Lunar Lake AI Boost NPU 4000** (47 TOPS INT8) | Local vectorization and embedding using OpenVINO (`minilm_l6`), cosine similarity ranking, Whisper speech-to-text (>2,500x RTF), and computer vision (1,260+ FPS). **Vector searches and semantic similarity checks run locally in <3ms with ZERO cloud API token consumption.** |
| **`llmfit`** | `C:\Users\Janmejai\AppData\Roaming\npm\llmfit.exe` | v1.1.14 | Hardware profiler analyzing CPU, RAM, and GPU VRAM to match local LLMs to memory budgets. |

### D. Agent Orchestration & Federation Infrastructure
| Tool | Location & Endpoint | Operational Role | Capacity / State |
| :--- | :--- | :--- | :--- |
| **`antigravity-perpetual`** | PID 32704 (`http://127.0.0.1:8765`) | Autonomous supervisor holding Win32 Away Mode (`0x80000041`), thermal monitor, silence deadlock watchdog, and SQLite ledger (`runtime_state.db`). | Multi-day execution with laptop lid closed; thermals regulated at ~50°C. |
| **`cli-anything-antigravity-tools`** | PID 26376 (`http://127.0.0.1:8045`) | Agent-native CLI harness interfacing with Antigravity Tools desktop app. Manages local OpenAI/Anthropic proxies, account switching, and model discovery. | **4x Google AI Pro federation** (1,440 RPM, 16M TPM, 120k RPD), 94 models available on `/v1`. |
| **`antigravity-skills-manager`** | `.agents/skills/antigravity-skills-manager` | Progressive disclosure engine for 300+ specialized agent skills (AAS Core). | Prevents prompt bloat by lazy-loading skill instructions on demand rather than loading all into the system prompt. |

---

## 2. Vulnerability & Weakness Analysis (Where Agents Fail in Long Horizons)

Long-running autonomous execution across hundreds of turns creates specific failure vectors that must be actively countered:

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                      LONG-RUNNING AGENT VULNERABILITY MATRIX                    │
├───────────────────────────────┬─────────────────────────────────────────────────┤
│ Failure Mode                  │ Underlying Mechanism                            │
├───────────────────────────────┼─────────────────────────────────────────────────┤
│ 1. Attention Sink Dilution    │ Massive context lengths dilute attention on     │
│    ("Needle-in-a-Haystack")   │ foundational constraints; hallucinated variables│
├───────────────────────────────┼─────────────────────────────────────────────────┤
│ 2. Context Drift & Ghost APIs │ Remembering older, modified versions of files;  │
│                               │ hallucinating deleted methods or phantom imports│
├───────────────────────────────┼─────────────────────────────────────────────────┤
│ 3. Tool Loop Cascades         │ Repeating minor syntax fixes blindly in a loop, │
│                               │ consuming thousands of tokens without progress  │
├───────────────────────────────┼─────────────────────────────────────────────────┤
│ 4. Behavioral Discontinuity   │ "Goldfish effect" from blind compaction where   │
│    ("Compaction Amnesia")     │ agent loses identity, scope, or active roadmap  │
└───────────────────────────────┴─────────────────────────────────────────────────┘
```

### Detailed Failure Modes:
1. **Context Window Saturation & Sunk Attention**:
   - When a conversation transcript expands past 60,000–100,000 tokens, the LLM's softmax attention becomes diffused across historical steps. The agent begins to ignore negative constraints and hallucinates file contents that it remembers seeing 40 turns earlier.
2. **Ghost References & State Desynchronization**:
   - If a file is refactored, the agent's historical memory still contains the *old* signature. Without deterministic verification, the agent writes new code calling deprecated or deleted functions.
3. **Tool Retry Spiral**:
   - A subagent runs a command that fails, reads the error, modifies one line incorrectly, runs it again, and repeats 10 times, burning quota and context.
4. **Context Compaction Discontinuity**:
   - Checkpoint summarization often compresses away nuanced architectural decisions or operational constraints, turning the agent into an unfocused generalist.

---

## 3. GitHub & SOTA Research Solutions

To address these vulnerabilities, we draw from the latest 2025–2026 breakthroughs in agentic engineering:

### A. Hierarchical Memory & Structured Distillation (Mem0, AgeMem, LangMem)
- **Concept**: Treat memory not as raw conversation history, but as an active, structured data store with CRUD operations (Store, Retrieve, Update, Discard).
- **GitHub Reference**: `mem0ai/mem0`, `langchain-ai/langmem`, *Agentic Memory (ACL 2026)*.
- **Application**: Convert intermediate learnings into structured YAML/JSON records (`docs/BUG_LEDGER.md`, `runtime_state.db`) rather than unstructured prose chat.

### B. AST Repository Mapping (Aider / Tree-sitter / CodeMap)
- **Concept**: Send only the concise AST tree (classes, methods, signatures) of the project to the model, rather than full file dumps.
- **GitHub Reference**: `paul-gauthier/aider` (Repo Map), `codemap` CLI.
- **Application**: Use `codemap query` and `codemap trace` to fetch exact symbol definitions in <50 tokens instead of reading 500-line files.

### C. Agent-Computer Interface (ACI) Verification (SWE-agent / OpenHands)
- **Concept**: Wrap all code operations in deterministic validation harnesses (linters, type checkers, unit tests). Never allow an agent to commit a change without automated verification.
- **GitHub Reference**: `princeton-nlp/SWE-agent`, `All-Hands-AI/OpenHands`.
- **Application**: Enforce the **Grounded Verification Triad** (`rtk test` / `pytest` / `mypy`) on every modification.

### D. Verbal Self-Reflection & Circuit Breakers (Reflexion / Hystrix)
- **Concept**: When a task fails twice, force an explicit diagnostic step (Root Cause, Expected vs Actual, Hypothesis) before allowing any further file modifications. If failures hit 3, trigger a circuit breaker.
- **GitHub Reference**: `noahshinn/reflexion`.
- **Application**: Integrated into `antigravity-perpetual`'s `CircuitBreaker` and `docs/BUG_LEDGER.md`.

---

## 4. Concrete Anti-Hallucination Protocol for Perpetual Execution

During this long-running perpetual development process, hallucination is prevented through **six deterministic, non-negotiable mechanisms**:

```
                  ┌───────────────────────────────────────────┐
                  │    ANTI-HALLUCINATION EXECUTION PIPELINE   │
                  └─────────────────────┬─────────────────────┘
                                        │
             ┌──────────────────────────┴──────────────────────────┐
             ▼                                                     ▼
┌─────────────────────────┐                               ┌─────────────────────────┐
│ 1. AST Grounding        │                               │ 2. Grounded Triad       │
│ • `codemap query`       │                               │ • Compiler / Linter     │
│ • No speculative grep   │                               │ • Automated pytest      │
└────────────┬────────────┘                               └────────────┬────────────┘
             │                                                         │
             ▼                                                         ▼
┌─────────────────────────┐                               ┌─────────────────────────┐
│ 3. Git Ground Truth     │                               │ 4. Neuro-Symbolic Gate  │
│ • Commit after green    │                               │ • Deterministic regex   │
│ • `git reset` on drift  │                               │ • Silicon circuit break │
└────────────┬────────────┘                               └────────────┬────────────┘
             │                                                         │
             ▼                                                         ▼
┌─────────────────────────┐                               ┌─────────────────────────┐
│ 5. Structured Ledger    │                               │ 6. Bounded Delegation   │
│ • `docs/BUG_LEDGER.md`  │                               │ • Council of 13 roles   │
│ • SQLite `runtime_state`│                               │ • Clean subagent context│
└─────────────────────────┘                               └─────────────────────────┘
```

### Protocol 1: The Grounded Verification Triad (No "Belief-Based" Completion)
- The agent NEVER claims code works because it looks correct.
- Verification requires executing the relevant test suite via `python -m pytest` or `rtk test`.
- **Rule**: If the exit code is not `0`, the task is NOT done. Zero exceptions.

### Protocol 2: Git as the Immutable Ground Truth
- The file system state is always anchored to Git commits.
- Before starting a refactor, inspect `git status` and `git diff`.
- If an agent veers off track or hallucinates inconsistent state, immediately run `git checkout -- .` to restore verified ground truth rather than trying to "fix forward" a hallucination.

### Protocol 3: AST Symbol Navigation via CodeMap
- Never guess an import path or class method name.
- Query the pre-indexed AST graph:
  ```powershell
  codemap query src/lunar_engine.py .
  codemap trace src/memory_vector.py .
  ```
- This guarantees 100% type and symbol accuracy while using less than 100 tokens per lookup.

### Protocol 4: Local NPU Hypersphere Grounding
- Key technical recipes and architecture constraints from the 14-Chapter monograph are embedded onto the unit hypersphere $S^{383}$ using the Intel Lunar Lake NPU (`minilm_l6`).
- When checking requirements, semantic similarity against the monograph recipes runs in $<3\text{ ms}$ on local silicon, ensuring code strictly adheres to documented hardware specifications.

### Protocol 5: Silicon Guardrails & Deterministic Regex DFAs (Chapter 13 Recipe 6)
- High-risk operations (file deletions, unauthorized network calls, out-of-bounds parameters) are filtered through deterministic deterministic finite automaton (DFA) regex patterns before execution.
- If a proposed action violates a safety contract, the execution layer rejects it before the tool call is dispatched.

### Protocol 6: Bounded Subagent Delegation (Council of 13 Specialists)
- Long, sprawling tasks are NOT run in a single monolithic context.
- Tasks are broken down and delegated to specialized subagents (e.g., `Speculative NPU Architect`, `Hardware Security Specialist`, `Autonomous DevOps Lead`).
- Each subagent operates with a fresh, compact context window focused exclusively on its single responsibility, eliminating needle-in-a-haystack attention degradation.
- Subagents report structured JSON outcomes back to the parent coordinator, updating `docs/BUG_LEDGER.md`.

---

## 5. Token Savings Summary

| Layer | Technique | Token Reduction Impact |
| :--- | :--- | :--- |
| **CLI Interception** | `rtk.exe` output compression (`diff`, `test`, `git`) | **60% – 90%** |
| **Code Navigation** | `codemap` AST query vs full directory reads | **80% – 95%** |
| **Skill Disclosure** | `antigravity-skills-manager` on-demand loading | **90%** prompt footprint reduction |
| **Vector Search** | Intel NPU local INT8 embeddings | **100%** cloud token elimination for retrieval |
| **Tool Execution** | Anthropic MCP script execution pattern | **95%** reduction in tool chatter |
