# AUTONOMOUS AGENT PROMPT: BUILD OUT THE SILICON-REFLEX NEURAL-COMPILER (SRNC) & `agy` PLATFORM

> **Target Agent Execution Mode**: Google Antigravity `/goal` Long-Running Autonomous Execution  
> **Target Environment**: Windows PowerShell, Intel Lunar Lake Silicon (Core Ultra 200V / NPU 4000), Local Unified Memory (UMA)  
> **Source Constitution**: [`docs/MASTER_TECHNICAL_CONSTITUTION_50_PAGES.md`](file:///c:/Users/Janmejai/Documents/antigravity/jolly-meitner/docs/MASTER_TECHNICAL_CONSTITUTION_50_PAGES.md)  
> **Graphify Knowledge Base**: `.graphify/graph.json` (Query via `npx @sentropic/graphify query`)  

---

## The Master Autonomous Directive (Copy & Run with `/goal`)

```markdown
/goal Build out the complete, end-to-end production implementation of the Silicon-Reflex Neural-Compiler (SRNC) and the `agy` developer platform as specified in docs/MASTER_TECHNICAL_CONSTITUTION_50_PAGES.md.

You are acting as a Principal Systems & Compiler Architect and Lead Open-Source Maintainer (combining the systems rigor of vLLM, Bun, Biome, and shadcn/ui). You have full operational autonomy to create, compile, wire, and formally verify every subsystem. Do NOT stop to ask trivial questions or leave placeholder stubs; execute sprints autonomously, self-heal on any build/test failures, enforce 100/100 Craft Score on all UI deliverables, update the Graphify knowledge graph, and terminate ONLY when all verification criteria pass with <!-- GOAL_COMPLETE -->.

### PRIMARY OBJECTIVES & EXECUTION SPRINTS

#### SPRINT 1: Silicon Hardware & Zero-Copy IPC Engine (`lunar_core`)
1. Extend `lunar_core/srnc.py` to support high-throughput, bidirectional zero-copy ring buffers over `Local\LunarNpuRingBuffer` (64MB UMA shared memory).
2. Wire real atomic Compare-And-Swap (CAS) pointers (`head`, `tail`) using Python `ctypes` / Windows Win32 API (`CreateFileMappingW`, `MapViewOfFile`).
3. Benchmark round-trip IPC latency: verify and assert round-trip dispatch latency < 50 microseconds.
4. Integrate the Deterministic DFA Circuit Breaker (256-state transition jump table) into the pre-tool command execution pipeline, guaranteeing < 2.2µs safety evaluation for shell commands.
5. Provide fallback graceful degradation for systems without Intel Lunar Lake NPU (DirectML / CPU AVX2 fallback).

#### SPRINT 2: Rowan Red-Green CST Engine (`agent-craft/src/engine/srnc-cst.ts`)
1. Implement a complete Rowan-style Red-Green Concrete Syntax Tree engine for TypeScript/JSX in `agent-craft`:
   - Green Node: Immutable, syntax-only, offset-agnostic, deduplicated via structural hash-consing.
   - Red Node: Lazy, zero-allocation cursor with parent pointers and absolute span calculation.
   - Trivia Invariance: Parse whitespace, indentation, comments, and delimiters losslessly, guaranteeing `Yield(CST) === SourceText` with 100% fidelity.
2. Implement surgical AST mutation methods:
   - `replaceNode(target: RedNode, replacement: GreenNode): SyntaxTree`
   - `wrapElement(target: RedNode, wrapperKind: string, attributes: Record<string, string>): SyntaxTree`
   - `updateClassName(target: RedNode, mutator: (classes: string[]) => string[]): SyntaxTree`
3. Verify that mutating a single JSX element preserves 100% of surrounding comments, spacing, and git blame across untouched lines.

#### SPRINT 3: Embedded Cassowary Simplex Solver & APCA OKLCH Optimizer
1. Implement the Cassowary linear simplex inequality constraint solver inside `agent-craft/src/engine/` and `lunar_core/`:
   - Mathematical formulation: minimize sum of slack variables subject to `width >= 44`, `height >= 44`, and viewport boundary limits.
   - Run constraint solving during CST traversal to automatically compute minimal padding/margin adjustments.
2. Implement the APCA (Accessible Perceptual Contrast Algorithm) solver with OKLCH lightness traversal:
   - Calculate APCA `Lc` contrast values across arbitrary foreground/background pairs.
   - Execute directional lightness stepping along constant Chroma and Hue in OKLCH space until `|Lc| >= 60.0` (body text) or `|Lc| >= 45.0` (headings).
3. Validate with unit tests in `agent-craft/tests/` demonstrating that low-contrast inputs are mathematically resolved in < 1ms.

#### SPRINT 4: The `agy` Unified CLI & Toolchain
1. Build the standalone CLI entry point (`npx agy` / `lunar_core/cli.py` / `agent-craft/bin/agy.js`):
   - `agy init`: Auto-detects project framework (Next.js, Vite, Remix), probes for local Lunar Lake NPU 4000 hardware, and instantiates `.agy/` config.
   - `agy map [path]`: Traverses repository ASTs and generates an interactive SVG/HTML architecture knowledge graph in < 1.5s using Graphify.
   - `agy gate`: Runs the 400ms pre-commit and GitHub Action PR linter. Fails on contrast regressions or viewport overflows; outputs surgical patch diffs.
   - `agy add <primitive>`: Copies verified, copy-paste accessible UI component primitives directly into `components/ui/` (shadcn pattern).
   - `agy audit`: Produces an APCA, WCAG 2.1 AAA, and layout health audit report.

#### SPRINT 5: The Verified Primitive Registry (`components/ui/` & Atelier Pavilions)
1. Deconstruct the 10 Atelier Antigravity 2.0 pavilions into standalone, modular, copy-paste primitives in `components/ui/`:
   - `DynamicAmbientIsland`: Viewport-aware morphing island with zero-clipping popovers.
   - `CalabiYauCard`: Procedural 3D WebGL shader card with fallback 2D vector mode.
   - `TactileMicroSwitch`: WebAudio synthesized 432Hz/528Hz Solfeggio acoustic feedback switch.
   - `SovereignLedgerTable`: Delta reconciliation audit table with 1-click dispute dossier export.
   - `NumberFlowOdometer`: Smooth tabular numerical transition gauge.
2. Ensure every single primitive is certified with a **100/100 Flawless Craft Score** via `agent-craft audit`.
3. Provide zero-dependency code that developers can drop into any React/Vite/Next.js application.

#### SPRINT 6: End-to-End Verification, Graphify Ingestion & Quality Gate
1. Run all unit and integration test suites (`npm test` in `agent-craft`, `pytest` in `lunar_core`).
2. Run `npx @sentropic/graphify update --scope all --force` to re-index all newly created files, symbols, methods, and relationships into `.graphify/graph.json`.
3. Verify graph queryability:
   - `npx @sentropic/graphify query "siliconreflexneuralcompiler"`
   - `npx @sentropic/graphify query "CassowarySimplex"`
   - `npx @sentropic/graphify query "RowanCstEngine"`
4. Audit all HTML deliverables with `agent-craft audit` and verify a **100/100 Flawless Craft Score**.
5. Update `walkthrough.md` with complete benchmark numbers, file links, and architecture diagrams.
6. Terminate with `<!-- GOAL_COMPLETE -->`.

### RULES OF ENGAGEMENT & OPERATIONAL STANDARDS
- **PowerShell First**: All CLI commands must be formulated for Windows PowerShell. Avoid raw bashisms or `cd` navigation.
- **Token Efficiency**: Use `rtk` (e.g., `rtk test`, `rtk git status`) to compress verbose CLI outputs whenever appropriate.
- **Surgical Diffs**: Modify files contiguously without overwriting unrelated logic or destroying existing tests.
- **Self-Healing Loop**: If a build error, type error, or test failure occurs, inspect the error output immediately, formulate a surgical fix, apply it, and re-run until passing. Do NOT ask user permission to fix compile errors.
- **Zero Placeholders**: Never write `// TODO`, `/* Implement later */`, or truncated dummy mock code. Write real, executable, production-grade algorithms.
```

---

## Key Prompt Engineering Elements Incorporated

| Prompt Engineering Technique | Implementation in this Prompt | Benefit for `/goal` Mode |
| :--- | :--- | :--- |
| **Clear Persona & Authority Framing** | "Principal Systems & Compiler Architect (vLLM, Bun, Biome, shadcn/ui)" | Primes the LLM for systems-level performance, rigorous types, and clean primitives instead of superficial mockups. |
| **Decomposed Linear Sprints** | 6 clearly delineated sprints with strict inputs, outputs, and dependencies | Prevents context wandering; allows the agent to systematically execute and check off milestones. |
| **Self-Healing Protocol** | Explicit instructions to intercept errors, read compiler diagnostics, and patch without stopping | Eliminates premature stoppage on minor syntax or type errors. |
| **Strict Acceptance Metrics** | Latency budgets (<50µs IPC, <2.2µs DFA, <1ms Cassowary), 100/100 craft score, 100% test pass rate | Establishes unambiguous mathematical pass/fail boundaries. |
| **Knowledge Graph Grounding** | Requires re-indexing into `@sentropic/graphify` and verifying via `graphify query` | Guarantees that all newly built code becomes permanently searchable by subsequent agent turns. |
| **Termination Token** | `<!-- GOAL_COMPLETE -->` | Enforces completion discipline so the agent continues until all 6 sprints are fully verified. |

---

## How to Trigger

1. In the Antigravity chat input, type `/goal` followed by the contents of the Master Autonomous Directive block above.
2. The agent will enter autonomous goal mode, execute Sprints 1 through 6, verify all tests and craft scores, update the Graphify knowledge base, and report the verified build upon completion.
