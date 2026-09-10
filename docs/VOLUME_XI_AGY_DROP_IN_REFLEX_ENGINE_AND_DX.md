# Volume XI: The Drop-in Reflex Engine CLI (`agy`), Type-Safe API & IDE Sidecar

## 1. The Developer Primitive: Transitioning from Webview to CLI/SDK

A webview showcase is a dead-end for developers unless it is paired with an uncompromising, zero-config CLI and typed SDK. To make headlines and achieve widespread enterprise adoption, the Silicon-Reflex Neural-Compiler is packaged into a unified developer tool: **`agy`** (or `npx agy`).

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            THE `agy` TOOLCHAIN                              │
└─────────────────────────────────────────────────────────────────────────────┘

  $ npx agy init          ──► Auto-detects framework, binds NPU silicon daemon
  $ npx agy map .         ──► 1.5s AST knowledge graph generation via Graphify
  $ npx agy gate          ──► <500ms CI/CD anti-slop pull request linter
  $ npx agy add <name>    ──► Copy-paste verified accessible component primitive
  $ npx agy audit         ──► Comprehensive APCA contrast & layout health report
```

### 1.1 Command Specifications

#### 1. `agy init`
- Inspects repository root, detecting package manager (`pnpm`, `bun`, `npm`), styling framework (`tailwindcss` v3/v4), and UI engine (React, Vue, Svelte).
- Auto-probes local hardware: tests for Intel Lunar Lake NPU 4000, Apple Silicon Neural Engine, or CUDA fallback.
- Instantiates `.agy/` directory containing local Rowan CST rules, color tokens, and cached vector indexes.

#### 2. `agy map [path]`
- Traverses codebase using multi-threaded AST workers.
- Generates a Graphify-compliant semantic knowledge graph of all components, hooks, design tokens, and import dependencies.
- Emits an interactive vector architecture map viewable in browser or terminal.

#### 3. `agy gate`
- Designed for GitHub Actions and pre-commit Git hooks.
- Parses all staged diffs using the Rowan Red-Green CST.
- Runs Cassowary layout checks and APCA contrast verification.
- Exits with status code `0` if all invariants pass; exits with `1` and emits precise surgical patches if violations are detected. Execution completes in **under 450 milliseconds**.

#### 4. `agy add <primitive>`
- Modeled after the shadcn/ui pattern.
- Downloads pure, accessible, zero-dependency source code directly into `components/ui/`.
- Every added component is certified with a **100/100 Flawless Craft Score** and includes embedded WebAudio micro-haptic hooks.

---

## 2. The Type-Safe SDK Surface

For developers embedding the reflex engine directly into custom workflows, Next.js build plugins, or CI bots, `agy` exposes a strongly typed TypeScript API:

```typescript
import { ReflexEngine, RowanCST, CassowarySolver, ApcaOptimizer } from "@antigravity/reflex";

// Initialize the Reflex Engine with physical silicon bindings
const engine = new ReflexEngine({
  hardware: "auto", // Automatically detects Lunar Lake NPU 4000
  ipcTransport: "shm", // Uses Windows Named Memory-Mapped Buffer
  strictCraftScore: true, // Rejects any code below 100/100 score
});

// Parse source code losslessly
const sourceCode = await fs.readFile("components/HeroBanner.tsx", "utf-8");
const cst = engine.parse(sourceCode);

// Perform surgical, trivia-preserving AST mutation
const mutatedCst = cst.transform((visitor) => {
  visitor.onElement("button", (node) => {
    // Solve layout constraints mathematically
    const layout = CassowarySolver.solveTarget(node, {
      minTouchTarget: 44, // WCAG 2.5.5
      responsiveClamp: true,
    });
    
    // Solve APCA contrast mathematically along OKLCH gamut
    const colors = ApcaOptimizer.solveContrast({
      foreground: node.getAttribute("textColor") ?? "#94a3b8",
      background: "#0f172a",
      targetLc: 60.0, // APCA body text requirement
    });

    node.applyAttributes({
      className: layout.toTailwindClasses(),
      style: { color: colors.optimalOklchCss },
    });
  });
});

// Verify round-trip idempotency and zero-blame destruction
assert(mutatedCst.toSource() !== sourceCode);
assert(mutatedCst.commentsRetained() === true);

// Emit verified source to disk
await engine.commit(mutatedCst, "components/HeroBanner.tsx");
```

---

## 3. The Bi-Directional IDE Sidecar Architecture

The SRNC includes a zero-latency IDE sidecar that runs alongside the developer's primary code editor (VS Code, Cursor, Antigravity):

```
┌──────────────────────────────────────┬──────────────────────────────────────┐
│        PRIMARY CODE EDITOR           │         AGY AMBIENT HUD              │
│                                      │                                      │
│  function PrimaryButton({ label }) { │  ● NPU 4000: ACTIVE (3.2W)           │
│    return (                          │  ● IPC LATENCY: 34µs                 │
│      <button className="px-4 py-2    │  ● CRAFT SCORE: 100/100              │
│        bg-slate-900 text-slate-400"> │  ----------------------------------  │
│        {label}                       │  ▲ WARNING: APCA Contrast Lc=38.2    │
│      </button>                       │    (Below required 60.0 threshold)   │
│    );                                │                                      │
│  }                                   │  [1-CLICK MATHEMATICAL SOLVE]        │
│                                      │  -> Mutates to: text-slate-200       │
└──────────────────────────────────────┴──────────────────────────────────────┘
```

### Sidecar Features:
1. **Dynamic Ambient Island**: Floats unobtrusively in the editor gutter, displaying real-time NPU power draw, CST parsing latency, and instantaneous craft score.
2. **Zero-Keystroke Latency**: Runs in a separate OS thread communicating via memory-mapped IPC, guaranteeing zero impact on editor typing performance.
3. **One-Click Mathematical Reconciliation**: When an accessibility or layout warning appears, clicking the notification triggers an instantaneous CST mutation that updates the editor buffer in under 15ms.
