# agent-craft ◆

> **The Anti-Slop Visual Craft Linter, Perceptual Contrast Engine & MCP Server for AI Coding Agents and Frontend Engineers.**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen.svg)]()
[![Execution Speed](https://img.shields.io/badge/latency-%3C15ms-00A9B8.svg)]()
[![Zero Token Overhead](https://img.shields.io/badge/token%20cost-%240.00-10B981.svg)]()

---

## The Problem: AI Slop & Token Bloat

Over the last 18 months, AI coding agents (**Claude Code**, **Cursor**, **Windsurf**, **Antigravity**, **OpenCode**) have become standard developer companions. However, their frontend visual generation is plagued by **"AI Slop"**:

* ❌ **Predictable Purple Gradients:** Every hero button defaults to `from-purple-500 to-indigo-500`.
* ❌ **Nested Card Hell:** Cards inside cards inside cards (`border` recursion) creating visual claustrophobia.
* ❌ **Contrast Failures:** Light gray text (`text-slate-400` on white, 2.56:1) failing WCAG 2.1 AA accessibility.
* ❌ **Arbitrary Bracket Bleed:** Hallucinated values (`p-[17px]`, `w-[800px]`, `rounded-[11px]`) destroying design token systems.
* ❌ **Mobile Viewport Jumps:** `h-screen` causing dynamic address bar jumping on iOS Safari; tap targets under 44px.
* ❌ **Prompt Skill Context Decay:** Existing tools (*Impeccable*, *Hallmark*) rely on injecting 5,000–12,000 tokens of natural language rules into every conversation turn, costing real money and suffering from LLM attention decay.

---

## The MOAT: Why `agent-craft` Wins

```
┌─────────────────────────────────────────────────────────────┐
│                       AI AGENT LOOP                         │
│  (Claude Code, Cursor, Windsurf, Antigravity, OpenCode)     │
└──────────────────────────────┬──────────────────────────────┘
                               │ MCP Call / CLI Subprocess (<15ms)
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                        AGENT-CRAFT                          │
│                                                             │
│  ┌───────────────────────┐       ┌───────────────────────┐  │
│  │    Fast AST Parser    │       │ Perceptual Color Math │  │
│  │ (JSX, TSX, Svelte,Vue)│       │  (OKLCH / APCA / WCAG)│  │
│  └───────────┬───────────┘       └───────────┬───────────┘  │
│              │                               │              │
│  ┌───────────▼───────────────────────────────▼───────────┐  │
│  │          6-Layer Deterministic Rule Matrix            │  │
│  │ • Anti-Slop Cliché Detector (AST patterns)            │  │
│  │ • Perceptual Contrast Auto-Solver                     │  │
│  │ • Token Drift & Arbitrary Value Enforcer              │  │
│  │ • Ergonomics & Touch Target Validator                 │  │
│  │ • Mobile Viewport & Layout Physics Verifier           │  │
│  └───────────────────────────┬───────────────────────────┘  │
│                              │                              │
│              ┌───────────────┴───────────────┐              │
│              ▼                               ▼              │
│    Structured JSON Diagnostics       One-Click Auto-Fix     │
│   (For AI Agent Context Injection)   (AST Code Rewrite)     │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                      CI / CD & HOOKS                        │
│   (GitHub Actions, Pre-commit, Local Terminal, Zero LLM Cost)│
└─────────────────────────────────────────────────────────────┘
```

| Dimension | Prompt Skills (Impeccable / Hallmark) | Traditional Linters (oxlint) | **`agent-craft`** |
| :--- | :--- | :--- | :--- |
| **Token Cost per Run** | 5,000–12,000 tokens ($0.05–$0.20) | 0 tokens | **0 Tokens ($0.00)** |
| **Execution Latency** | 3,000–8,000 ms (LLM latency) | 50–200 ms | **< 15 ms (Deterministic Engine)** |
| **Contrast Math** | Hallucinated estimates | None | **Exact WCAG 2.1 AA/AAA + APCA** |
| **OKLCH Auto-Solver** | None | None | **Preserves hue while shifting lightness** |
| **CI / CD Quality Gate** | Impossible (Cannot run in CI) | Syntax only | **Full GitHub Action & Husky Hooks** |
| **AI Agent MCP Server** | None (Prompt only) | None | **Stdio MCP Server for all agents** |
| **Surgical Code Fixer** | Stochastic LLM rewrites | Class sorting only | **Deterministic AST/line auto-fixer** |

---

## Quickstart

### 1. Developer CLI

```powershell
# Audit current codebase for visual anti-patterns & contrast bugs
npx agent-craft audit src/

# Audit and output an interactive HTML visual report
npx agent-craft audit src/ --html craft-report.html

# Apply surgical auto-fixes to repair contrast, viewports, and tokens
npx agent-craft fix src/components/Hero.tsx

# Mathematically solve contrast between any two colors
npx agent-craft contrast text-slate-400 bg-white
# Output:
# Contrast Ratio: 2.56:1 [FAIL]
# ⚡ Algorithmic OKLCH Correction (Preserving Hue):
#   Target Hex:     #69788C
#   Achieved Ratio: 4.50:1
#   Tailwind Class: text-slate-500

# Generate curated design tokens and DESIGN.md constitution
npx agent-craft tokens waibi-sabi --format tailwind-v4
```

### 2. Multi-Agent Setup (MCP Server)

Add `agent-craft` to your MCP client configuration (`.cursor/mcp.json`, `claude_desktop_config.json`, or Windsurf):

```json
{
  "mcpServers": {
    "agent-craft": {
      "command": "npx",
      "args": ["-y", "agent-craft", "mcp"]
    }
  }
}
```

Now your AI coding agent has direct access to 5 native craft tools:
- `craft_audit`: Run static AST craft audit on any file or code snippet in <15ms.
- `craft_fix`: Automatically fix craft and contrast violations in code.
- `craft_contrast`: Compute exact WCAG & APCA contrast and return the OKLCH mathematical fix.
- `craft_tokens`: Generate calibrated design tokens in CSS variables, Tailwind v4 `@theme`, or `DESIGN.md`.
- `craft_archetype`: Fetch aesthetic guidelines, font pairings, and banned patterns for any design archetype.

---

## The 6-Layer Deterministic Rule Matrix

### Layer 1: Slop & Cliché Gates (`rules/slop.ts`)
* `slop/gradient-hero`: Flags overused AI linear gradient buttons (`from-purple-500 to-indigo-500`).
* `slop/card-in-card`: Flags nested rounded/bordered cards inside parent cards.
* `slop/eyebrow-number`: Flags arbitrary pseudo-technical numbering badges (`01 · FEATURES`).
* `slop/generic-copy`: Flags AI placeholder copy (`99.9% Uptime`, `Trusted by 100k devs`, `Acme Corp`, `John Doe`).
* `slop/neon-glow`: Flags saturated neon box-shadows (`shadow-purple-500/50`).
* `slop/pure-black-white`: Flags uncalibrated pitch black `#000000` canvas backgrounds.

### Layer 2: Ergonomics & Viewport Physics (`rules/ergonomics.ts` & `rules/viewport.ts`)
* `ergonomics/tap-target`: Enforces Apple HIG / WCAG 2.2 minimum 44x44px touch bounding box on mobile.
* `ergonomics/fixed-width`: Flags fixed pixel widths (`w-[800px]`) that blowout mobile viewports; converts to `max-w-[800px] w-full`.
* `ergonomics/safe-area`: Enforces `pb-safe` / `env(safe-area-inset-bottom)` on fixed bottom bars.
* `viewport/safari-jump`: Flags legacy `h-screen` (`100vh`) that jumps on mobile address bar scroll; upgrades to `h-[100dvh]`.
* `viewport/unconstrained-modal`: Enforces `max-h-[85vh] overflow-y-auto` on dialog and drawer overlays.

### Layer 3: Accessibility & Mathematical Contrast (`rules/contrast.ts`)
* `a11y/wcag-aa-contrast`: Mathematical contrast calculation of text vs background using exact W3C relative luminance.
* `a11y/outline-suppression`: Flags `outline-none` without corresponding `focus-visible:ring-*`.
* `a11y/missing-icon-label`: Interactive `<button>` with icon must provide `aria-label` or `title`.
* `a11y/image-alt`: Flags `<img>` missing `alt` attribute.

### Layer 4: Design Token Bleed (`rules/tokens.ts`)
* `token/arbitrary-spacing`: Snaps arbitrary pixel brackets (`p-[17px]`) to the standard 4px modular scale (`p-4`).
* `token/arbitrary-color`: Flags raw hex utility classes (`bg-[#1a1b26]`) when tokens should be used.

### Layer 5: Motion & Hardware Acceleration (`rules/motion.ts`)
* `perf/layout-animation`: Flags transitions targeting layout properties (`transition-all`, `transition-width`, `top`, `left`) that trigger CPU reflows.
* `perf/reduced-motion`: Enforces `motion-reduce:animate-none` on infinite animations (`animate-spin`, `animate-bounce`).
* `perf/mobile-blur-overuse`: Flags heavy backdrop blur (`backdrop-blur-2xl`) that drops frames on mobile GPUs.

---

## Curated Design Archetypes

`agent-craft` provides 4 anti-slop design archetypes to prevent aesthetic convergence:

1. **wAIbi-sabi (`waibi-sabi`)**: Japanese imperfection meets modern tech. Deep obsidian canvas (`#0B0E14`), mist surface (`#141923`), mineral water accent (`#00A9B8`), moss accent (`#6E8C63`), low optical fatigue.
2. **Obsidian Slate (`slate-minimal`)**: Ultra-crisp engineered precision. Zinc-950 canvas (`#09090B`), cobalt accent (`#2563EB`), dense micro-borders, zero visual noise.
3. **Swiss International (`editorial-swiss`)**: Asymmetric typography, museum-grade paper canvas (`#FBFBFA`), vermilion cinnabar accent (`#E11D48`), high editorial authority.
4. **Cyber Terminal (`precision-terminal`)**: Mission-critical operations HUD. Graphite canvas (`#050608`), phosphor emerald accent (`#10B981`), monospace data tables.

---

## CI/CD Quality Gate (GitHub Actions)

Add `.github/workflows/agent-craft-ci.yml` to your repository:

```yaml
name: Visual Craft & Anti-Slop Quality Gate

on: [push, pull_request]

jobs:
  audit:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
        with:
          node-version: 20
      - name: Run agent-craft Audit
        run: npx agent-craft audit src/ --fail-on-error --html craft-report.html
      - uses: actions/upload-artifact@v4
        if: always()
        with:
          name: craft-report
          path: craft-report.html
```

---

## Contributing & License

Contributions are welcome! Please submit PRs with test cases in `tests/`.

License: [MIT](LICENSE)
