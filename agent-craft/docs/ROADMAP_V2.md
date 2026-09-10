# agent-craft v2.0 Architecture & Development Roadmap ◆

> **The Autonomous Visual Verification, Spatial Ergonomics & Multi-Framework Engine for AI Coding Agents.**

---

## 1. Executive Summary & v2 Vision

While **`agent-craft` v1** established the industry benchmark for sub-15ms deterministic anti-slop pattern detection, mathematical WCAG/APCA contrast solving, and Stdio MCP integration, modern web applications present challenges that transcend static source code inspection:

1. **Dynamic CSS Cascades & Utility Merges**: Classes concatenated via `cn()`, `clsx()`, `cva()`, or computed state (`isActive ? "bg-zinc-900" : "bg-white"`).
2. **True Rendered Pixel Semantics**: Complex multi-stop background gradients, `backdrop-filter: blur()`, blend modes, and CSS canvas shaders where contrast depends on computed pixels, not class names alone.
3. **Spatial Hierarchy & Negative Space Balance**: Unbalanced padding ratios, irregular typographic scale steps, and dead visual zones that feel "AI-generated" even if they pass basic linting.
4. **Cross-Framework Modernity**: Svelte 5 runes, Vue 3 `<script setup>`, React 19 Server Components, and Astro islands.

**`agent-craft` v2** evolves from an AST/token linter into a **comprehensive local visual verification platform**—combining native Tree-sitter AST intelligence, ultra-fast headless visual evaluation, and bidirectional Figma design token synchronization.

---

## 2. Core Architectural Pillars for v2

```
┌────────────────────────────────────────────────────────────────────────┐
│                          AGENT-CRAFT v2 CORE                           │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
    ┌───────────────────────────────┼───────────────────────────────┐
    ▼                               ▼                               ▼
┌───────────────────────┐   ┌───────────────────────┐   ┌───────────────────────┐
│       PILLAR 1        │   │       PILLAR 2        │   │       PILLAR 3        │
│  Tree-Sitter / SWC    │   │ Headless Browser      │   │ Spatial Balance &     │
│  AST Semantic Graph   │   │ Visual Verification   │   │ Typographic Harmony   │
│                       │   │                       │   │                       │
│ • TSX / JSX / HTML5   │   │ • Real Computed Styles│   │ • Modular Scale Math  │
│ • Svelte 5 / Vue 3    │   │ • Pixel Contrast Diff │   │ • Negative Space Ratio│
│ • cn() / cva() Eval   │   │ • Sub-50ms Headless   │   │ • Optical Alignment   │
└───────────────────────┘   └───────────────────────┘   └───────────────────────┘
    │                               │                               │
    └───────────────────────────────┼───────────────────────────────┘
                                    │
    ┌───────────────────────────────┴───────────────────────────────┐
    ▼                                                               ▼
┌───────────────────────┐                               ┌───────────────────────┐
│       PILLAR 4        │                               │       PILLAR 5        │
│ Bidirectional Tokens  │                               │ Agent Self-Correction │
│ & Figma W3C Bridge    │                               │ & Extensible Plugins  │
│                       │                               │                       │
│ • $tokens.json Sync   │                               │ • <200 Token Feedback │
│ • Tailwind v4 @theme  │                               │ • agent-craft.config  │
│ • Figma Variables API │                               │ • Community Rulesets  │
└───────────────────────┘                               └───────────────────────┘
```

---

### Pillar 1: Native Tree-Sitter / SWC Multi-Framework AST Engine

In v1, fast regex pattern matching and line analysis achieved sub-15ms audits. In v2, we introduce a native parser backend using `@swc/core` and `web-tree-sitter` for deep semantic awareness:

1. **`cn()` / `clsx()` / `cva()` Expression Evaluation**:
   * Evaluates ternary and boolean expressions in utility classes:
     ```tsx
     <button className={cn("px-4 py-2", isPrimary ? "bg-indigo-600 text-white" : "text-slate-400")}>
     ```
   * Resolves active branches to detect contrast failures even inside dynamic expressions.
2. **True Framework Support**:
   * **React 19 / JSX / TSX**: Full JSX element tree traversal with component prop tracking.
   * **Vue 3 SFC**: Parsing `<template>`, `<script setup>`, and scoped `<style>`.
   * **Svelte 5**: Native support for Svelte 5 runes (`$state`, `$derived`, `$props`) and snippet blocks.
   * **Astro**: Multi-framework island parsing.
3. **Z-Index & Stacking Context Tracker**:
   * Maps DOM hierarchy and flags accidental stacking collisions (e.g. modals trapped beneath fixed headers).

---

### Pillar 2: Headless Browser Visual Verification (<50ms Local)

Static linters cannot compute the final color of text placed over an image or multi-stop CSS gradient. v2 introduces an optional `--visual` flag powered by a lightweight local headless runner:

1. **Computed Style Extraction**:
   * Inspects `window.getComputedStyle(element)` to get actual resolved RGB values, line heights, and margins.
2. **Real Pixel Contrast on Complex Surfaces**:
   * Renders element bounding boxes into an offscreen canvas.
   * Runs pixel-sampling contrast algorithms across background gradients, glassmorphism backdrops, and video overlays.
3. **Real Bounding-Box Overlap & Touch Ergonomics**:
   * Calls `element.getBoundingClientRect()` across standard viewport matrices (375x812, 768x1024, 1440x900).
   * Flags overlapping touch targets where tap hitboxes collide.
4. **Layout Shift & Viewport Bleed Verification**:
   * Emulates mobile browser viewport collapse to prove zero horizontal scrollbars and 100dvh stability.

---

### Pillar 3: Spatial Balance & Typographic Harmony Math

AI agents frequently emit irregular spacing steps (`mt-7`, `pb-11`, `text-[19px]`) that lack visual rhythm. v2 introduces mathematical aesthetic scoring:

1. **Modular Scale Enforcement**:
   * Validates typographic steps against classic harmonic ratios:
     * *Minor Third (1.200)*
     * *Major Third (1.250)*
     * *Perfect Fourth (1.333)*
     * *Golden Ratio (1.618)*
   * Flags non-harmonic font-size and line-height parings.
2. **Negative Space Density Scoring**:
   * Computes the ratio of ink (content pixels) to canvas (whitespace).
   * Detects "cramped" or "sparse" sections before user testing.
3. **Optical Weight & Symmetry Solver**:
   * Evaluates visual weight across multi-column grids and suggests balanced flex basis adjustments.

---

### Pillar 4: Bidirectional Design System & Figma Token Bridge

Seamlessly bridge developer code and designer canvas:

1. **W3C Design Tokens Standard (`$tokens.json`)**:
   * Parses and emits standardized design tokens format.
2. **Tailwind v4 Native `@theme` Integration**:
   * Generates modern CSS `@theme` declarations:
     ```css
     @theme {
       --color-canvas: oklch(0.14 0.02 240);
       --color-surface: oklch(0.18 0.03 240);
       --color-accent: oklch(0.68 0.16 195);
     }
     ```
3. **Figma REST API Sync**:
   * `agent-craft sync --figma <file-key>`: Pulls Figma Variables (colors, spacing, radius) directly into the project.
   * `agent-craft export --figma`: Exports calibrated OKLCH ramps back to Figma.

---

### Pillar 5: Agent Self-Correction & Extensible Plugin API

Make AI agents self-healing without burning context tokens:

1. **Compact Prompt Feedback Mode (`agent-craft feedback`)**:
   * Produces ultra-dense, actionable corrective feedback for LLMs (<180 tokens):
     ```json
     {
       "action": "fix_craft",
       "file": "src/Hero.tsx",
       "issues": [
         { "line": 24, "defect": "low_contrast", "replace": "text-slate-400", "with": "text-slate-600", "reason": "WCAG AA (4.5:1)" },
         { "line": 42, "defect": "sub_44px_touch", "replace": "h-8 w-8", "with": "h-11 w-11", "reason": "Apple HIG" }
       ]
     }
     ```
   * Gives agents exact line-by-line replacement instructions, eliminating iterative guessing.
2. **Custom Configuration & Community Rulesets (`agent-craft.config.ts`)**:
   ```typescript
   import { defineConfig } from "agent-craft";
   import shadcnRules from "@agent-craft/rules-shadcn";

   export default defineConfig({
     archetype: "waibi-sabi",
     rules: [
       ...shadcnRules,
       {
         id: "company/brand-accent",
         category: "tokens",
         severity: "error",
         check: (node, ctx) => {
           // Custom AST inspection
         }
       }
     ]
   });
   ```

---

## 3. Implementation Phasing & Milestones

| Phase | Milestone | Target Deliverables | Est. Timeline |
| :--- | :--- | :--- | :--- |
| **Phase 1** | **Parser Modernization** | Replace regex with `@swc/core` AST visitor; dynamic `cn()` evaluation; support Vue & Svelte. | Weeks 1–3 |
| **Phase 2** | **Headless Visual Runner** | Add Playwright/Camoufox visual mode; computed style extraction; real gradient contrast. | Weeks 4–6 |
| **Phase 3** | **Spatial & Typographic Math** | Modular scale rules; optical alignment checks; negative space balance metric. | Weeks 7–8 |
| **Phase 4** | **Figma & Token Bridge** | `$tokens.json` W3C format; Tailwind v4 `@theme` generator; Figma variables sync. | Weeks 9–10 |
| **Phase 5** | **Plugin API & v2 Release** | `agent-craft.config.ts`; community rulesets; public v2 release. | Weeks 11–12 |

---

## 4. Immediate Development Setup for Contributors

```powershell
# Clone and install dependencies
git clone https://github.com/janmejai2002/agent-craft.git
cd agent-craft
bun install

# Run build & watch mode
bun run watch

# Run test suite
bun test
```
