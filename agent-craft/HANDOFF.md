# `agent-craft` v1.0.0 — GitHub & npm Publishing Handoff Guide ◆

This document provides step-by-step instructions to publish `agent-craft` as an open-source tool on GitHub and npm, register it in the MCP registry, and prepare for v2 development.

---

## 1. Pre-Publish Verification Status

Before releasing to the public, the package has been fully verified locally:

* [x] **TypeScript Build**: `tsc` cleanly emits ES modules to `./dist/` with `.d.ts` declarations.
* [x] **Automated Test Suite**: 8/8 tests pass in <1s across relative luminance, OKLCH solver, AST detection, surgical repair, and Stdio MCP handshake.
* [x] **Self-Audit**: `node ./bin/agent-craft.js audit src/` passes with **100/100 Flawless Craft**.
* [x] **Binary Shebang**: `bin/agent-craft.js` has `#!/usr/bin/env node` configured.
* [x] **Package Exports**: `package.json` specifies `"bin"`, `"main"`, `"types"`, and `"files": ["dist", "bin", "README.md", "LICENSE"]`.
* [x] **License**: Open-source MIT license included.

---

## 2. GitHub Repository Initialization & Push

You can publish `agent-craft` either as a standalone GitHub repository (recommended for maximum open-source adoption) or as part of your monorepo.

### Option A: Standalone GitHub Repository (Recommended)

Run the following commands in PowerShell from the `agent-craft` folder:

```powershell
# Navigate to agent-craft
Set-Location c:\Users\Janmejai\Documents\antigravity\jolly-meitner\agent-craft

# Initialize git repository (if standalone)
git init -b main

# Add all files (respects .gitignore)
git add .

# Initial commit
git commit -m "feat: initial release v1.0.0 of agent-craft visual craft linter & MCP server"

# Create public repository on GitHub using GitHub CLI:
gh repo create agent-craft --public --source=. --remote=origin --push

# (Or if creating manually via github.com/new):
# git remote add origin https://github.com/janmejai2002/agent-craft.git
# git push -u origin main
```

### Option B: Push Git Release Tag

```powershell
git tag -a v1.0.0 -m "Release v1.0.0: The Anti-Slop Visual Craft Linter & MCP Server"
git push origin v1.0.0
```

---

## 3. npm Registry Publishing

Publishing to the global npm registry enables developers and AI coding agents to run `npx agent-craft audit .` anywhere in the world without prior installation.

### Step 1: Log in to npm
```powershell
npm login
```
*(Follow prompt to enter your npm username, password, and 2FA code)*

### Step 2: Dry Run (Verify Tarball Contents)
```powershell
npm publish --dry-run
```
*Verify that only `dist/`, `bin/`, `README.md`, `LICENSE`, and `package.json` are packaged (~45 KB).*

### Step 3: Publish to npm (Public Access)
```powershell
npm publish --access public
```

### Step 4: Test Global Execution
```powershell
npx agent-craft --version
# Output: 1.0.0

npx agent-craft contrast text-slate-400 bg-white
```

---

## 4. MCP Registry & Ecosystem Integration

### A. Smithery.ai Registry
Smithery is the premier MCP server catalog used by Claude Desktop, Cursor, and Windsurf users.
```powershell
# In the agent-craft directory:
npx -y @smithery/cli@latest init
```

### B. Client Configuration Snippets for Users
Include these in the README or share with users:

#### Claude Code / Claude Desktop (`claude_desktop_config.json`)
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

#### Cursor (`~/.cursor/mcp.json`)
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

#### Antigravity (`~/.gemini/config/mcp_config.json`)
```json
{
  "agent-craft": {
    "command": "node",
    "args": ["c:/Users/Janmejai/Documents/antigravity/jolly-meitner/agent-craft/bin/agent-craft.js", "mcp"]
  }
}
```

---

## 5. GitHub Release Notes Template (v1.0.0)

When creating the release on GitHub (`gh release create v1.0.0`), use the following copy:

```markdown
# 🚀 agent-craft v1.0.0 — Stop AI Slop in Frontend Code

We are thrilled to announce the initial open-source release of **`agent-craft`**, the deterministic visual craft linter, mathematical contrast solver, and MCP server designed specifically for AI coding agents and frontend developers.

### 🌟 Highlights
- **Sub-15ms AST Analysis**: Scans TSX, JSX, HTML, Svelte, and Vue files locally in <15ms with 0 cloud tokens consumed.
- **6-Layer Deterministic Rule Matrix**:
  1. *Layer 1: Anti-Slop Gate*: Detects purple-to-indigo linear clichés, card-in-card nesting, generic copy, and 01 eyebrows.
  2. *Layer 2: Mathematical Contrast*: Computes exact WCAG 2.1 AA/AAA and APCA ratios with dark/light canvas awareness and Tailwind opacity alpha compositing.
  3. *Layer 3: OKLCH Auto-Solver*: Algorithmic color shift that preserves brand hue while converging to accessible contrast.
  4. *Layer 4: Physical Touch Ergonomics*: Flags sub-44px touch targets on mobile viewports.
  5. *Layer 5: Viewport Physics*: Replaces legacy `h-screen` jumping with `min-h-[100dvh]`.
  6. *Layer 6: Token Drift*: Enforces deliberate spacing/radius scales and eliminates arbitrary bracket bleed (`p-[17px]`).
- **Stdio MCP Server**: Native integration with Claude Code, Cursor, Windsurf, Antigravity, and OpenCode.
- **Surgical Auto-Fixer**: Safely repairs accessibility and viewport defects in place without stochastic LLM rewrites.

### 📦 Installation
```bash
npx agent-craft audit src/
npx agent-craft fix src/
npx agent-craft contrast text-slate-400 bg-white
```
```

---

## 6. Next Steps: Developing v2

See [docs/ROADMAP_V2.md](./docs/ROADMAP_V2.md) for the complete v2 technical specification, Tree-Sitter parsing architecture, and headless visual verification engine.
