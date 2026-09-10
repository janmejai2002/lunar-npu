"""
FastMCP 2.0 Extension Pack & Universal Client Auto-Installer
===========================================================
Discovers and configures local AI coding agent clients on Windows:
- Claude Desktop
- Cursor (IDE & Roo-Cline)
- Windsurf (Codeium)
- Antigravity (Google)
- VS Code (Cline / Roo-Code)

Safely updates mcpServers configuration with automatic backup creation.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


def get_default_lunar_server_config() -> Dict[str, Any]:
    """Return the canonical MCP server specification for Lunar NPU."""
    python_path = sys.executable
    return {
        "command": python_path,
        "args": ["-m", "lunar_core.mcp_server"],
        "env": {
            "PYTHONIOENCODING": "utf-8",
        },
        "description": "Intel Lunar Lake 47 TOPS NPU Sovereign Runtime & Silicon Circuit Breaker",
    }


def get_known_client_configs() -> Dict[str, Dict[str, Any]]:
    """Return path mapping for known client applications."""
    home = Path.home()
    appdata = Path(os.environ.get("APPDATA", home / "AppData" / "Roaming"))

    return {
        "claude": {
            "name": "Claude Desktop",
            "path": appdata / "Claude" / "claude_desktop_config.json",
            "server_key": "mcpServers",
        },
        "cursor": {
            "name": "Cursor IDE",
            "path": home / ".cursor" / "mcp.json",
            "server_key": "mcpServers",
        },
        "windsurf": {
            "name": "Windsurf",
            "path": home / ".codeium" / "windsurf" / "mcp_config.json",
            "server_key": "mcpServers",
        },
        "antigravity": {
            "name": "Antigravity",
            "path": home / ".gemini" / "antigravity" / "mcp_config.json",
            "server_key": "mcpServers",
        },
        "vscode": {
            "name": "VS Code (Cline / Roo)",
            "path": appdata / "Code" / "User" / "globalStorage" / "saoudrizwan.claude-dev" / "settings" / "cline_mcp_settings.json",
            "server_key": "mcpServers",
        },
    }


def inspect_client_status() -> List[Dict[str, Any]]:
    """Inspect installation status of Lunar MCP across all known client editors."""
    clients = get_known_client_configs()
    results = []

    for client_id, meta in clients.items():
        cfg_path = Path(meta["path"])
        installed = False
        active_config = None

        if cfg_path.exists():
            try:
                with open(cfg_path, "r", encoding="utf-8-sig") as f:
                    data = json.load(f)
                servers = data.get(meta["server_key"], {})
                installed = "lunar-npu" in servers or "intel-npu" in servers
                active_config = servers.get("lunar-npu") or servers.get("intel-npu")
            except Exception:
                pass

        results.append({
            "id": client_id,
            "name": meta["name"],
            "config_path": str(cfg_path),
            "file_exists": cfg_path.exists(),
            "lunar_registered": installed,
            "active_entry": active_config,
        })

    return results


def install_lunar_mcp(
    client_target: str = "all",
    dry_run: bool = False,
    override_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Install or update Lunar NPU MCP server definition in target client configurations.
    """
    clients = get_known_client_configs()
    server_spec = override_config or get_default_lunar_server_config()

    targets = list(clients.keys()) if client_target.lower() == "all" else [client_target.lower()]
    reports = []

    for cid in targets:
        if cid not in clients:
            reports.append({
                "client": cid,
                "status": "SKIPPED_UNKNOWN_CLIENT",
                "message": f"Client '{cid}' not recognized",
            })
            continue

        meta = clients[cid]
        cfg_path = Path(meta["path"])
        s_key = meta["server_key"]

        data: Dict[str, Any] = {}
        if cfg_path.exists():
            try:
                with open(cfg_path, "r", encoding="utf-8-sig") as f:
                    data = json.load(f)
            except Exception as e:
                reports.append({
                    "client": cid,
                    "name": meta["name"],
                    "status": "ERROR_CORRUPT_CONFIG",
                    "error": str(e),
                })
                continue

        if s_key not in data or not isinstance(data[s_key], dict):
            data[s_key] = {}

        data[s_key]["lunar-npu"] = server_spec

        if not dry_run:
            cfg_path.parent.mkdir(parents=True, exist_ok=True)
            if cfg_path.exists():
                backup_path = cfg_path.with_suffix(".json.bak")
                try:
                    shutil.copy2(cfg_path, backup_path)
                except Exception:
                    pass

            with open(cfg_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)

        reports.append({
            "client": cid,
            "name": meta["name"],
            "status": "CONFIGURED" if not dry_run else "DRY_RUN_PLANNED",
            "path": str(cfg_path),
            "registered_name": "lunar-npu",
        })

    return {
        "operation": "install_lunar_mcp",
        "dry_run": dry_run,
        "results": reports,
    }
