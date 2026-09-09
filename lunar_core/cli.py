"""
Unified Command-Line Interface for Lunar Project Core
Enables testing, benchmarking, embedding, auditing, and bug ledger tracking from terminal.
"""

from __future__ import annotations

import json
import sys
from typing import Any, Optional

import click
import numpy as np

if sys.platform == "win32":
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from lunar_core.bug_ledger import BugLedgerEngine
from lunar_core.circuit_breaker import SiliconCircuitBreaker
from lunar_core.engine import LunarNPUEngine
from lunar_core.mamba_ssm import LunarMambaEngine
from lunar_core.speculative import LunarSpeculativePipeline
from lunar_core.vector_memory import LunarVectorMemory


def _output(data: Any, json_mode: bool):
    if json_mode:
        click.echo(json.dumps(data, indent=2))
    else:
        if isinstance(data, dict):
            for k, v in data.items():
                click.echo(f"  {k:<24}: {v}")
        elif isinstance(data, list):
            for item in data:
                click.echo(f"  • {item}")
        else:
            click.echo(str(data))


@click.group()
@click.option("--json", "json_mode", is_flag=True, help="Output machine-readable JSON")
@click.pass_context
def cli(ctx: click.Context, json_mode: bool):
    """Lunar Core CLI — Intel Lunar Lake NPU Ambient & Speculative Platform."""
    ctx.ensure_object(dict)
    ctx.obj["json_mode"] = json_mode
    ctx.obj["engine"] = LunarNPUEngine()


@cli.command("status")
@click.option("--json", "json_mode", is_flag=True, help="Output machine-readable JSON")
@click.pass_context
def status(ctx: click.Context, json_mode: bool):
    """Inspect NPU silicon hardware, driver version, and compiler properties."""
    engine: LunarNPUEngine = ctx.obj["engine"]
    json_mode = json_mode or ctx.obj.get("json_mode", False)
    info = engine.device_info
    if json_mode:
        click.echo(json.dumps(info, indent=2))
    else:
        click.echo("=" * 64)
        click.echo(" LUNAR LAKE NPU SILICON HARDWARE STATUS")
        click.echo("=" * 64)
        for k, v in info.items():
            click.echo(f"  {k:<24}: {v}")
        click.echo("=" * 64)


@cli.command("embed")
@click.argument("text")
@click.option("--json", "json_mode", is_flag=True, help="Output machine-readable JSON")
@click.pass_context
def embed(ctx: click.Context, text: str, json_mode: bool):
    """Compute dense semantic vector embedding on Intel NPU."""
    engine: LunarNPUEngine = ctx.obj["engine"]
    json_mode = json_mode or ctx.obj.get("json_mode", False)
    vmem = LunarVectorMemory(engine=engine)
    vec, lat = vmem.embed(text)

    res = {
        "text": text,
        "dimension": int(vec.shape[0]),
        "l2_norm": float(np.linalg.norm(vec)),
        "latency_ms": lat,
        "device": engine.device,
    }
    _output(res, json_mode)


@cli.command("mamba")
@click.option("--steps", default=100, help="Number of recurrent steps to benchmark")
@click.option("--json", "json_mode", is_flag=True, help="Output machine-readable JSON")
@click.pass_context
def mamba_bench(ctx: click.Context, steps: int, json_mode: bool):
    """Benchmark Mamba SSM recurrent step latency and throughput."""
    engine: LunarNPUEngine = ctx.obj["engine"]
    json_mode = json_mode or ctx.obj.get("json_mode", False)
    mamba = LunarMambaEngine(engine=engine)
    results = mamba.benchmark(num_steps=steps)
    _output(results, json_mode)


@cli.command("speculative")
@click.option("--gamma", default=4, help="Number of speculative draft tokens")
@click.option("--json", "json_mode", is_flag=True, help="Output machine-readable JSON")
@click.pass_context
def speculative_cycle(ctx: click.Context, gamma: int, json_mode: bool):
    """Run dual-accelerator speculative decoding cycle."""
    engine: LunarNPUEngine = ctx.obj["engine"]
    json_mode = json_mode or ctx.obj.get("json_mode", False)
    spec = LunarSpeculativePipeline(draft_engine=engine, gamma=gamma)
    prefix = [101, 2054, 2003, 1037, 3231]
    res = spec.speculative_cycle(prefix_tokens=prefix)
    _output(res, json_mode)


@cli.command("audit")
@click.argument("command_str")
@click.option("--json", "json_mode", is_flag=True, help="Output machine-readable JSON")
@click.pass_context
def audit_command(ctx: click.Context, command_str: str, json_mode: bool):
    """Audit shell command using Silicon Circuit Breaker."""
    engine: LunarNPUEngine = ctx.obj["engine"]
    json_mode = json_mode or ctx.obj.get("json_mode", False)
    cb = SiliconCircuitBreaker(engine=engine)
    res = cb.audit_command(command_str)
    _output(res, json_mode)


@cli.group("bugs")
def bugs_group():
    """Manage defect ledger (docs/BUG_LEDGER.md)."""
    pass


@bugs_group.command("list")
@click.option("--status", default=None, help="Filter by status (OPEN, VERIFIED_FIX, etc.)")
@click.option("--json", "json_mode", is_flag=True, help="Output machine-readable JSON")
@click.pass_context
def bugs_list(ctx: click.Context, status: Optional[str], json_mode: bool):
    """List entries from docs/BUG_LEDGER.md."""
    json_mode = json_mode or ctx.obj.get("json_mode", False)
    ledger = BugLedgerEngine()
    entries = ledger.read_entries()
    if status:
        entries = [e for e in entries if status.lower() in e.get("status", "").lower()]

    if json_mode:
        click.echo(json.dumps(entries, indent=2))
    else:
        click.echo(f"Bug Ledger Entries ({len(entries)} total):")
        for e in entries:
            click.echo(f"  [{e['id']}] ({e['status']}) {e['summary']} - Assigned: {e['specialist']}")


def main():
    cli()


if __name__ == "__main__":
    main()
