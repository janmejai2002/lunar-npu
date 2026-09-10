"""Unit and integration tests for LunarSwarm autonomous multi-agent execution."""

import json
from pathlib import Path
import pytest
from click.testing import CliRunner

from lunar_core.swarm import LunarSwarm, SwarmResult, PERSONA_SYSTEM_PROMPTS
from lunar_core.cli import cli


def test_swarm_initialization():
    swarm = LunarSwarm()
    assert swarm.engine is not None
    assert swarm.memory is not None
    assert swarm.router is not None
    assert swarm.circuit_breaker is not None
    assert "CODER" in PERSONA_SYSTEM_PROMPTS
    assert "SECURITY_AUDITOR" in PERSONA_SYSTEM_PROMPTS
    assert "TESTER_DEVOPS" in PERSONA_SYSTEM_PROMPTS
    assert "ARCHITECT" in PERSONA_SYSTEM_PROMPTS


def test_swarm_execute_coding_task():
    swarm = LunarSwarm()
    prompt = "Write a python function to compute the fibonacci sequence"
    res = swarm.execute_task(prompt, max_tokens=64)

    assert isinstance(res, SwarmResult)
    assert res.task == prompt
    assert res.lead_persona == "CODER"
    assert res.route_confidence > 0.3
    assert res.total_latency_ms > 0.0
    assert len(res.stages) >= 4
    assert res.safety_decision in ("ALLOWED", "BLOCKED")
    assert res.memory_doc_id is not None

    d = res.to_dict()
    assert "stages" in d
    assert "lead_persona" in d
    assert d["lead_persona"] == "CODER"


def test_swarm_execute_security_task():
    swarm = LunarSwarm()
    prompt = "Audit this application for sql injection and credential exposure in production database"
    res = swarm.execute_task(prompt, max_tokens=64)

    assert isinstance(res, SwarmResult)
    assert res.lead_persona == "SECURITY_AUDITOR"
    assert res.route_confidence > 0.3
    assert len(res.stages) >= 4


def test_swarm_fallback_generation():
    swarm = LunarSwarm()
    coder_out = swarm._fallback_generation("CODER", "implement quicksort")
    assert "def solve_task" in coder_out or "quicksort" in coder_out

    devops_out = swarm._fallback_generation("TESTER_DEVOPS", "test user login")
    assert "pytest" in devops_out

    sec_out = swarm._fallback_generation("SECURITY_AUDITOR", "check vulnerabilities")
    assert "Security Audit" in sec_out


def test_swarm_cli():
    runner = CliRunner()
    result = runner.invoke(cli, ["swarm", "Implement an LRU cache in python", "--max-tokens", "32", "--json"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert data["lead_persona"] == "CODER"
    assert "stages" in data
    assert len(data["stages"]) >= 4
    assert data["safety_decision"] in ("ALLOWED", "BLOCKED")
