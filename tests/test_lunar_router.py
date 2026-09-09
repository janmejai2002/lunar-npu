"""Unit and integration tests for MicroRouter task classification on Intel Lunar Lake NPU."""

import pytest
from lunar_core.router import MicroRouter, RouteDecision


def test_router_initialization_and_centroids():
    router = MicroRouter()
    assert "CODER" in router.centroids
    assert "ARCHITECT" in router.centroids
    assert "TESTER_DEVOPS" in router.centroids
    assert "RESEARCHER" in router.centroids
    assert "SECURITY_AUDITOR" in router.centroids
    assert router.centroids["CODER"].shape[0] in (384, 768)




def test_router_coder_classification():
    router = MicroRouter()
    prompt = "Write a python function to implement binary search over sorted arrays"
    decision = router.route(prompt)
    assert isinstance(decision, RouteDecision)
    assert decision.target_agent == "CODER"
    assert decision.confidence > 0.3
    assert decision.latency_ms > 0.0
    assert "CODER" in decision.scores


def test_router_devops_classification():
    router = MicroRouter()
    prompt = "Configure GitHub Actions CI workflow to run pytest tests across windows and ubuntu matrix"
    decision = router.route(prompt)
    assert decision.target_agent == "TESTER_DEVOPS"
    assert decision.confidence > 0.3


def test_router_security_classification():
    router = MicroRouter()
    prompt = "Scan bash scripts for malicious command injection rm -rf / and privilege escalation exploits"
    decision = router.route(prompt)
    assert decision.target_agent == "SECURITY_AUDITOR"
    assert decision.confidence > 0.3


def test_router_custom_archetype():
    router = MicroRouter()
    router.register_archetype("UI_DESIGNER", [
        "Create responsive Tailwind CSS layouts Figma tokens dark theme glassmorphism typography",
        "Design visual UI UX components color palette animations micro-interactions layout styling",
    ])
    assert "UI_DESIGNER" in router.centroids

    prompt = "Create a modern dark mode dashboard layout with responsive Tailwind CSS components"
    decision = router.route(prompt)
    assert decision.target_agent == "UI_DESIGNER"
