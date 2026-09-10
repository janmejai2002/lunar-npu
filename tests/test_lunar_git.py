"""Unit and integration tests for GitTimeMachine semantic repository search."""

import json
from pathlib import Path
import pytest
from click.testing import CliRunner

from lunar_core.git_time_machine import GitTimeMachine, CommitSearchResult
from lunar_core.cli import cli


def test_git_time_machine_initialization():
    gtm = GitTimeMachine()
    assert gtm.engine is not None
    assert gtm.memory is not None


def test_git_time_machine_index_and_search():
    gtm = GitTimeMachine()
    indexed_count = gtm.index_repository(max_commits=15)
    assert indexed_count > 0

    results = gtm.search("systolic saturation", top_k=3)
    assert isinstance(results, list)
    assert len(results) > 0
    first = results[0]
    assert isinstance(first, CommitSearchResult)
    assert first.commit_hash is not None
    assert first.message is not None
    assert first.similarity_score > 0.0

    d = first.to_dict()
    assert "commit_hash" in d
    assert "message" in d
    assert "similarity_score" in d


def test_git_cli_index_and_search():
    runner = CliRunner()
    res_idx = runner.invoke(cli, ["git-index", "--max-commits", "10", "--json"])
    assert res_idx.exit_code == 0
    data_idx = json.loads(res_idx.output)
    assert data_idx["commits_indexed"] > 0

    res_srch = runner.invoke(cli, ["git-search", "circuit breaker", "--top-k", "3", "--json"])
    assert res_srch.exit_code == 0
    data_srch = json.loads(res_srch.output)
    assert isinstance(data_srch, list)
    assert len(data_srch) > 0
    assert "commit_hash" in data_srch[0]
    assert "similarity_score" in data_srch[0]
