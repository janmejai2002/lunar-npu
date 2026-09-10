"""
Recipe 10: LunarGit — Semantic Git Time-Machine & Intent-Based Commit Memory
===========================================================================
Indexes local Git repository history onto the Intel Lunar Lake NPU's S^383 unit
hypersphere, enabling sub-3ms natural language search across historical commits,
PR descriptions, and bug fixes without cloud dependencies.
"""

from __future__ import annotations

import os
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from lunar_core.engine import LunarNPUEngine
from lunar_core.vector_memory import LunarVectorMemory

DEFAULT_GIT_MEMORY_PATH = Path(".lunar_git_memory.json")


@dataclass
class CommitSearchResult:
    commit_hash: str
    author: str
    date: str
    message: str
    similarity_score: float
    files_changed: List[str]
    latency_ms: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class GitTimeMachine:
    """
    Semantic search engine for local Git repository history accelerated on Intel NPU.
    """

    def __init__(
        self,
        engine: Optional[LunarNPUEngine] = None,
        memory: Optional[LunarVectorMemory] = None,
        repo_root: Optional[Path] = None,
    ):
        self.engine = engine or LunarNPUEngine()
        self.memory = memory or LunarVectorMemory(engine=self.engine)
        self.repo_root = Path(repo_root or Path.cwd())

        if DEFAULT_GIT_MEMORY_PATH.exists():
            try:
                self.memory.load_from_disk(DEFAULT_GIT_MEMORY_PATH)
            except Exception:
                pass

    def index_repository(self, max_commits: int = 50) -> int:
        """
        Parse Git commit log and index each commit into S^383 vector memory.
        Returns the number of commits newly indexed.
        """
        try:
            # git log format: %H|%an|%ad|%s
            cmd = [
                "git", "log", f"-n{max_commits}",
                "--pretty=format:%H|%an|%ad|%s",
                "--date=iso",
            ]
            raw_log = subprocess.check_output(cmd, cwd=str(self.repo_root), text=True, errors="replace")
        except Exception:
            return 0

        lines = [line.strip() for line in raw_log.split("\n") if line.strip()]
        indexed_count = 0

        for line in lines:
            parts = line.split("|", 3)
            if len(parts) < 4:
                continue
            commit_hash, author, date_str, subject = parts

            # Get files changed in this commit
            files_changed = []
            try:
                files_out = subprocess.check_output(
                    ["git", "diff-tree", "--no-commit-id", "--name-only", "-r", commit_hash],
                    cwd=str(self.repo_root),
                    text=True,
                    errors="replace",
                )
                files_changed = [f.strip() for f in files_out.split("\n") if f.strip()]
            except Exception:
                pass

            doc_text = f"Git Commit {commit_hash[:8]}: {subject}. Changed: {', '.join(files_changed[:5])}"
            metadata = {
                "type": "git_commit",
                "commit_hash": commit_hash,
                "author": author,
                "date": date_str,
                "subject": subject,
                "files_changed": files_changed[:10],
            }

            self.memory.add_document(
                text=doc_text,
                metadata=metadata,
                doc_id=f"git_{commit_hash[:10]}",
            )
            indexed_count += 1

        if indexed_count > 0:
            self.memory.save_to_disk(DEFAULT_GIT_MEMORY_PATH)

        return indexed_count

    def search(self, query: str, top_k: int = 5) -> List[CommitSearchResult]:
        """
        Query repository commits semantically using NPU dot product on S^383.
        """
        t0 = time.perf_counter()
        results = self.memory.query(query, top_k=top_k)
        elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 2)

        search_results: List[CommitSearchResult] = []
        for r in results:
            meta = r.get("metadata", {})
            if meta.get("type") == "git_commit" or "commit_hash" in meta:
                search_results.append(
                    CommitSearchResult(
                        commit_hash=meta.get("commit_hash", r["id"]),
                        author=meta.get("author", "Unknown"),
                        date=meta.get("date", "N/A"),
                        message=meta.get("subject", r["text"]),
                        similarity_score=round(r["score"], 4),
                        files_changed=meta.get("files_changed", []),
                        latency_ms=elapsed_ms,
                    )
                )

        return search_results
