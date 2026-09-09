"""
Antigravity Lifecycle Hooks for Project Lunar NPU
==================================================
Hardware-gated pre-execution safety gates and post-execution memory indexing.
"""

from .circuit_breaker_hook import run_circuit_breaker_hook
from .memory_indexer_hook import run_memory_indexer_hook

__all__ = ["run_circuit_breaker_hook", "run_memory_indexer_hook"]
