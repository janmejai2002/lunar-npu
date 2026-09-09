"""
Intel Lunar Lake Hardware Test Rig Package.
Contains automated compilation verification, Level Zero driver recovery,
microsecond latency regression suites, synthetic audio/D3D11 ingest harnesses,
memory soak testing, and autonomous agent TDD loops.
"""

from src.hardware_rig.config import (
    TARGET_DEVICE,
    CACHE_DIR,
    MODELS_DIR,
    COMPILATION_CONFIG,
    LATENCY_CONFIG,
    SOAK_CONFIG,
    INGEST_CONFIG,
)
from src.hardware_rig.compilation_rig import NPUCompilationRig
from src.hardware_rig.recovery_manager import NPUHardwareRecoveryManager, CircuitState
from src.hardware_rig.latency_suite import NPULatencySuite, BenchmarkResult
from src.hardware_rig.audio_ingest_harness import (
    AudioLoopbackRingBuffer,
    WasapiSyntheticAudioFeeder,
    calculate_wer,
)
from src.hardware_rig.d3d11_capture_harness import (
    SyntheticD3D11SurfaceGenerator,
    LevelZeroUSMFrameHarness,
    D3D11SurfaceStats,
)
from src.hardware_rig.memory_soak_test import (
    WindowsMemoryProfiler,
    NPUMemoryLeakSoakTest,
    SoakTestReport,
)
from src.hardware_rig.tdd_orchestrator import (
    HardwareTDDOrchestrator,
    TDDPhase,
    TDDLoopReport,
)

__all__ = [
    "TARGET_DEVICE",
    "CACHE_DIR",
    "MODELS_DIR",
    "COMPILATION_CONFIG",
    "LATENCY_CONFIG",
    "SOAK_CONFIG",
    "INGEST_CONFIG",
    "NPUCompilationRig",
    "NPUHardwareRecoveryManager",
    "CircuitState",
    "NPULatencySuite",
    "BenchmarkResult",
    "AudioLoopbackRingBuffer",
    "WasapiSyntheticAudioFeeder",
    "calculate_wer",
    "SyntheticD3D11SurfaceGenerator",
    "LevelZeroUSMFrameHarness",
    "D3D11SurfaceStats",
    "WindowsMemoryProfiler",
    "NPUMemoryLeakSoakTest",
    "SoakTestReport",
    "HardwareTDDOrchestrator",
    "TDDPhase",
    "TDDLoopReport",
]
