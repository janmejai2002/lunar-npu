"""
Hardware Test Rig Configuration & Thresholds for Intel Lunar Lake (Core Ultra Series 2).
Defines performance budgets, model architectures, memory budgets, and driver parameters.
"""

from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, Any, List

# Workspace and tool directories
RIG_DIR = Path(__file__).resolve().parent
WORKSPACE_DIR = RIG_DIR.parent.parent
NPU_TOOL_DIR = Path("C:/Users/Janmejai/.tools/npu")

# Model & Cache Storage
CACHE_DIR = NPU_TOOL_DIR / "cache" if NPU_TOOL_DIR.exists() else WORKSPACE_DIR / "cache"
MODELS_DIR = NPU_TOOL_DIR / "models" if NPU_TOOL_DIR.exists() else WORKSPACE_DIR / "models"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Hardware Platform Identifiers
TARGET_DEVICE = "NPU"
FALLBACK_DEVICES = ["GPU", "CPU"]
LUNAR_LAKE_NPU_NAME = "Intel(R) AI Boost"
LUNAR_LAKE_ARCH = "4000"
LUNAR_LAKE_MAX_TILES = 6
LUNAR_LAKE_TOP_INT8_TOPS = 47.0


@dataclass(frozen=True)
class CompilationThresholds:
    """Thresholds for NPU compilation and persistent blob caching."""
    cold_compile_budget_ms: float = 8000.0   # Cold compilation can take several seconds
    cache_hit_budget_ms: float = 25.0       # Hard gate: warm cache load must be sub-25ms
    min_blob_size_bytes: int = 100 * 1024   # Valid NPU blob is at least 100KB (e.g. Mamba/MiniLM)
    target_cache_mode: str = "OPTIMIZE_SPEED"


@dataclass(frozen=True)
class LatencyThresholds:
    """Microsecond latency thresholds for Lunar Lake NPU 4000."""
    # FastMiniLM: 6-layer transformer encoder, seq_len=128
    fastminilm_p50_ms: float = 2.20
    fastminilm_p95_ms: float = 2.80
    fastminilm_hard_limit_ms: float = 3.00

    # MobileNetV4 / Vision: 224x224x3 edge perception
    mobilenet_p50_ms: float = 0.95
    mobilenet_p95_ms: float = 1.15
    mobilenet_hard_limit_ms: float = 1.20   # Hard limit corresponds to >833 FPS
    mobilenet_target_fps: float = 857.0     # Target: 857 FPS (1.166 ms)

    # Moonshine / Whisper ASR: 30-second speech acoustic encoder (80x3000)
    asr_p50_ms: float = 11.50
    asr_p95_ms: float = 14.00
    asr_hard_limit_ms: float = 15.00
    asr_min_real_time_factor: float = 2000.0  # 30s processed in <15ms -> >2000x RTF


@dataclass(frozen=True)
class SoakTestThresholds:
    """Memory stability budgets across 10,000 continuous inference iterations."""
    total_iterations: int = 10000
    checkpoint_interval: int = 1000
    max_working_set_growth_mb: float = 5.0    # Max permissible growth in Working Set
    max_private_bytes_growth_mb: float = 5.0  # Max permissible growth in Private Bytes
    max_leak_rate_kb_per_iter: float = 0.5    # Unbounded leak detection slope
    catastrophic_leak_limit_mb: float = 15.0


@dataclass(frozen=True)
class IngestTestThresholds:
    """Thresholds for synthetic audio and vision media ingest pipelines."""
    # Audio WASAPI loopback test
    audio_sample_rate_hz: int = 16000
    audio_frame_size_ms: int = 20
    audio_max_wer_percent: float = 5.0
    audio_ring_buffer_capacity_sec: float = 10.0

    # D3D11 Vision frame capture test
    d3d11_target_fps: float = 857.0
    d3d11_frame_budget_us: float = 1166.6  # 1 / 857 fps in microseconds
    d3d11_max_dropped_frames: int = 0
    d3d11_buffer_count: int = 3            # Triple-buffering pipeline


# Global Threshold Instances
COMPILATION_CONFIG = CompilationThresholds()
LATENCY_CONFIG = LatencyThresholds()
SOAK_CONFIG = SoakTestThresholds()
INGEST_CONFIG = IngestTestThresholds()

# Model Model Specifications
MODEL_SPECS: Dict[str, Dict[str, Any]] = {
    "fastminilm": {
        "xml_path": MODELS_DIR / "embed" / "minilm_l6.xml",
        "bin_path": MODELS_DIR / "embed" / "minilm_l6.bin",
        "input_shape": [1, 128],
        "input_dtype": "int64",
        "output_shape": [1, 384],
        "thresholds": {
            "p50_ms": LATENCY_CONFIG.fastminilm_p50_ms,
            "p95_ms": LATENCY_CONFIG.fastminilm_p95_ms,
            "limit_ms": LATENCY_CONFIG.fastminilm_hard_limit_ms,
        }
    },
    "mobilenet": {
        "xml_path": MODELS_DIR / "vision" / "mobilenet_v3.xml",
        "bin_path": MODELS_DIR / "vision" / "mobilenet_v3.bin",
        "input_shape": [1, 3, 224, 224],
        "input_dtype": "float32",
        "output_shape": [1, 1000],
        "thresholds": {
            "p50_ms": LATENCY_CONFIG.mobilenet_p50_ms,
            "p95_ms": LATENCY_CONFIG.mobilenet_p95_ms,
            "limit_ms": LATENCY_CONFIG.mobilenet_hard_limit_ms,
            "target_fps": LATENCY_CONFIG.mobilenet_target_fps,
        }
    },
    "moonshine_asr": {
        "xml_path": MODELS_DIR / "whisper" / "whisper_encoder.xml",
        "bin_path": MODELS_DIR / "whisper" / "whisper_encoder.bin",
        "input_shape": [1, 80, 3000],
        "input_dtype": "float32",
        "output_shape": [1, 1500, 384],
        "thresholds": {
            "p50_ms": LATENCY_CONFIG.asr_p50_ms,
            "p95_ms": LATENCY_CONFIG.asr_p95_ms,
            "limit_ms": LATENCY_CONFIG.asr_hard_limit_ms,
            "min_rtf": LATENCY_CONFIG.asr_min_real_time_factor,
        }
    }
}
