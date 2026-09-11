from lunar_core.engine import LunarNPUEngine
from lunar_core.vector_memory import LunarVectorMemory
from lunar_core.circuit_breaker import SiliconCircuitBreaker
from lunar_core.mamba_ssm import LunarMambaEngine
from lunar_core.speculative import LunarSpeculativePipeline
from lunar_core.bug_ledger import BugLedgerEngine
from lunar_core.router import MicroRouter, RouteDecision
from lunar_core.swarm import LunarSwarm, SwarmResult
from lunar_core.vision import LunarVisionEngine, VisionAnalysisResult, UIElement
from lunar_core.audio import LunarAudioEngine, TranscriptionResult
from lunar_core.git_time_machine import GitTimeMachine, CommitSearchResult

from lunar_core.dxgi_capture import DXGICaptureEngine
from lunar_core.diffusion import LunarHeterogeneousDiffusion
from lunar_core.hdc import BinaryHypervector, HyperdimensionalMemoryEngine

__all__ = [
    'LunarNPUEngine',
    'LunarVectorMemory',
    'SiliconCircuitBreaker',
    'LunarMambaEngine',
    'LunarSpeculativePipeline',
    'BugLedgerEngine',
    'MicroRouter',
    'RouteDecision',
    'LunarSwarm',
    'SwarmResult',
    'LunarVisionEngine',
    'VisionAnalysisResult',
    'UIElement',
    'LunarAudioEngine',
    'TranscriptionResult',
    'GitTimeMachine',
    'CommitSearchResult',
    'DXGICaptureEngine',
    'LunarHeterogeneousDiffusion',
    'HyperdimensionalMemoryEngine',
    'BinaryHypervector',
    '__version__',
]

__version__ = '2.3.0'


from .srnc import SiliconReflexNeuralCompiler, RowanCSTBridge, CassowarySimplexLayoutSolver, ApcaOklchConvexOptimizer, WindowsNamedRingBufferShm
