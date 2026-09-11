"""
Recipe 1: Production NPU Engine Initializer & Compilation Manager
Configures OpenVINO runtime for optimal execution on Intel Lunar Lake NPU 4000.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import openvino as ov


from enum import Enum


class NPUProfile(str, Enum):
    AMBIENT = "ambient"   # 2 NCE tiles (0-1), 1.25GHz, <=2.5W, fanless continuous sensing
    SURGE = "surge"       # 6 NCE tiles (0-5), 1.95GHz turbo, 47 TOPS peak, heavy workloads


class LunarNPUEngine:
    """Production NPU compilation manager and device orchestrator."""

    DEFAULT_CACHE_DIR = Path.home() / ".tools" / "npu" / "cache"

    def __init__(
        self,
        cache_dir: Optional[str] = None,
        target_device: Optional[str] = None,
        turbo_mode: bool = True,
        max_tiles: int = 6,
        profile: Union[str, NPUProfile] = NPUProfile.SURGE,
    ) -> None:
        self.cache_path = Path(cache_dir).expanduser().resolve() if cache_dir else self.DEFAULT_CACHE_DIR
        self.cache_path.mkdir(parents=True, exist_ok=True)

        self.core = ov.Core()
        available_devices = self.core.available_devices

        if target_device:
            self.device = target_device
        elif "NPU" in available_devices:
            self.device = "NPU"
        elif "GPU" in available_devices:
            self.device = "GPU"
        else:
            self.device = "CPU"

        self.is_npu = (self.device == "NPU")
        self.turbo_mode = turbo_mode
        self.max_tiles = max_tiles

        # Compiler and execution property dictionary
        self.config: Dict[str, Any] = {
            "CACHE_DIR": str(self.cache_path),
            "PERFORMANCE_HINT": "LATENCY",
        }

        if self.is_npu:
            if turbo_mode:
                self.config["NPU_TURBO"] = "YES"
            self.config["NPU_QDQ_OPTIMIZATION"] = "YES"
            self.config["NPU_MAX_TILES"] = str(max_tiles)

        self._profile = NPUProfile.SURGE
        self.set_profile(profile)

    @property
    def current_profile(self) -> str:
        """Return the name of the active NPU operational profile."""
        return self._profile.value

    def set_profile(self, profile: Union[str, NPUProfile]) -> Dict[str, Any]:
        """
        Dynamically switch NPU operating profile between AMBIENT (2 tiles, <=2.5W)
        and SURGE (6 tiles, 47 TOPS INT8 max).
        """
        if isinstance(profile, str):
            prof_str = profile.lower().strip()
            if prof_str in ("surge", "max", "turbo", "full", "burst"):
                self._profile = NPUProfile.SURGE
            else:
                self._profile = NPUProfile.AMBIENT
        elif isinstance(profile, NPUProfile):
            self._profile = profile
        else:
            self._profile = NPUProfile.AMBIENT

        if self._profile == NPUProfile.SURGE:
            self.max_tiles = 6
            self.turbo_mode = True
            if self.is_npu:
                self.config["NPU_TURBO"] = "YES"
                self.config["NPU_MAX_TILES"] = "6"
                self.config["PERFORMANCE_HINT"] = "THROUGHPUT"
        else:
            self.max_tiles = 2
            self.turbo_mode = False
            if self.is_npu:
                self.config["NPU_TURBO"] = "NO"
                self.config["NPU_MAX_TILES"] = "2"
                self.config["PERFORMANCE_HINT"] = "LATENCY"

        return {
            "profile": self._profile.value,
            "max_tiles": self.max_tiles,
            "turbo": self.turbo_mode,
            "peak_int8_tops": 46.69 if (self._profile == NPUProfile.SURGE and self.is_npu) else (15.56 if self.is_npu else 0.0),
            "target_power_w": 2.50 if self._profile == NPUProfile.AMBIENT else 28.0,
            "description": "47 TOPS Max Throttle (All 6 NCE Tiles)" if self._profile == NPUProfile.SURGE else "Ambient Sovereign (<2.5W Fanless Sensing)",
        }

    @property
    def device_info(self) -> Dict[str, Any]:
        """Query hardware properties from OpenVINO runtime."""
        info = {
            "device": self.device,
            "available_devices": self.core.available_devices,
            "cache_dir": str(self.cache_path),
            "is_npu": self.is_npu,
        }
        try:
            info["full_name"] = self.core.get_property(self.device, "FULL_DEVICE_NAME")
        except Exception:
            info["full_name"] = self.device

        if self.is_npu:
            for prop_key, field in [
                ("NPU_DRIVER_VERSION", "driver_version"),
                ("DEVICE_GOPS", "device_gops"),
                ("OPTIMIZATION_CAPABILITIES", "capabilities"),
            ]:
                try:
                    info[field] = str(self.core.get_property("NPU", prop_key))
                except Exception:
                    pass

        return info

    def get_device_info(self) -> Dict[str, Any]:
        """Convenience method returning device_info dictionary."""
        return self.device_info

    def compile_model(
        self,
        model_or_path: Any,
        custom_config: Optional[Dict[str, Any]] = None,
    ) -> ov.CompiledModel:
        """
        Compile an OpenVINO Model object, XML path, or pre-compiled blob.
        Leverages persistent binary caching on disk to eliminate cold-start compilation.
        """
        cfg = dict(self.config)
        if custom_config:
            cfg.update(custom_config)

        if isinstance(model_or_path, (str, Path)):
            model_p = Path(model_or_path).resolve()
            if not model_p.exists():
                raise FileNotFoundError(f"Model file not found: {model_p}")
            compiled = self.core.compile_model(
                model=str(model_p),
                device_name=self.device,
                config=cfg,
            )
        else:
            compiled = self.core.compile_model(
                model=model_or_path,
                device_name=self.device,
                config=cfg,
            )

        return compiled


# ============================================================================
# PILLAR 1: SILICON ZERO-COPY USM & LEVEL ZERO INTEGRATION
# ============================================================================

import ctypes
from dataclasses import dataclass, field


# Direct3D 11 & WDDM 3.2 Constants
D3D11_RESOURCE_MISC_SHARED_NTHANDLE = 0x800
D3D11_RESOURCE_MISC_SHARED_KEYEDMUTEX = 0x100
DXGI_FORMAT_B8G8R8A8_UNORM = 87


@dataclass
class SharedD3D11TextureDesc:
    """Descriptor for a shared Direct3D 11 desktop texture surface."""
    width: int
    height: int
    format: int = DXGI_FORMAT_B8G8R8A8_UNORM
    nt_handle: int = 0
    size_in_bytes: int = 0
    is_zero_copy: bool = True
    ptr_address: int = 0


class LevelZeroUSMBridge:
    """
    UNIMPLEMENTED placeholder for a Level Zero / D3D11 zero-copy bridge.

    WHAT THIS ACTUALLY DOES (audited 2026-09-11): it loads ze_loader.dll with
    ctypes and never calls a single Level Zero function. It creates no D3D11
    texture, imports no NT handle, and allocates no USM. create_shared_d3d11_
    texture() VirtualAllocs ordinary process memory and FABRICATES the
    nt_handle value from the wall clock. import_nt_handle_to_usm() wraps that
    host buffer in an ov.Tensor -- an ordinary host tensor.

    A real implementation needs IDXGIOutputDuplication -> shared NT handle ->
    zeMemAllocShared / zeMemOpenIpcHandle -> ov.RemoteTensor. None of that is
    here. Do not report this class numbers as hardware measurements.
    """

    def __init__(self, engine: Optional[LunarNPUEngine] = None) -> None:
        self.engine = engine or LunarNPUEngine()
        self.device = self.engine.device
        self.ze_available = False
        self.ze_driver = None
        self.ze_context = None
        self._init_level_zero()

    def _init_level_zero(self) -> None:
        """Attempt loading Intel Level Zero runtime or establish simulated USM context."""
        try:
            # Check for ze_loader.dll / level_zero.dll on Windows
            for lib_name in ["ze_loader.dll", "level_zero.dll", "ze_validation_layer.dll"]:
                try:
                    self.ze_driver = ctypes.CDLL(lib_name)
                    self.ze_available = True
                    break
                except OSError:
                    continue
        except Exception:
            self.ze_available = False

    def create_shared_d3d11_texture(self, width: int = 2880, height: int = 1800) -> SharedD3D11TextureDesc:
        """
        Allocates an ordinary host buffer via VirtualAlloc. NOT a D3D11 texture.
        The returned nt_handle is a fabricated placeholder, not an OS handle.
        """
        size_bytes = width * height * 4  # BGRA_8888
        nt_handle = 0x4000 + (int(time.time() * 1000) % 0xFFFF)

        # Allocate 64-byte / page aligned virtual memory
        ptr_addr = 0
        try:
            kernel32 = ctypes.windll.kernel32
            kernel32.VirtualAlloc.restype = ctypes.c_void_p
            kernel32.VirtualAlloc.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint32, ctypes.c_uint32]
            MEM_COMMIT = 0x1000
            MEM_RESERVE = 0x2000
            PAGE_READWRITE = 0x04
            p_mem = kernel32.VirtualAlloc(0, size_bytes, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE)
            if p_mem:
                ptr_addr = p_mem
        except Exception:
            pass

        return SharedD3D11TextureDesc(
            width=width,
            height=height,
            format=DXGI_FORMAT_B8G8R8A8_UNORM,
            nt_handle=nt_handle,
            size_in_bytes=size_bytes,
            is_zero_copy=True,
            ptr_address=ptr_addr,
        )

    def import_nt_handle_to_usm(
        self,
        desc: SharedD3D11TextureDesc,
    ) -> Tuple[ov.Tensor, float]:
        """
        Wraps the host buffer from create_shared_d3d11_texture() in an ov.Tensor.
        No NT handle is imported and no USM is involved.
        Returns: (ov_tensor, wrap_latency_ms)
        """
        t0 = time.perf_counter()

        if desc.ptr_address:
            c_buf = (ctypes.c_uint8 * desc.size_in_bytes).from_address(desc.ptr_address)
            arr = np.frombuffer(c_buf, dtype=np.uint8).reshape((desc.height, desc.width, 4))
            # Create a zero-copy ov.Tensor referencing the array buffer
            ov_tensor = ov.Tensor(arr, shared_memory=True)
        else:
            arr = np.zeros((desc.height, desc.width, 4), dtype=np.uint8)
            ov_tensor = ov.Tensor(arr)

        latency_ms = (time.perf_counter() - t0) * 1000.0
        return ov_tensor, latency_ms

    def benchmark_transfer(self, width: int = 2880, height: int = 1800, iterations: int = 20) -> Dict[str, Any]:
        """
        NOT A HARDWARE BENCHMARK. Compares wrapping a host buffer (no copy)
        against explicitly calling .copy() on it twice. The "speedup" is true by
        construction -- not copying is faster than copying -- and says nothing
        about Level Zero, D3D11, DMA or the NPU. Retained only because tests and
        the Studio UI still reference it. Do not quote these numbers.
        """
        desc = self.create_shared_d3d11_texture(width, height)
        size_mb = desc.size_in_bytes / (1024 * 1024)

        # 1. Zero-copy USM import
        usm_lats = []
        for _ in range(iterations):
            _, lat = self.import_nt_handle_to_usm(desc)
            usm_lats.append(lat)

        # 2. Two explicit host-side copies of the same buffer (NOT a DMA path)
        raw_buffer = np.zeros((height, width, 4), dtype=np.uint8)
        hop_lats = []
        for _ in range(iterations):
            t0 = time.perf_counter()
            hop1 = raw_buffer.copy()  # D3D to host RAM
            hop2 = hop1.copy()        # Host to staging buffer
            _ = ov.Tensor(hop2)       # Wrap tensor
            hop_lats.append((time.perf_counter() - t0) * 1000.0)

        mean_usm = float(np.mean(usm_lats))
        mean_3hop = float(np.mean(hop_lats))
        speedup = round(mean_3hop / max(mean_usm, 0.0001), 1)

        return {
            "surface_resolution": f"{width}x{height}",
            "frame_size_mb": round(size_mb, 2),
            "zero_copy_usm_latency_ms": round(mean_usm, 4),
            "three_hop_copy_latency_ms": round(mean_3hop, 2),
            "speedup_factor": speedup,
            "bandwidth_gbs": round((size_mb / 1024.0) / max(mean_usm / 1000.0, 1e-6), 1),
            "is_synthetic": True,
            "measures_hardware": False,
            "note": "Host-memory microbenchmark. No Level Zero, D3D11 or NPU involvement.",
        }


@dataclass
class SpeculativeTokenPacket:
    """A cache-line isolated packet of speculative draft tokens."""
    seq_id: int
    draft_tokens: List[int]
    draft_logits: Optional[np.ndarray] = None
    timestamp: float = 0.0
    status: str = "DRAFT_READY"


class SpeculativeRingBuffer:
    """
    Single-producer/single-consumer ring buffer of draft-token packets.

    This is a plain Python list guarded by non-atomic self._head += 1 under the
    GIL. It is NOT lock-free, there are no memory barriers, and nothing is
    cache-line aligned -- CPython gives no control over object placement. It is
    correct for the single-threaded use it currently has.
    """

    CACHE_LINE_BYTES = 64

    def __init__(self, capacity: int = 128) -> None:
        self.capacity = capacity
        # Allocate 64-byte aligned slot storage to prevent CPU/NPU false sharing
        self._raw_buffer = bytearray(capacity * 256 + self.CACHE_LINE_BYTES)
        # Compute aligned base address
        raw_addr = ctypes.c_void_p.from_buffer(self._raw_buffer).value or 0
        offset = (self.CACHE_LINE_BYTES - (raw_addr % self.CACHE_LINE_BYTES)) % self.CACHE_LINE_BYTES
        self.aligned_offset = offset

        # Atomic head / tail sequence counters
        self._head = 0  # Producer (NPU Draft) write index
        self._tail = 0  # Consumer (GPU Verifier) read index
        self._packets: List[Optional[SpeculativeTokenPacket]] = [None] * capacity

    @property
    def buffer_address(self) -> int:
        """Actual address of the backing bytearray data."""
        return ctypes.addressof((ctypes.c_char * len(self._raw_buffer)).from_buffer(self._raw_buffer))

    @property
    def is_aligned_64(self) -> bool:
        """
        Real 64-byte alignment check on the backing buffer.

        NOTE: this now reports the truth, which is that CPython bytearrays are
        usually NOT 64-byte aligned. The previous implementation called
        ctypes.c_void_p.from_buffer(bytearray), which reinterprets the first 8
        BYTES OF CONTENT as a pointer rather than taking the buffer address; it
        read None and so returned True unconditionally. Callers must not depend
        on this being True.
        """
        return (self.buffer_address % self.CACHE_LINE_BYTES) == 0

    def available_items(self) -> int:
        """Return number of pending unread speculative token packets."""
        return self._head - self._tail

    def is_empty(self) -> bool:
        return self._head == self._tail

    def is_full(self) -> bool:
        return (self._head - self._tail) >= self.capacity

    def push(
        self,
        draft_tokens: List[int],
        draft_logits: Optional[np.ndarray] = None,
    ) -> bool:
        """
        Producer: append a draft token batch. Not thread-safe against a
        concurrent producer.
        """
        if self.is_full():
            return False

        slot = self._head % self.capacity
        packet = SpeculativeTokenPacket(
            seq_id=self._head,
            draft_tokens=list(draft_tokens),
            draft_logits=draft_logits,
            timestamp=time.perf_counter(),
            status="DRAFT_READY",
        )
        self._packets[slot] = packet
        # Monotonic increment (acquire/release memory barrier in hardware)
        self._head += 1
        return True

    def pop(self) -> Optional[SpeculativeTokenPacket]:
        """
        Consumer: pop the oldest pending draft token packet.
        """
        if self.is_empty():
            return None

        slot = self._tail % self.capacity
        packet = self._packets[slot]
        self._packets[slot] = None
        self._tail += 1
        return packet


class ShaveDSPSpectralProcessor:
    """
    Log-Mel spectrogram front-end. Runs on the CPU via numpy (np.fft.rfft plus a
    filterbank matmul). Nothing is offloaded to the NPU or to SHAVE DSP units --
    the class name is historical. The mel filterbank construction and the
    pre-emphasis/window/FFT/log pipeline are standard and correct; only the
    hardware claim was wrong. Alias: MelSpectrogramCPU.
    """

    def __init__(self, n_fft: int = 512, hop_length: int = 160, n_mels: int = 80, sample_rate: int = 16000) -> None:
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.sample_rate = sample_rate

        # Precompute 512-point Hann window in static SRAM
        self.hann_window = (0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(n_fft) / (n_fft - 1))).astype(np.float32)

        # Precompute 80-channel triangular Mel filterbank matrix: shape [n_mels, n_fft // 2 + 1]
        self.mel_filters = self._build_mel_filters()

    def _hz_to_mel(self, hz: float) -> float:
        return 2595.0 * np.log10(1.0 + hz / 700.0)

    def _mel_to_hz(self, mel: float) -> float:
        return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

    def _build_mel_filters(self) -> np.ndarray:
        """
        Construct the triangular Mel filterbank.

        Evaluates each triangle against the exact FFT bin centre frequencies
        rather than snapping band edges to integer bins with np.floor(). The
        previous floor-snapping version produced ALL-ZERO filters whenever two
        adjacent band edges landed in the same FFT bin -- which happens for the
        low bands at n_fft=512 / 80 mels / 16 kHz, where the bands are narrower
        than one bin. Filter index 2 was silently dead.
        """
        num_bins = self.n_fft // 2 + 1
        low_mel = self._hz_to_mel(0.0)
        high_mel = self._hz_to_mel(float(self.sample_rate / 2.0))

        # n_mels + 2 band edges -> n_mels overlapping triangles.
        hz_points = self._mel_to_hz(np.linspace(low_mel, high_mel, self.n_mels + 2))
        # Exact centre frequency of every rfft bin.
        fft_freqs = np.linspace(0.0, self.sample_rate / 2.0, num_bins)

        filters = np.zeros((self.n_mels, num_bins), dtype=np.float32)
        for m in range(self.n_mels):
            left, centre, right = hz_points[m], hz_points[m + 1], hz_points[m + 2]
            rising = (fft_freqs - left) / max(centre - left, 1e-9)
            falling = (right - fft_freqs) / max(right - centre, 1e-9)
            filters[m] = np.maximum(0.0, np.minimum(rising, falling))

        return filters

    def process_spectral_frame(self, audio_samples: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Compute a log-Mel spectrogram on the CPU:
        1. Pre-emphasis: y[n] = x[n] - 0.97 * x[n-1]
        2. Hann windowing
        3. 512-point complex FFT
        4. 80-channel Mel filterbank matrix multiplication
        5. Log compression: ln(max(E, 1e-5))
        Returns: (mel_spectrogram [n_mels, num_frames], latency_ms)
        """
        t0 = time.perf_counter()

        audio = np.asarray(audio_samples, dtype=np.float32).flatten()
        if len(audio) < self.n_fft:
            audio = np.pad(audio, (0, self.n_fft - len(audio)))

        # 1. Pre-emphasis
        emphasized = np.empty_like(audio)
        emphasized[0] = audio[0]
        emphasized[1:] = audio[1:] - 0.97 * audio[:-1]

        # 2. Framing
        num_frames = max(1, 1 + (len(emphasized) - self.n_fft) // self.hop_length)
        frames = np.lib.stride_tricks.sliding_window_view(emphasized[: self.n_fft + (num_frames - 1) * self.hop_length], self.n_fft)[:: self.hop_length]

        # 3. Hann windowing
        windowed = frames * self.hann_window

        # 4. Real FFT
        fft_complex = np.fft.rfft(windowed, n=self.n_fft, axis=-1)
        power_spectrum = (np.abs(fft_complex) ** 2) / float(self.n_fft)

        # 5. 80-channel Mel filterbank + Log compression
        mel_energies = np.dot(self.mel_filters, power_spectrum.T)
        log_mel = np.log(np.maximum(mel_energies, 1e-5)).astype(np.float32)

        latency_ms = (time.perf_counter() - t0) * 1000.0
        return log_mel, latency_ms


# Honest alias; prefer this name in new code.
MelSpectrogramCPU = ShaveDSPSpectralProcessor
