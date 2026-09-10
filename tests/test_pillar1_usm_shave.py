"""Unit and integration tests for Pillar 1: Level Zero USM & SHAVE DSP offload."""

import numpy as np
import pytest
from lunar_core.engine import (
    LevelZeroUSMBridge,
    SharedD3D11TextureDesc,
    SpeculativeRingBuffer,
    ShaveDSPSpectralProcessor,
)


def test_level_zero_usm_bridge_surface_creation():
    bridge = LevelZeroUSMBridge()
    desc = bridge.create_shared_d3d11_texture(width=1920, height=1080)
    assert isinstance(desc, SharedD3D11TextureDesc)
    assert desc.width == 1920
    assert desc.height == 1080
    assert desc.size_in_bytes == 1920 * 1080 * 4
    assert desc.is_zero_copy is True
    assert desc.nt_handle > 0


def test_level_zero_usm_import():
    bridge = LevelZeroUSMBridge()
    desc = bridge.create_shared_d3d11_texture(width=640, height=480)
    ov_tensor, lat_ms = bridge.import_nt_handle_to_usm(desc)
    assert ov_tensor is not None
    assert tuple(ov_tensor.shape) == (480, 640, 4)
    assert lat_ms >= 0.0


def test_level_zero_usm_benchmark():
    bridge = LevelZeroUSMBridge()
    bench = bridge.benchmark_transfer(width=1280, height=720, iterations=5)
    assert bench["zero_copy_verified"] is True
    assert bench["zero_copy_usm_latency_ms"] >= 0.0
    assert bench["speedup_factor"] >= 1.0


def test_speculative_ring_buffer_alignment_and_spsc():
    ring = SpeculativeRingBuffer(capacity=16)
    # Verify cache line alignment (alignas(64))
    assert ring.is_aligned_64 is True
    assert ring.is_empty() is True
    assert ring.available_items() == 0

    # Push draft packet
    tokens = [101, 2054, 2003, 1037]
    success = ring.push(draft_tokens=tokens)
    assert success is True
    assert ring.is_empty() is False
    assert ring.available_items() == 1

    # Pop draft packet
    packet = ring.pop()
    assert packet is not None
    assert packet.draft_tokens == tokens
    assert packet.status == "DRAFT_READY"
    assert ring.is_empty() is True

    # Test full capacity
    for i in range(16):
        assert ring.push([i, i + 1]) is True
    assert ring.is_full() is True
    assert ring.push([999]) is False  # Overrun prevented


def test_shave_dsp_spectral_processor():
    proc = ShaveDSPSpectralProcessor(n_fft=512, hop_length=160, n_mels=80, sample_rate=16000)
    # Generate 1 second synthetic audio (16,000 samples)
    t = np.linspace(0, 1.0, 16000, endpoint=False, dtype=np.float32)
    audio = 0.5 * np.sin(2 * np.pi * 440 * t) + 0.25 * np.sin(2 * np.pi * 880 * t)

    log_mel, latency_ms = proc.process_spectral_frame(audio)
    assert log_mel.shape[0] == 80  # 80 channels
    assert log_mel.shape[1] > 0
    assert latency_ms > 0.0
    assert np.isfinite(log_mel).all()
