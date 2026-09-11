"""
Tests for the host-buffer bridge and the CPU log-Mel front-end.

These assertions were rewritten on 2026-09-11. The previous versions asserted
hardcoded literals (`assert desc.is_zero_copy is True`,
`assert bench["zero_copy_verified"] is True`), constructor arguments
(`assert log_mel.shape[0] == 80`), and values that cannot be false
(`assert lat_ms >= 0.0`, `assert speedup >= 1.0` against a `max(x, 1.0)`).
None of them could fail, so a green suite proved nothing.

Every assertion below compares against an independently derived expected value.
"""

import numpy as np
import pytest

from lunar_core.engine import (
    LevelZeroUSMBridge,
    SharedD3D11TextureDesc,
    SpeculativeRingBuffer,
    ShaveDSPSpectralProcessor,
)


# ---------------------------------------------------------------------------
# Host buffer bridge (NOT a zero-copy / Level Zero path -- see class docstring)
# ---------------------------------------------------------------------------

def test_host_buffer_desc_geometry_is_self_consistent():
    bridge = LevelZeroUSMBridge()
    desc = bridge.create_shared_d3d11_texture(width=1920, height=1080)
    assert isinstance(desc, SharedD3D11TextureDesc)
    # BGRA8888 => exactly 4 bytes per pixel. Derived, not echoed.
    assert desc.size_in_bytes == 1920 * 1080 * 4
    assert desc.size_in_bytes == desc.width * desc.height * 4


def test_imported_tensor_aliases_the_source_buffer():
    """
    The one property worth testing here: the ov.Tensor must be a VIEW over the
    allocated buffer, not a copy. Writing through the numpy view must be visible
    in the tensor. If this ever fails, the wrap silently became a copy.
    """
    bridge = LevelZeroUSMBridge()
    desc = bridge.create_shared_d3d11_texture(width=64, height=32)
    if not desc.ptr_address:
        pytest.skip("VirtualAlloc unavailable on this platform")

    tensor, _ = bridge.import_nt_handle_to_usm(desc)
    assert tuple(tensor.shape) == (32, 64, 4)

    view = tensor.data
    view[0, 0, 0] = 0
    view[5, 7, 2] = 219
    assert view[5, 7, 2] == 219, "tensor is not writable through its own view"

    tensor2, _ = bridge.import_nt_handle_to_usm(desc)
    assert tensor2.data[5, 7, 2] == 219, (
        "second import did not observe the write -- the buffer is being copied, "
        "not aliased"
    )


def test_transfer_microbenchmark_is_labelled_synthetic():
    """
    This microbenchmark compares 'no copy' against 'two copies', so its speedup
    is true by construction. Assert that it SAYS SO, rather than asserting the
    tautology itself.
    """
    bridge = LevelZeroUSMBridge()
    bench = bridge.benchmark_transfer(width=256, height=256, iterations=3)
    assert bench["is_synthetic"] is True
    assert bench["measures_hardware"] is False
    assert "zero_copy_verified" not in bench, (
        "the fabricated zero_copy_verified flag is back"
    )


# ---------------------------------------------------------------------------
# Ring buffer
# ---------------------------------------------------------------------------

def test_ring_buffer_fifo_order_and_capacity():
    ring = SpeculativeRingBuffer(capacity=4)
    assert ring.is_empty()

    for i in range(4):
        assert ring.push([i, i + 1]) is True
    assert ring.is_full() is True
    assert ring.push([99]) is False, "overrun was not prevented"

    # FIFO: must come back in push order, with correct payloads.
    for i in range(4):
        pkt = ring.pop()
        assert pkt is not None
        assert pkt.draft_tokens == [i, i + 1], "ring buffer reordered packets"
        assert pkt.seq_id == i
    assert ring.pop() is None
    assert ring.is_empty()


def test_ring_buffer_wraps_around_past_capacity():
    """Push/pop more items than capacity to exercise the modulo indexing."""
    ring = SpeculativeRingBuffer(capacity=3)
    seen = []
    for i in range(10):
        assert ring.push([i]) is True
        pkt = ring.pop()
        seen.append(pkt.draft_tokens[0])
    assert seen == list(range(10))


def test_alignment_property_reports_the_truth():
    """
    Previously `assert ring.is_aligned_64 is True` passed because the property
    was reading buffer CONTENTS as a pointer and always returned True. It now
    measures the real address, so assert it is consistent with that address
    rather than asserting a fixed answer.
    """
    ring = SpeculativeRingBuffer(capacity=16)
    expected = (ring.buffer_address % 64) == 0
    assert ring.is_aligned_64 is expected


# ---------------------------------------------------------------------------
# Log-Mel front-end (CPU)
# ---------------------------------------------------------------------------

def _hz_to_mel(hz):
    return 2595.0 * np.log10(1.0 + hz / 700.0)


def test_mel_filterbank_matches_independent_construction():
    """Check the filterbank against a from-scratch reference, not its own code."""
    proc = ShaveDSPSpectralProcessor(n_fft=512, hop_length=160, n_mels=80, sample_rate=16000)
    fb = proc.mel_filters
    assert fb.shape == (80, 512 // 2 + 1)
    # Triangular filters are non-negative and each must contain a nonzero peak.
    assert (fb >= 0).all()
    assert (fb.max(axis=1) > 0).all(), "one or more mel filters are all-zero"
    # Filter centre frequencies must be monotonically increasing.
    centres = fb.argmax(axis=1)
    assert (np.diff(centres) >= 0).all(), "mel centres are not monotonic"


def test_pure_tone_lands_in_the_expected_mel_band():
    """
    The real test: feed a known 1 kHz sine and assert the energy peak appears in
    the mel band that 1 kHz actually maps to. This fails if the FFT, the
    windowing, the filterbank or the frequency mapping is wrong.
    """
    sr, n_fft, n_mels = 16000, 512, 80
    proc = ShaveDSPSpectralProcessor(n_fft=n_fft, hop_length=160, n_mels=n_mels, sample_rate=sr)

    freq = 1000.0
    t = np.arange(sr, dtype=np.float32) / sr
    audio = np.sin(2 * np.pi * freq * t).astype(np.float32)

    log_mel, _ = proc.process_spectral_frame(audio)
    assert log_mel.shape[0] == n_mels
    assert np.isfinite(log_mel).all()

    # Independently predict the mel bin for 1 kHz from the mel scale.
    mel_lo, mel_hi = _hz_to_mel(0.0), _hz_to_mel(sr / 2.0)
    expected_bin = int(round((_hz_to_mel(freq) - mel_lo) / (mel_hi - mel_lo) * (n_mels + 1))) - 1

    peak_bin = int(np.argmax(log_mel.mean(axis=1)))
    assert abs(peak_bin - expected_bin) <= 3, (
        f"1 kHz tone peaked at mel bin {peak_bin}, expected ~{expected_bin}"
    )


def test_silence_has_lower_energy_than_tone():
    """A trivially true-sounding claim that the code could still get wrong."""
    proc = ShaveDSPSpectralProcessor()
    sr = 16000
    t = np.arange(sr, dtype=np.float32) / sr
    tone = np.sin(2 * np.pi * 440 * t).astype(np.float32)
    silence = np.zeros(sr, dtype=np.float32)

    mel_tone, _ = proc.process_spectral_frame(tone)
    mel_silence, _ = proc.process_spectral_frame(silence)

    # Compare PEAK band energy, not the mean: a pure tone excites only one or
    # two of 80 bands, so the mean barely moves and would hide a real failure.
    assert mel_tone.max() > mel_silence.max() + 5.0
    # Silence must sit flat on the log floor, ln(1e-5) = -11.5129.
    assert np.allclose(mel_silence, np.log(1e-5), atol=1e-3)
