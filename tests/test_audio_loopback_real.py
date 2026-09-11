"""Real integration and unit tests for Windows WASAPI loopback audio capture, energy VAD, and Whisper transcription."""

import json
import numpy as np
import pytest
from click.testing import CliRunner

from lunar_core.audio import (
    EnergyBasedVAD,
    LunarAudioEngine,
    NativeWASAPILoopbackClient,
    TranscriptionResult,
    WASAPILoopbackCapture,
)
from lunar_core.cli import cli


def test_energy_based_vad_silence():
    vad = EnergyBasedVAD(energy_threshold=0.005)
    silence = np.zeros(16000, dtype=np.float32)
    decision = vad.analyze(silence)
    assert decision.is_speech is False
    assert decision.rms_energy == 0.0
    assert "is_speech" in decision.to_dict()


def test_energy_based_vad_speech():
    vad = EnergyBasedVAD(energy_threshold=0.005)
    t = np.linspace(0, 0.5, 8000, endpoint=False, dtype=np.float32)
    tone = (0.25 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    decision = vad.analyze(tone)
    assert decision.is_speech is True
    assert decision.rms_energy > 0.05


def test_wasapi_loopback_client_init():
    client = NativeWASAPILoopbackClient(target_sample_rate=16000)
    assert isinstance(client.is_available, bool)


def test_wasapi_loopback_client_capture():
    client = NativeWASAPILoopbackClient(target_sample_rate=16000)
    samples = client.capture(duration_s=0.2)
    assert isinstance(samples, np.ndarray)
    assert samples.dtype == np.float32
    assert len(samples) == int(0.2 * 16000)


def test_wasapi_ring_buffer_capture_live():
    ring = WASAPILoopbackCapture(buffer_seconds=2.0, sample_rate=16000)
    written = ring.capture_live(duration_s=0.2)
    assert written == int(0.2 * 16000)
    read = ring.read_latest_seconds(0.1)
    assert len(read) == int(0.1 * 16000)


def test_lunar_audio_transcribe_loopback():
    audio = LunarAudioEngine()
    res = audio.transcribe_loopback(duration_s=0.2)
    assert isinstance(res, TranscriptionResult)
    assert res.audio_source == "wasapi_loopback_stream"
    assert res.audio_duration_s == 0.2
    assert res.latency_ms > 0.0
    assert isinstance(res.text, str)


def test_lunar_audio_cli_transcribe_loopback():
    runner = CliRunner()
    result = runner.invoke(cli, ["transcribe", "--loopback", "--duration", "0.2", "--json"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert data["audio_source"] == "wasapi_loopback_stream"
    assert "latency_ms" in data
    assert "vad_active" in data
