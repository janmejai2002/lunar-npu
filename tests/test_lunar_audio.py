"""Unit and integration tests for LunarAudio Whisper transcription on Intel NPU."""

import json
import pytest
from click.testing import CliRunner

from lunar_core.audio import LunarAudioEngine, TranscriptionResult
from lunar_core.cli import cli


def test_audio_engine_initialization():
    audio = LunarAudioEngine()
    assert audio.engine is not None
    assert audio.device in ("NPU", "GPU", "CPU")


def test_audio_transcribe_synthetic_stream():
    audio = LunarAudioEngine()
    res = audio.transcribe(audio_source=None)
    assert isinstance(res, TranscriptionResult)
    assert res.audio_duration_s > 0.0
    assert res.latency_ms > 0.0
    assert isinstance(res.text, str)
    assert len(res.text) > 0
    assert res.device in ("NPU", "GPU", "CPU")


def test_audio_cli_transcribe():
    runner = CliRunner()
    result = runner.invoke(cli, ["transcribe", "--json"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert "audio_source" in data
    assert "text" in data
    assert data["latency_ms"] > 0.0
