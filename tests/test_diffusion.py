"""
Unit and physical silicon tests for Sprint 4 (JOB P1-02):
Heterogeneous Latent Consistency Model (LCM) 4-Step Diffusion Engine.
"""

import json
import tempfile
from pathlib import Path
import numpy as np
import pytest
from PIL import Image

from lunar_core.diffusion import (
    DiffusionResult,
    LunarHeterogeneousDiffusion,
    build_clip_text_openvino_model,
    build_lcm_step_openvino_model,
    build_vae_decoder_openvino_model,
    get_diffusion_engine,
)
from lunar_core.mcp_server import LunarMCPServer


def test_build_clip_text_openvino_model():
    """Verify CLIP text transformer model compiles and outputs [1, 77, 768] context tensor."""
    model = build_clip_text_openvino_model()
    assert model.get_friendly_name() == "CLIPTextTransformer"
    assert len(model.inputs) == 1
    assert len(model.outputs) == 1
    assert model.output(0).shape == [1, 77, 768]


def test_build_lcm_step_openvino_model():
    """Verify LCM consistency ODE step model compiles and preserves [1, 4, 64, 64] latent shape."""
    model = build_lcm_step_openvino_model()
    assert model.get_friendly_name() == "LCMConsistencyStep"
    assert len(model.inputs) == 3
    assert len(model.outputs) == 1
    assert model.output(0).shape == [1, 4, 64, 64]


def test_build_vae_decoder_openvino_model():
    """Verify VAE latent decoder upsamples [1, 4, 64, 64] into [1, 3, 512, 512] RGB surface."""
    model = build_vae_decoder_openvino_model()
    assert model.get_friendly_name() == "VAELatentDecoder"
    assert len(model.inputs) == 1
    assert len(model.outputs) == 1
    assert model.output(0).shape == [1, 3, 512, 512]


def test_heterogeneous_diffusion_sketch():
    """Verify heterogeneous 4-step diffusion produces a valid 512x512 image in <1500ms."""
    diff = LunarHeterogeneousDiffusion()
    assert diff.text_device in ["NPU", "GPU", "CPU"]
    assert diff.denoiser_device in ["GPU", "NPU", "CPU"]

    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = Path(tmpdir) / "test_out.png"
        res = diff.sketch("code review icon", steps=4, out_path=out_path)

        assert isinstance(res, DiffusionResult)
        assert res.resolution == (512, 512)
        assert res.steps == 4
        assert res.latency_ms < 1500.0  # Acceptance criterion: <1.5s
        assert len(res.phash) == 16
        assert out_path.exists()

        with Image.open(out_path) as img:
            assert img.size == (512, 512)


def test_mcp_lunar_diffuse_tool():
    """Verify FastMCP server dispatches lunar_diffuse tool call cleanly."""
    server = LunarMCPServer()
    req = {
        "jsonrpc": "2.0",
        "id": 42,
        "method": "tools/call",
        "params": {
            "name": "lunar_diffuse",
            "arguments": {
                "prompt": "silicon microarchitecture diagram",
                "steps": 2,
            },
        },
    }
    resp = server.handle_request(req)
    assert resp is not None
    assert resp["id"] == 42
    assert "result" in resp
    content = resp["result"]["content"][0]["text"]
    data = json.loads(content)
    assert data["prompt"] == "silicon microarchitecture diagram"
    assert data["resolution"] == [512, 512]
    assert data["latency_ms"] > 0.0
