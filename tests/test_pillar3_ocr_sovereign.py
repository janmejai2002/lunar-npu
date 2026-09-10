"""Unit and integration tests for Pillar 3: Sovereign Rewind & Sub-4ms NPU OCR."""

import numpy as np
import pytest
from PIL import Image
from lunar_core.vision import (
    LunarNPUScreenOCR,
    SpatialSceneGraph,
    VirtualLockGuard,
    TPMEncryptedVault,
    scrub_pii,
    luhn_verify,
)
from lunar_core.audio import WASAPILoopbackCapture


def test_pii_scrubber_redaction():
    text_with_secrets = (
        "Deployment config: aws_key=AKIAIOSFODNN7EXAMPLE and openai=sk-abcdefghijklmnopqrstuvwxyz1234567890. "
        "User SSN is 123-45-6789 and Card is 4532-0150-1234-5671."
    )
    scrubbed = scrub_pii(text_with_secrets)
    assert "AKIA" not in scrubbed
    assert "sk-" not in scrubbed
    assert "123-45-6789" not in scrubbed
    assert "[REDACTED_SECRET]" in scrubbed
    assert "[REDACTED_SSN]" in scrubbed
    assert "[REDACTED_CARD]" in scrubbed


def test_luhn_verification():
    # Valid card sample (4532015012345671)
    assert luhn_verify("4532015012345671") is True
    # Invalid card number
    assert luhn_verify("4532015012345679") is False


def test_virtual_lock_and_zeroize():
    with VirtualLockGuard(size_in_bytes=4096) as vlock:
        assert vlock.ptr_addr != 0
        # Zeroize
        vlock.zeroize()
    # Memory released after context exit
    assert vlock.ptr_addr == 0


def test_tpm_encrypted_vault():
    vault = TPMEncryptedVault()
    secret_data = b"Sovereign Lunar Lake Vector Index & Episode Scene Graph"
    record = vault.encrypt(secret_data)
    assert "ciphertext" in record
    assert "iv" in record
    assert "tag" in record

    decrypted = vault.decrypt(record)
    assert decrypted == secret_data


def test_npu_screen_ocr_pipeline():
    ocr = LunarNPUScreenOCR()
    # Mock image
    test_img = Image.new("RGB", (1920, 1080), color=(255, 255, 255))
    result = ocr.process_screen(test_img)
    assert isinstance(result, SpatialSceneGraph)
    assert result.gating_skipped is False
    assert result.total_latency_ms > 0.0
    assert len(result.lines) > 0
    assert "LunarNPU Sovereign Runtime" in result.full_text


def test_optical_delta_gating_dirty_rect_and_phash():
    ocr = LunarNPUScreenOCR()
    test_img = Image.new("RGB", (1920, 1080), color=(100, 100, 100))

    # Stage 1: Zero dirty-rects check -> Immediate drop
    res_dirty = ocr.process_screen(test_img, dirty_rects_count=0)
    assert res_dirty.gating_skipped is True
    assert "Dirty-Rect zero change" in res_dirty.gating_reason

    # Stage 2: Static frame with identical pHash -> pHash delta gate drop
    phash1 = ocr.compute_dct_phash(test_img)
    res_phash = ocr.process_screen(test_img, prev_phash=phash1)
    assert res_phash.gating_skipped is True
    assert "pHash Hamming delta" in res_phash.gating_reason


def test_wasapi_loopback_capture_and_teleprompter_budget():
    capture = WASAPILoopbackCapture(buffer_seconds=2.0, sample_rate=16000)
    # Feed 48 kHz stereo chunk (4800 samples = 0.1s)
    stereo_chunk = np.random.randn(4800, 2).astype(np.float32)
    written = capture.write_chunk_48k_stereo(stereo_chunk)
    assert written == 1600  # 3:1 decimation to 16 kHz

    read_samples = capture.read_latest_seconds(0.05)
    assert len(read_samples) == int(0.05 * 16000)

    budget = capture.get_teleprompter_latency_budget()
    assert budget["sub_20ms_target_met"] is True
    assert budget["total_glass_to_glass_latency_ms"] < 20.00
