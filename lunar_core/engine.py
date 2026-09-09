"""
Recipe 1: Production NPU Engine Initializer & Compilation Manager
Configures OpenVINO runtime for optimal execution on Intel Lunar Lake NPU 4000.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional
import openvino as ov


class LunarNPUEngine:
    """Production NPU compilation manager and device orchestrator."""

    DEFAULT_CACHE_DIR = Path.home() / ".tools" / "npu" / "cache"

    def __init__(
        self,
        cache_dir: Optional[str] = None,
        target_device: Optional[str] = None,
        turbo_mode: bool = True,
        max_tiles: int = 6,
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
