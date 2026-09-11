# Lunar Lake NPU â€” Hardware & Runtime Reference

**Purpose.** This is the single factual reference for building on this machine.
It replaces `research/01`â€“`research/13`, `LUNAR_NPU_MASTER_TREATISE_50_PAGES.md`,
`MASTER_TECHNICAL_CONSTITUTION_50_PAGES.md`, `MAMBA_DEEP_RESEARCH_COMPENDIUM_2026.md`,
`DEEPENING_THE_LUNAR_MOAT_AND_NEXT_GEN_SYSTEM_SPEC.md` and
`RESEARCH_VS_REALITY_GAP_ANALYSIS.md` (all archived under `docs/archive/`).
Those documents mixed verified facts with fabricated benchmarks and invented
"world-first" claims, with no way to tell which was which.

**Every claim below carries a tag:**

| Tag | Meaning |
| :-- | :-- |
| **[MEASURED]** | Produced by running code on this machine on 2026-09-11. Command included. |
| **[DRIVER]** | Read directly from the OpenVINO/driver property API on this machine. |
| **[THEORY]** | Standard, textbook-checkable ML/systems knowledge. Not machine-specific. |
| **[UNVERIFIED]** | Plausible vendor/architecture claim carried over from the old research. Not checked. Do not cite as a result. |

If you need a number not tagged **[MEASURED]** or **[DRIVER]**, measure it before
you rely on it.

---

## 1. The one fact that should drive every design decision

> **[MEASURED] The NPU is the slowest of the three engines at every workload
> size tested. It has a fixed ~0.22 ms dispatch floor. It never wins on latency.**

Median inference latency, dense `f32` MLP, batch 1, 30 iterations after warmup:

| Workload | CPU | GPU | NPU | Fastest |
| :-- | --: | --: | --: | :-- |
| 128Ã—128 Ã—1 | **0.022 ms** | 0.061 ms | 0.216 ms | CPU |
| 512Ã—512 Ã—1 | **0.035 ms** | 0.084 ms | 0.247 ms | CPU |
| 512Ã—512 Ã—4 | **0.075 ms** | 0.123 ms | 0.258 ms | CPU |
| 1024Ã—1024 Ã—4 | 0.170 ms | **0.162 ms** | 0.458 ms | GPU |
| 2048Ã—2048 Ã—4 | 0.864 ms | **0.496 ms** | 0.936 ms | GPU |
| 1024Ã—1024 Ã—16 | 1.123 ms | **0.557 ms** | 0.933 ms | GPU |
| 2048Ã—2048 Ã—16 | 5.488 ms | **2.290 ms** | 2.610 ms | GPU |

Cold compile cost, same graph: **[MEASURED]** CPU â‰ˆ 26 ms, NPU â‰ˆ 99 ms,
GPU â‰ˆ 1219 ms. The GPU's compile cost is ~12Ã— the NPU's â€” cache compiled blobs
(`CACHE_DIR`) or the GPU's latency win is erased on first use.

**Consequences you must design around:**

1. A "sub-3 ms on the NPU!" claim is not an achievement. The CPU does the same
   small op in 0.03 ms. **Do not move small ops to the NPU for speed.** The
   `router`, `circuit_breaker` and `vector_memory` hot paths in this repo are
   all in the size regime where the CPU wins outright.
2. The NPU's real value is **energy per inference** and **not occupying the
   CPU/GPU** â€” it lets a continuous background workload run without stealing
   cycles from the foreground or spinning the fan. Justify NPU placement on
   those axes, and measure them (see Â§5), not on latency.
3. The per-call floor means **batch or fuse**. Six separate 0.25 ms NPU calls
   cost 1.5 ms, nearly all of it dispatch. One fused graph costs ~0.25 ms.
4. The NPU only becomes competitive with the GPU above ~30 MFLOP/inference.
   Below that it is pure overhead.

Reproduce: the sweep script is small enough to rewrite; it compiles an NÃ—N
Ã—L-layer ReLU MLP per device and takes the median of 30 timed `infer()` calls.

---

## 2. This machine â€” verified identity

**[DRIVER]** Read via `openvino.Core().get_property(dev, prop)`.

```
OpenVINO runtime : 2026.2.1-21919-ede283a88e3-releases/2026/2
Available devices: ['CPU', 'GPU', 'NPU']
CPU              : Intel(R) Core(TM) Ultra 7 256V
GPU              : Intel(R) Arc(TM) 140V GPU (8GB) (iGPU)
NPU              : Intel(R) AI Boost
```

> **Correction to existing docs.** The README, `CONTEXT_HANDOFF.md`, the project
> factsheet and the archived gap-analysis all stated the CPU was a **Core Ultra 7
> 258V**. It is a **256V** (see `FULL_DEVICE_NAME` above). This has been corrected
> across the repo. The archived gap-analysis also claims OpenVINO 2025.0.0; the
> installed runtime is 2026.2.1.

### 2.1 NPU properties **[DRIVER]**

```
DEVICE_ARCHITECTURE          = 4000
DEVICE_TYPE                  = INTEGRATED
DEVICE_PCI_INFO              = domain 0, bus 0, device 0xb, function 0
NPU_MAX_TILES                = 6
NPU_TILES                    = -1        # auto-partition across all tiles
NPU_DEVICE_TOTAL_MEM_SIZE    = 8589934592   (8.0 GB)
NPU_DRIVER_VERSION           = 1004723
NPU_COMPILER_VERSION         = 524289
OPTIMIZATION_CAPABILITIES    = ['FP16', 'INT8', 'EXPORT_IMPORT']
RANGE_FOR_ASYNC_INFER_REQUESTS = (1, 10, 1)
RANGE_FOR_STREAMS            = (1, 4)
OPTIMAL_NUMBER_OF_INFER_REQUESTS = 1
INFERENCE_PRECISION_HINT     = f16
NPU_TURBO                    = False     # default; settable
NPU_QDQ_OPTIMIZATION         = False     # default; settable

DEVICE_GOPS = {
    int8   : 46694.398,   # 46.69 TOPS
    uint8  : 46694.398,
    float16: 23347.199,   # 23.35 TFLOPS
    bfloat16: 0.0,        # NOT supported natively
    float32: 0.0,         # NOT supported natively
}
```

**Read these carefully.** `float32: 0.0` and `bfloat16: 0.0` mean the NPU has
**no native FP32 or BF16 path**. `OPTIMIZATION_CAPABILITIES` lists only FP16 and
INT8. Every FP32 graph in this repo (`circuit_breaker`, `speculative`,
`micro_lora`, `diffusion` all build `ov.Type.f32` parameters) is being converted
to FP16 by the compiler before it runs. That conversion is silent. If you need
FP32 semantics, the NPU is the wrong device.

`DEVICE_GOPS` is the **driver's theoretical peak**, not a measurement. 46.69 TOPS
is what the hardware could do at 100% systolic utilisation on INT8. Nothing in
this repo comes remotely close, and quoting it as a project capability is the
single most common misrepresentation in the archived docs.

### 2.2 GPU properties **[DRIVER]**

```
FULL_DEVICE_NAME             = Intel(R) Arc(TM) 140V GPU (8GB) (iGPU)
GPU_EXECUTION_UNITS_COUNT    = 64
GPU_DEVICE_TOTAL_MEM_SIZE    = 8535658496   (7.95 GB)
GPU_UARCH_VERSION            = 20.4.4
OPTIMIZATION_CAPABILITIES    = ['FP32','BIN','FP16','INT8','GPU_HW_MATMUL','GPU_USM_MEMORY','EXPORT_IMPORT']
GPU_ENABLE_LORA_OPERATION    = True
GPU_ENABLE_SDPA_OPTIMIZATION = True

DEVICE_GOPS = { int8: 63897.6, float16: 31948.8, float32: 3993.6 }
```

Note `GPU_USM_MEMORY` in the capability list: **the GPU advertises USM support,
the NPU does not.** Any zero-copy shared-memory work targets the GPU. See Â§6.1.

### 2.3 CPU properties **[DRIVER]**

```
FULL_DEVICE_NAME          = Intel(R) Core(TM) Ultra 7 256V
OPTIMIZATION_CAPABILITIES = ['BF16','FP32','FP16','INT8','BIN','EXPORT_IMPORT']
RANGE_FOR_STREAMS         = (1, 8)
KV_CACHE_PRECISION        = u8
```

The CPU is the only device here with native BF16 and FP32.

---

## 3. Architecture background **[UNVERIFIED]**

Carried over from the old `research/01`. Broadly consistent with public Lunar
Lake material and with the driver values above, but **not independently checked**.
Use for orientation, never as a result.

- NPU 4 is organised as **6 Neural Compute Engine (NCE) tiles**. `NPU_MAX_TILES = 6`
  is the one part of this confirmed **[DRIVER]**.
- Each tile pairs a **DPU** (systolic MAC array, INT8/FP16) with a **SHAVE DSP**
  vector unit for pointwise/activation/normalisation work.
- ~**12 MB** on-die software-managed scratchpad SRAM; the compiler (`vpux-compiler`)
  schedules tile movement explicitly rather than relying on cache hardware.
- **LPDDR5X-8533 on-package**, ~136 GB/s, shared by CPU/GPU/NPU.
- The 46.69 TOPS figure is consistent with `6 tiles Ã— 1024 MAC/cycle Ã— 2 OP/MAC
  Ã— ~1.90 GHz`. That arithmetic checks out against the driver's `DEVICE_GOPS`,
  which is itself derived the same way â€” it is not corroboration.
- DVFS range ~400 MHzâ€“1.95 GHz; deep sleep when the command queue is empty.

Claims in the archived docs that were **stated as measurements without any
measuring code** and should not be reused: "1.2 TB/s SRAM crossbar", "<15 mW
idle", "0.8â€“1.4 W continuous perception", "<2 ms C6 entry", "zero stall cycles",
"15â€“30 TOPS/W". Some may well be true. None were measured here.

---

## 4. Quantisation â€” the genuinely useful theory **[THEORY]**

This section is textbook-correct and is the highest-value technical content
salvaged from the research corpus. It matters because Â§2.1 shows the NPU is
INT8/FP16-only.

### 4.1 Affine vs symmetric

Affine (asymmetric): `q = clip(round(x/S) + Z, q_min, q_max)`, dequantised as
`xÌ‚ = SÂ·(q âˆ’ Z)`, with `S = (Î²âˆ’Î±)/(q_maxâˆ’q_min)` and `Z = round(âˆ’Î±/S) + q_min`.

Symmetric: `Z = 0`, `S = Î³/q_max` where `Î³ = max(|Î±|,|Î²|)`. For int8,
`q = clip(round(x/S), âˆ’128, 127)`.

### 4.2 Why NPU weights must be symmetric

For `Y = XW` with both sides asymmetric, expanding gives four terms:

```
Î£ QxÂ·Qw          <- the systolic array's actual job
âˆ’ Zw Â· Î£ Qx      <- depends on RUNTIME activations
âˆ’ Zx Â· Î£ Qw      <- static, foldable into bias at compile time
+ K Â· Zx Â· Zw    <- constant, foldable
```

The second term is the problem: if `Zw â‰  0` it must be recomputed per inference
as a row-reduction over activations, which lands on the vector units and forces a
DPUâ†”vector sync. **Rule: quantise weights symmetrically (`Zw = 0`).** Activations
may stay asymmetric. This is why NNCF's `PERFORMANCE` preset exists.

### 4.3 Activation outliers and SmoothQuant

Transformer activations have per-channel outliers up to ~100Ã— the typical token
magnitude. Per-tensor INT8 then maps 99%+ of real tokens into 2â€“3 integer bins
and destroys the representation.

SmoothQuant migrates the difficulty into the weights, which quantise easily:

```
Y = XW = (X Â· diag(s)â»Â¹)(diag(s) Â· W)
s_j = max(|X_j|)^Î± / max(|W_j|)^(1âˆ’Î±)
```

`Î± = 0.5` is the usual balance point. `diag(s)â»Â¹` folds into the preceding
LayerNorm/RMSNorm weights, so it costs nothing at runtime.

### 4.4 Precision schemes

| Scheme | Weights | Activations | Memory | Typical accuracy cost |
| :-- | :-- | :-- | :-- | :-- |
| FP16 | 16-bit | 16-bit | 100% | baseline |
| W8A8 | int8 sym | int8 | 50% | small |
| W4A16 | int4 packed | fp16 | 25% | small |
| W4A8 | int4 packed | int8 | 25% | moderate |

**[THEORY]** For batch-1 autoregressive decoding you are memory-bandwidth bound,
not compute bound â€” weight streaming dominates. W4A16 roughly doubles effective
bandwidth. This is why the 0.5B int4 model in `~/.tools/npu/models/slm_real/` is
the right shape for this machine.

### 4.5 NNCF pipeline

```python
import nncf, openvino as ov

quantized = nncf.quantize(
    ov.Core().read_model("model.xml"),
    nncf.Dataset(samples, transform_fn),      # 100-300 in-domain samples
    model_type=nncf.ModelType.TRANSFORMER,
    preset=nncf.QuantizationPreset.PERFORMANCE,   # symmetric weights (see 4.2)
    smooth_quant_alpha=0.5,                       # see 4.3
    fast_bias_correction=True,
    subset_size=256,
    target_device=nncf.TargetDevice.NPU,          # enforces NPU constraints
)
ov.save_model(quantized, "model_npu.xml")
```

**[UNVERIFIED]** The archived research claims "3.8Ã—â€“5.2Ã— faster than FP16 with
>99.5% accuracy retained". Not measured here. Measure on your model.

---

## 5. Measuring energy, since latency is the wrong metric

Â§1 establishes the NPU loses on latency, so its case must be made on power.
`lunar_core/power_telemetry.py` reads Intel RAPL via the Windows PDH API:

```
\Energy Meter(rapl_package0_pkg)\Power
\Energy Meter(rapl_package0_pp0)\Power     # cores
\Energy Meter(rapl_package0_pp1)\Power     # uncore
\Energy Meter(rapl_package0_dram)\Power
\Thermal Zone Information(\_TZ.TZS0)\Temperature
```

**Always check `is_live` before using any value.** When the counters are
unavailable the module returns hardcoded values (18.5 W package, 51 Â°C) tagged
`"Simulated RAPL Baseline"`. Treating those as real is how fabricated power
numbers entered the old docs.

Also note:
- `npu_power_est_w` is a **made-up formula** over uncore power, not a sensor.
  There is no per-engine NPU power rail exposed here.
- The package rail is **shared**. You cannot attribute watts to the NPU by
  subtraction. The only sound measurement is a **differential experiment**: run
  workload on CPU for N seconds and integrate package power, then the same
  workload on NPU, and compare total energy.
- `RAPLPowerGovernor` **does not govern anything** â€” it reads and returns a
  status string. There is no actuation and no control loop.

---

## 6. Constraints that are real

### 6.1 There is no zero-copy path implemented
**[MEASURED]** `LevelZeroUSMBridge` in `engine.py` loads `ze_loader.dll` and
never calls a Level Zero function. It allocates ordinary host memory via
`VirtualAlloc` and fabricates the "NT handle" from the wall clock. A real path
needs `IDXGIOutputDuplication` â†’ shared NT handle â†’ `zeMemAllocShared` /
`zeMemOpenIpcHandle` â†’ `ov.RemoteTensor`. Note from Â§2.2 that **only the GPU
advertises `GPU_USM_MEMORY`**; target the GPU for this, not the NPU.

### 6.2 Desktop isolation blocks screen capture in agent runners
Background/agent runner processes execute on a non-interactive Windows desktop
station (`WinSta0\exebox-â€¦`). Windows DWM will not let them read pixels from the
interactive `Default` desktop, so `dxgi_capture.py` returns an all-zero frame
there. This is a genuine OS constraint, not a bug â€” it will not be fixed by
changing the capture code. Test screen capture from an interactive session.

### 6.3 Static shapes
The NPU compiler wants static shapes. `vector_memory.py` already reshapes
dynamic inputs to `[1, seq_len]` before compiling. Dynamic-shape graphs either
fail to compile or silently fall back.

### 6.4 Compile caching
Set `CACHE_DIR` in the compile config. Cold compile is ~99 ms (NPU) / ~1219 ms
(GPU) **[MEASURED]**; cached blob load is far cheaper. `EXPORT_IMPORT` is in the
capability list for all three devices.

---

## 7. Model assets present on this machine **[MEASURED]**

Under `~/.tools/npu/models/`:

| Path | What it is | Size |
| :-- | :-- | --: |
| `bge_real/` | BGE embedding model + tokenizer, OpenVINO IR | 104.5 MB |
| `embed/minilm_l6.*` | MiniLM-L6 embeddings | 25.7 MB |
| `slm_real/` | Qwen2.5-Coder-0.5B-Instruct int4 | â€” |
| `whisper_real/` | Whisper encoder + decoder + tokenizer | â€” |
| `yolo_real/yolo11n.*` | YOLO11n detector | â€” |
| `lcm_real/` | LCM: text_encoder, unet, vae_encoder, vae_decoder | â€” |
| `vision/mobilenet_v3.*` | MobileNetV3 | â€” |

These are real trained weights and are the honest core of the project. Two
cautions:

1. **`vector_memory.py` silently falls back to random projections** if
   `bge_real/` is missing â€” it now emits a `RuntimeWarning` and sets
   `is_degraded`, surfaced at `/api/health` as `embeddings_degraded`. Check it.
2. **YOLO11n is trained on COCO** (people, cars, animals). It cannot detect UI
   elements. Any "screen understanding" built on it needs OCR/DBNet instead.

---

## 8. Things in this repo that are NOT what their names say

Short index so you do not have to rediscover these. Full detail in `CLAUDE.md` Â§2.

| Name | Reality |
| :-- | :-- |
| `build_hazard_classifier_openvino` | Untrained random MLP. Constant output 0.0474. Advisory only. |
| `LevelZeroUSMBridge` | No Level Zero, no D3D11, no zero-copy. |
| `SpeculativeRingBuffer` | Not lock-free, not cache-aligned. Plain list under the GIL. |
| `ShaveDSPSpectralProcessor` | numpy on the CPU. Aliased to `MelSpectrogramCPU`. |
| `SRAMAdamW` | numpy arrays in host DRAM. Correct AdamW; wrong name. |
| `build_draft_model_openvino` | Untrained. **[MEASURED]** speculation runs at 0.0017Ã— â€” ~580Ã— slower than plain decoding. |
| `GeodesicMicroRouter.route_geodesic` | `arccos(cosine)` is monotonic in cosine, so it ranks identically; the distances are computed, logged, then discarded. |
| `RAPLPowerGovernor` | Monitor, not a governor. |
| `HyperdimensionalMemoryEngine.query` | SHA-256 per whole string; one word changed â‡’ ~0.50 similarity (noise). |
| `srnc.py` | Cassowary/APCA/CST code from the unrelated `agent-craft` project. Nothing to do with the NPU. |

---

## 9. Useful OpenVINO patterns

```python
import openvino as ov, openvino.opset13 as ops, numpy as np

core = ov.Core()

# Build a graph directly (no framework import needed).
x = ops.parameter([1, 512], ov.Type.f32, name="x")
w = ops.constant(np.random.randn(512, 512).astype(np.float32))
y = ops.relu(ops.matmul(x, w, False, False))
model = ov.Model([y], [x], "mlp")

# Compile with caching. NPU wants static shapes and LATENCY hint for batch 1.
compiled = core.compile_model(model, "NPU", {
    "CACHE_DIR": "~/.tools/npu/cache",
    "PERFORMANCE_HINT": "LATENCY",
    "NPU_TURBO": "YES",
})

req = compiled.create_infer_request()
req.infer({0: np.random.randn(1, 512).astype(np.float32)})
out = req.get_output_tensor(0).data

# Read real device facts rather than asserting them.
print(core.get_property("NPU", "DEVICE_GOPS"))
print(core.get_property("NPU", "SUPPORTED_PROPERTIES"))
```

Timing rule: **always warm up** (first `infer()` includes lazy setup), then take
a median over â‰¥30 runs. A single timed call is noise.

---

## 10. Provenance

Written 2026-09-11 from a direct audit of the codebase plus live probes on this
machine. Measurements came from: `openvino.Core()` property reads; a per-device
compile+infer sweep; a `DualStageSiliconCircuitBreaker` input sweep; an
`HyperdimensionalMemoryEngine` paraphrase-recall probe; a `GeodesicMicroRouter`
ranking comparison; and `pytest tests/` (170 passing after the audit fixes).

When you add a fact here, tag it, and say how it was obtained.
