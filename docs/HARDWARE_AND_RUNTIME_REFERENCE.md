# Lunar Lake NPU - Hardware & Runtime Reference

**Purpose.** This is the single factual reference for building on this machine.
It replaces `research/01`-`research/13`, `LUNAR_NPU_MASTER_TREATISE_50_PAGES.md`,
`MASTER_TECHNICAL_CONSTITUTION_50_PAGES.md`, `MAMBA_DEEP_RESEARCH_COMPENDIUM_2026.md`,
`DEEPENING_THE_LUNAR_MOAT_AND_NEXT_GEN_SYSTEM_SPEC.md` and
`RESEARCH_VS_REALITY_GAP_ANALYSIS.md` (all archived under `docs/archive/`).
Those documents mixed verified facts with fabricated benchmarks and invented
"world-first" claims, with no way to tell which was which.

This file is deliberately ASCII-only. An earlier revision was corrupted by a
PowerShell encoding round-trip; keeping the punctuation plain avoids a repeat.

**Every claim below carries a tag:**

| Tag | Meaning |
| :-- | :-- |
| **[MEASURED]** | Produced by running code on this machine on 2026-09-11. |
| **[DRIVER]** | Read directly from the OpenVINO/driver property API on this machine. |
| **[THEORY]** | Standard, textbook-checkable ML/systems knowledge. Not machine-specific. |
| **[UNVERIFIED]** | Plausible vendor/architecture claim carried over from the old research. Not checked. Do not cite as a result. |

If you need a number not tagged **[MEASURED]** or **[DRIVER]**, measure it before
you rely on it.

---

## 1. The one fact that should drive every design decision

> **[MEASURED] On dense fp16/fp32 graphs the NPU is the slowest of the three
> engines at every size tested, with a fixed ~0.22 ms dispatch floor.**
>
> **CORRECTED 2026-09-12 -- see section 15.** That statement holds only for the
> synthetic dense matmuls below, which are the NPU's weak path. On Intel's
> pre-quantised INT8 production models the NPU BEATS the GPU by 1.15-1.46x,
> consistently. Read section 15 before drawing any conclusion from this table.

Median inference latency, dense `f32` MLP, batch 1, 30 iterations after warmup:

| Workload | CPU | GPU | NPU | Fastest |
| :-- | --: | --: | --: | :-- |
| 128x128 x1 | **0.022 ms** | 0.061 ms | 0.216 ms | CPU |
| 512x512 x1 | **0.035 ms** | 0.084 ms | 0.247 ms | CPU |
| 512x512 x4 | **0.075 ms** | 0.123 ms | 0.258 ms | CPU |
| 1024x1024 x4 | 0.170 ms | **0.162 ms** | 0.458 ms | GPU |
| 2048x2048 x4 | 0.864 ms | **0.496 ms** | 0.936 ms | GPU |
| 1024x1024 x16 | 1.123 ms | **0.557 ms** | 0.933 ms | GPU |
| 2048x2048 x16 | 5.488 ms | **2.290 ms** | 2.610 ms | GPU |

Cold compile cost, same graph: **[MEASURED]** CPU ~26 ms, NPU ~99 ms,
GPU ~1219 ms. The GPU's compile cost is ~12x the NPU's, so cache compiled blobs
(`CACHE_DIR`) or the GPU's latency win is erased on first use.

**Consequences you must design around:**

1. A "sub-3 ms on the NPU!" claim is not an achievement. The CPU does the same
   small op in 0.03 ms. **Do not move small ops to the NPU for speed.** The
   `router`, `circuit_breaker` and `vector_memory` hot paths in this repo are
   all in the size regime where the CPU wins outright.
2. The NPU's real value is **energy per inference** and **partially sparing the
   foreground** (quantified in section 11.3). Justify NPU placement on those
   axes, and measure them (see section 5), never on latency.
3. The per-call floor means **batch or fuse**. Six separate 0.25 ms NPU calls
   cost 1.5 ms, nearly all of it dispatch. One fused graph costs ~0.25 ms.
4. The NPU only becomes competitive with the GPU above ~30 MFLOP/inference.
   Below that it is pure overhead.

Reproduce: compile an NxN, L-layer ReLU MLP per device and take the median of 30
timed `infer()` calls after 3 warmup calls.

---

## 2. This machine - verified identity

**[DRIVER]** Read via `openvino.Core().get_property(dev, prop)`.

```
OpenVINO runtime : 2026.2.1-21919-ede283a88e3-releases/2026/2
Available devices: ['CPU', 'GPU', 'NPU']
CPU              : Intel(R) Core(TM) Ultra 7 256V
GPU              : Intel(R) Arc(TM) 140V GPU (8GB) (iGPU)
NPU              : Intel(R) AI Boost
```

> **Correction to existing docs.** The README, `CONTEXT_HANDOFF.md`, the project
> factsheet and the archived gap-analysis all stated the CPU was a Core Ultra 7
> **258V**. It is a **256V** (see `FULL_DEVICE_NAME` above). This has been
> corrected across the repo. The archived gap-analysis also claims OpenVINO
> 2025.0.0; the installed runtime is 2026.2.1.

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
    int8    : 46694.398,   # 46.69 TOPS
    uint8   : 46694.398,
    float16 : 23347.199,   # 23.35 TFLOPS
    bfloat16: 0.0,         # NOT supported natively
    float32 : 0.0,         # NOT supported natively
}
```

**Read these carefully.** `float32: 0.0` and `bfloat16: 0.0` mean the NPU has
**no native FP32 or BF16 path**. `OPTIMIZATION_CAPABILITIES` lists only FP16 and
INT8. Every FP32 graph in this repo (`circuit_breaker`, `speculative`,
`micro_lora` and `diffusion` all build `ov.Type.f32` parameters) is converted to
FP16 by the compiler before it runs. That conversion is silent. If you need FP32
semantics, the NPU is the wrong device.

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
the NPU does not.** Any zero-copy shared-memory work targets the GPU. See 6.1.

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
Lake material and with the driver values above, but **not independently
checked**. Use for orientation, never as a result.

- NPU 4 is organised as **6 Neural Compute Engine (NCE) tiles**.
  `NPU_MAX_TILES = 6` is the one part of this confirmed **[DRIVER]**.
- Each tile pairs a **DPU** (systolic MAC array, INT8/FP16) with a **SHAVE DSP**
  vector unit for pointwise, activation and normalisation work.
- ~**12 MB** on-die software-managed scratchpad SRAM; the compiler
  (`vpux-compiler`) schedules tile movement explicitly rather than relying on
  cache hardware.
- **LPDDR5X-8533 on-package**, ~136 GB/s, shared by CPU, GPU and NPU.
- The 46.69 TOPS figure is consistent with
  `6 tiles * 1024 MAC/cycle * 2 OP/MAC * ~1.90 GHz`. That arithmetic agrees with
  the driver's `DEVICE_GOPS`, which is derived the same way, so it is not
  independent corroboration.
- DVFS range ~400 MHz to 1.95 GHz; deep sleep when the command queue is empty.

Claims in the archived docs that were **stated as measurements with no measuring
code** and should not be reused: "1.2 TB/s SRAM crossbar", "<15 mW idle",
"0.8-1.4 W continuous perception", "<2 ms C6 entry", "zero stall cycles",
"15-30 TOPS/W". Some may well be true. None were measured here.

---

## 4. Quantisation - the genuinely useful theory **[THEORY]**

Textbook-correct, and the highest-value technical content salvaged from the
research corpus. It matters because 2.1 shows the NPU is INT8/FP16-only.

### 4.1 Affine vs symmetric

Affine (asymmetric): `q = clip(round(x/S) + Z, q_min, q_max)`, dequantised as
`x_hat = S*(q - Z)`, with `S = (beta-alpha)/(q_max-q_min)` and
`Z = round(-alpha/S) + q_min`.

Symmetric: `Z = 0`, `S = gamma/q_max` where `gamma = max(|alpha|,|beta|)`. For
int8, `q = clip(round(x/S), -128, 127)`.

### 4.2 Why NPU weights must be symmetric

For `Y = XW` with both sides asymmetric, expanding gives four terms:

```
sum(Qx*Qw)        <- the systolic array's actual job
- Zw * sum(Qx)    <- depends on RUNTIME activations
- Zx * sum(Qw)    <- static, foldable into bias at compile time
+ K * Zx * Zw     <- constant, foldable
```

The second term is the problem: if `Zw != 0` it must be recomputed per inference
as a row reduction over activations, which lands on the vector units and forces
a DPU/vector sync. **Rule: quantise weights symmetrically (`Zw = 0`).**
Activations may stay asymmetric. This is what NNCF's `PERFORMANCE` preset
enforces.

### 4.3 Activation outliers and SmoothQuant

Transformer activations have per-channel outliers up to ~100x the typical token
magnitude. Per-tensor INT8 then maps 99%+ of real tokens into 2-3 integer bins
and destroys the representation.

SmoothQuant migrates the difficulty into the weights, which quantise easily:

```
Y = XW = (X * diag(s)^-1) * (diag(s) * W)
s_j = max(|X_j|)^alpha / max(|W_j|)^(1-alpha)
```

`alpha = 0.5` is the usual balance point. `diag(s)^-1` folds into the preceding
LayerNorm/RMSNorm weights, so it costs nothing at runtime.

### 4.4 Precision schemes

| Scheme | Weights | Activations | Memory | Typical accuracy cost |
| :-- | :-- | :-- | :-- | :-- |
| FP16 | 16-bit | 16-bit | 100% | baseline |
| W8A8 | int8 sym | int8 | 50% | small |
| W4A16 | int4 packed | fp16 | 25% | small |
| W4A8 | int4 packed | int8 | 25% | moderate |

**[THEORY]** For batch-1 autoregressive decoding you are memory-bandwidth bound,
not compute bound - weight streaming dominates. W4A16 roughly doubles effective
bandwidth. Section 11.2 confirms this experimentally on this machine, and it is
why the int4 0.5B model in `~/.tools/npu/models/slm_real/` is the right shape
here.

### 4.5 NNCF pipeline

```python
import nncf, openvino as ov

quantized = nncf.quantize(
    ov.Core().read_model("model.xml"),
    nncf.Dataset(samples, transform_fn),          # 100-300 in-domain samples
    model_type=nncf.ModelType.TRANSFORMER,
    preset=nncf.QuantizationPreset.PERFORMANCE,   # symmetric weights, see 4.2
    smooth_quant_alpha=0.5,                       # see 4.3
    fast_bias_correction=True,
    subset_size=256,
    target_device=nncf.TargetDevice.NPU,          # enforces NPU constraints
)
ov.save_model(quantized, "model_npu.xml")
```

Note **[MEASURED]**: `nncf` and `optimum` are **not installed** on this machine,
so nothing above has been run here. `pip install nncf optimum-intel` first.

**[UNVERIFIED]** The archived research claims "3.8x-5.2x faster than FP16 with
>99.5% accuracy retained". Not measured here. Measure on your model.

---

## 5. Measuring energy, since latency is the wrong metric

Section 1 establishes the NPU loses on latency, so its case must be made on
power. `lunar_core/power_telemetry.py` reads Intel RAPL via the Windows PDH API:

```
\Energy Meter(rapl_package0_pkg)\Power
\Energy Meter(rapl_package0_pp0)\Power     # cores
\Energy Meter(rapl_package0_pp1)\Power     # uncore
\Energy Meter(rapl_package0_dram)\Power
\Thermal Zone Information(\_TZ.TZS0)\Temperature
```

**Always check `is_live` before using any value.** When the counters are
unavailable the module returns hardcoded values (18.5 W package, 51 C) tagged
`"Simulated RAPL Baseline"`. Treating those as real is how fabricated power
numbers entered the old docs.

Also note:
- `npu_power_est_w` is a **made-up formula** over uncore power, not a sensor.
  There is no per-engine NPU power rail exposed here.
- The package rail is **shared**. You cannot attribute watts to the NPU by
  subtraction. The only sound measurement is a **differential experiment**: run
  the workload on CPU for N seconds and integrate package power, then run the
  same workload on NPU, and compare total energy.
- `RAPLPowerGovernor` **does not govern anything** - it reads counters and
  returns a status string. No actuation, no control loop.

---

## 6. Constraints that are real

### 6.1 There is no zero-copy path implemented
**[MEASURED]** `LevelZeroUSMBridge` in `engine.py` loads `ze_loader.dll` and
never calls a Level Zero function. It allocates ordinary host memory via
`VirtualAlloc` and fabricates the "NT handle" from the wall clock. A real path
needs `IDXGIOutputDuplication` -> shared NT handle -> `zeMemAllocShared` /
`zeMemOpenIpcHandle` -> `ov.RemoteTensor`. Note from 2.2 that **only the GPU
advertises `GPU_USM_MEMORY`**; target the GPU for this, not the NPU.

### 6.2 Desktop isolation blocks screen capture in agent runners
Background and agent-runner processes execute on a non-interactive Windows
desktop station (`WinSta0\exebox-...`). Windows DWM will not let them read
pixels from the interactive `Default` desktop, so `dxgi_capture.py` returns an
all-zero frame there. This is a genuine OS constraint, not a bug - it will not
be fixed by changing the capture code. Test screen capture from an interactive
session.

### 6.3 Static shapes
The NPU compiler requires static shapes with known upper bounds.
`vector_memory.py` already reshapes dynamic inputs to `[1, seq_len]` before
compiling. Section 11.1 shows what happens when you skip this: the compile fails
outright with "Upper bounds are not specified".

### 6.4 Compile caching
Set `CACHE_DIR` in the compile config. Cold compile is ~99 ms (NPU) and
~1219 ms (GPU) **[MEASURED]**; cached blob load is far cheaper. `EXPORT_IMPORT`
is in the capability list for all three devices.

---

## 7. Model assets present on this machine **[MEASURED]**

Under `~/.tools/npu/models/`:

| Path | What it is | Size |
| :-- | :-- | --: |
| `bge_real/` | BGE embedding model + tokenizer, OpenVINO IR | 104.5 MB |
| `embed/minilm_l6.*` | MiniLM-L6 embeddings | 25.7 MB |
| `slm_real/` | Qwen2.5-Coder-0.5B-Instruct int4 | - |
| `whisper_real/` | Whisper encoder + decoder + tokenizer | - |
| `yolo_real/yolo11n.*` | YOLO11n detector | - |
| `lcm_real/` | LCM: text_encoder, unet, vae_encoder, vae_decoder | - |
| `vision/mobilenet_v3.*` | MobileNetV3 | - |

These are real trained weights and are the honest core of the project. Two
cautions:

1. **`vector_memory.py` silently falls back to random projections** if
   `bge_real/` is missing. It now emits a `RuntimeWarning` and sets
   `is_degraded`, surfaced at `/api/health` as `embeddings_degraded`. Check it.
2. **YOLO11n is trained on COCO** (people, cars, animals). It cannot detect UI
   elements. Any "screen understanding" built on it needs OCR/DBNet instead.

---

## 8. Things in this repo that are NOT what their names say

Short index so you do not have to rediscover these. Full detail in `CLAUDE.md`
section 2.

| Name | Reality |
| :-- | :-- |
| `build_hazard_classifier_openvino` | Untrained random MLP. Constant output 0.0474. Advisory only. |
| `LevelZeroUSMBridge` | No Level Zero, no D3D11, no zero-copy. |
| `SpeculativeRingBuffer` | Not lock-free, not cache-aligned. Plain list under the GIL. |
| `ShaveDSPSpectralProcessor` | numpy on the CPU. Aliased to `MelSpectrogramCPU`. |
| `SRAMAdamW` | numpy arrays in host DRAM. Correct AdamW, wrong name. |
| `build_draft_model_openvino` | Untrained. **[MEASURED]** speculation runs at 0.0017x, ~580x slower than plain decoding. |
| `GeodesicMicroRouter.route_geodesic` | `arccos(cosine)` is monotonic in cosine, so it ranks identically; the distances are computed, logged, then discarded. |
| `RAPLPowerGovernor` | Monitor, not a governor. |
| `HyperdimensionalMemoryEngine.query` | SHA-256 per whole string; one word changed gives ~0.50 similarity, i.e. noise. |
| `srnc.py` | Cassowary/APCA/CST code from the unrelated `agent-craft` project. Nothing to do with the NPU. |

---

## 9. Useful OpenVINO patterns

```python
import openvino as ov, openvino.opset13 as ops, numpy as np

core = ov.Core()

# Build a graph directly, no framework import needed.
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

Timing rule: **always warm up** (the first `infer()` includes lazy setup), then
take a median over at least 30 runs. A single timed call is noise.

---

## 10. Provenance

Written 2026-09-11 from a direct audit of the codebase plus live probes on this
machine. Measurements came from: `openvino.Core()` property reads; a per-device
compile and infer sweep; a batch-scaling sweep; a cross-process concurrency
experiment; `openvino_genai.LLMPipeline` runs on each device; a
`DualStageSiliconCircuitBreaker` input sweep; a
`HyperdimensionalMemoryEngine` paraphrase-recall probe; a
`GeodesicMicroRouter` ranking comparison; and `pytest tests/`.

When you add a fact here, tag it, and say how it was obtained.

---

## 11. Can an agentic loop run on the NPU? (measured 2026-09-11)

Three hypotheses, each tested rather than argued. Summary: **the main LLM cannot
run on this NPU at all, and the NPU is not free parallel capacity - but it is
meaningfully less disruptive than the GPU, which gives it a real if narrow
role.**

### 11.1 The LLM does not compile on the NPU **[MEASURED]**

`Qwen2.5-Coder-0.5B-Instruct-int4` (`~/.tools/npu/models/slm_real/`) has inputs
`input_ids [?,?]`, `attention_mask [?,?]`, `position_ids [?,?]`, `beam_idx [?]`,
and is stateful (48 sinks). Compiling it for NPU fails:

```
[IE::FrontEnd::importNetwork] Upper bounds are not specified for node
  '__module.model.embed_tokens/ov_ext::embedding/Gather'
  input '0' bounds are '[9223372036854775807, ...]'
[vpux-compiler] Got non broadcastable dimensions pair ...
```

The NPU compiler requires static upper bounds; a dynamic-shape LLM IR has none.

The documented workaround is OpenVINO GenAI's NPU path, which statically
reshapes the graph. `openvino_genai` 2026.2.1 is installed, and that path also
fails on this model:

```
og.LLMPipeline(MODEL, "NPU", MAX_PROMPT_LEN=1024, MIN_RESPONSE_LEN=128)
  -> [vpux-compiler] StopLocationVerifierPass Pass failed :
     Found 80 duplicated names after full verification
```

That is a compiler or export defect (duplicate node names in this int4 export),
not a shape problem. Re-exporting with `optimum-intel` might clear it, but
`optimum` and `nncf` are **not installed**, so this is untested. Treat "LLM on
the NPU" as unproven here, not merely slow.

### 11.2 Decode is bandwidth-bound, which is why no accelerator helps **[MEASURED]**

Generating 32 tokens from the same 0.5B model, greedy, via
`openvino_genai.LLMPipeline`:

| Device | Compile/load | Generate | Throughput |
| :-- | --: | --: | --: |
| CPU | 1.3 s | 584 ms | **54.8 tok/s** |
| GPU | 2.3 s | 597 ms | **53.6 tok/s** |
| NPU | - | - | will not compile |

The GPU has roughly 4x the CPU's FP16 compute (31.9 vs ~8 TFLOPS) and it buys
**nothing** - the two are within 2%. That is the signature of a
memory-bandwidth-bound workload. All three engines share the same LPDDR5X-8533
bus, so no amount of accelerator compute changes batch-1 decode. Any plan that
depends on an NPU making token generation faster is defeated by this before any
of the compiler issues matter.

Prefill, by contrast, is compute-bound and the GPU does win there
**[MEASURED]**, using the raw IR at fixed sequence length:

| seq_len | CPU | GPU |
| --: | --: | --: |
| 128 | 588 tok/s | **2,565 tok/s** |
| 512 | 479 tok/s | **2,690 tok/s** |

So for long agentic prompts the GPU is the right prefill device, roughly 5x the
CPU. The NPU could in principle compete here - it is the one regime where it
has a case - but 11.1 means it cannot be tested until the model compiles.

### 11.3 The NPU is not free capacity, but it is cheaper than the GPU **[MEASURED]**

The strongest argument for an NPU is not speed but separateness. Measuring
foreground GPU latency (1024-dim, 8-layer MLP, batch 1, median of 120) while a
**separate process** saturates each engine:

| Background load | Foreground GPU median | vs idle |
| :-- | --: | --: |
| idle (baseline) | 0.244 ms | - |
| CPU busy-spin (no inference) | 0.243 ms | **-0.4%** |
| NPU saturated | 0.360 ms | **+47.5%** |
| GPU saturated | 0.546 ms | **+123.8%** |

Baseline re-checked after the run: 0.236 ms (-3.4% drift), so the effects are
well clear of noise.

Read this carefully, because both halves matter:

- A fully busy CPU core costs the foreground **nothing** (-0.4%). So the NPU's
  +47.5% is *not* driver or submission CPU overhead. It is contention for shared
  memory bandwidth and power budget.
- But saturating the NPU is **~2.6x less disruptive** than doing the same work
  on the GPU (+47.5% vs +123.8%).

**The design rule that follows:** NPU work should be *duty-cycled and
bandwidth-light*, never saturating. A periodic, bursty, small-tensor background
task is nearly free. A continuously saturated NPU degrades the foreground
noticeably - you have moved the contention, not removed it.

> Methodology note: a first version of this experiment ran the background load
> in a Python **thread** and reported ~+70% for all three loads, including CPU.
> That was GIL contention inside the measurement harness, not hardware. The
> separate-process CPU control is what exposed it. If you repeat this, keep the
> control.

### 11.4 Batch amortises the dispatch floor, but never enough to win **[MEASURED]**

512-dim, 4-layer MLP, items/sec:

| Batch | CPU it/s | GPU it/s | NPU it/s | Best |
| --: | --: | --: | --: | :-- |
| 1 | **20,222** | 8,358 | 3,823 | CPU |
| 8 | **103,159** | 70,083 | 12,707 | CPU |
| 32 | 165,375 | **244,929** | 119,805 | GPU |
| 128 | 195,360 | **812,957** | 420,016 | GPU |

Batching helps the NPU enormously (3.8k to 420k items/s, a 110x gain) because
the ~0.22 ms dispatch floor amortises. But the GPU scales faster still and stays
~2x ahead. **Batch anything you put on the NPU** - but batching does not make it
the fastest device, only a viable one.

### 11.5 What the NPU is actually for, here

Given the above, the defensible role is **small, batched, duty-cycled auxiliary
models running alongside a foreground LLM on CPU or GPU**:

- Embedding batches for retrieval (batch >= 32, periodic, not continuous)
- The command-guard classifier, once trained
- Wake-word and VAD, which are genuinely always-on and tiny
- Rerankers and intent classifiers

And the honest framing for all of it is energy and foreground preservation, not
latency. Which means it has to be measured on those axes - see section 5 - and
nothing in this repo has been yet.

---

## 12. REFUTED: the NPU as a deterministic "deadline engine" **[MEASURED]**

Recorded because a tested-and-failed hypothesis is worth as much as a confirmed
one, and because this one is attractive enough that someone will propose it again.

**The hypothesis.** An NPU runs a statically compiled graph on a software-managed
scratchpad, with no cache hierarchy, no dynamic scheduling, and no contention
from the desktop compositor. That should give it a tight tail latency even when
the machine is busy - making it the only engine on the chip that can promise a
deadline. That would be a real product claim (real-time audio, frame-locked
work, live captioning) and nobody markets NPUs on it.

**The test.** 512-dim 4-layer MLP, batch 1, 1200 iterations, p99.9 measured with
all three devices compiled and warmed *before* load was applied. Load was 9
separate processes: one saturating the GPU, eight spinning CPU cores.

Quiet:

| dev | median | p99.9 | p99.9/median |
| :-- | --: | --: | --: |
| CPU | 0.046 ms | 0.139 ms | 3.0x |
| GPU | 0.078 ms | 0.189 ms | 2.4x |
| NPU | 0.256 ms | 0.682 ms | 2.7x |

Loaded (GPU saturated + all 8 CPU cores busy):

| dev | median | p99.9 | p99.9/median |
| :-- | --: | --: | --: |
| CPU | **0.135 ms** | 51.986 ms | **384x** |
| GPU | 4.682 ms | 15.432 ms | 3.3x |
| NPU | 2.291 ms | **14.896 ms** | 6.5x |

**Verdict: refuted.** On a quiet machine the NPU has no determinism advantage at
all (2.7x vs 2.4x and 3.0x). Under load its p99.9 degrades 22x, from 0.682 ms to
14.9 ms - statistically indistinguishable from the GPU's 15.4 ms. There is no
deadline guarantee here.

**Why, and this is the deepest constraint found in this audit:** you cannot
reach the NPU without a CPU core. Every inference needs a host thread to submit
the request and wait on it. When all cores are saturated the NPU becomes
effectively unreachable regardless of how idle the silicon is. This also caps
the "offload to free the CPU" argument - the submission path is CPU work.

A related practical note: an earlier version of this experiment compiled inside
the measurement loop, and **NPU compilation failed outright** under full CPU
saturation, because `vpux-compiler` itself runs on the host
(`COMPILATION_NUM_THREADS = 8`). Compile early and cache the blob; never compile
on a loaded machine.

**What did survive.** Two narrower but real findings:

1. Under heavy load the NPU's median (2.29 ms) is **2x better than the GPU's**
   (4.68 ms), with comparable tails. If the GPU is already contended - a game, a
   call, a render - the NPU is the better place for auxiliary inference.
2. The CPU keeps an excellent median under load (0.135 ms) but its p99.9 blows
   out to **52 ms**. So for anything latency-sensitive, the CPU is the fastest
   choice on average and the worst choice at the tail. That tradeoff is worth
   knowing and is not what anyone assumes.

---

## 13. CONFIRMED: SSM/Mamba is the architecture this NPU can actually run **[MEASURED]**

This section corrects an error in the earlier analysis. Sections 11 and 12
concluded that the NPU had no viable role. That conclusion was drawn entirely
from transformer workloads, and it does not generalise. **Every NPU failure
recorded in this document is transformer-specific**, and a state-space model
inverts all four.

| Constraint found | Transformer | SSM / Mamba |
| :-- | :-- | :-- |
| NPU needs static shapes (11.1) | dynamic `[?,?]`, will not compile | one token in, fixed state; **compiles** |
| Decode is bandwidth-bound (11.2) | streams weights + growing KV cache | streams constant-size state |
| State size vs context | KV cache grows without bound | constant, independent of context |
| ~0.22 ms dispatch floor (1) | n/a | amortised by fusing all layers into one graph |

### 13.1 The recurrence is correct and runs on the NPU **[MEASURED]**

`lunar_core/mamba_ssm.py` loads genuine Mamba-130M weights from HuggingFace
safetensors (`backbone.layers.0.mixer.A_log`, `.D`, `.dt_proj.bias`) and applies
the correct zero-order-hold discretisation:

```
A     = -exp(A_log)
dt    = softplus(dt_bias)
A_bar = exp(dt * A)        measured range [0.00000, 0.99878]  -- stable
```

Verified against an independent numpy reference
`h = A*h + B*x ; y = sum(C*h) + D*x`:

| device | compiles | max abs error vs reference |
| :-- | :-- | --: |
| CPU | yes | 2.384e-07 |
| GPU | yes | 1.588e-03 (FP16 rounding) |
| **NPU** | **yes** | 1.588e-03 (FP16 rounding) |

Running 4000 sequential steps on the NPU: state stayed finite, `max|h| = 3.332`,
no drift or blowup.

### 13.2 Per-token cost is flat; the transformer's is not **[MEASURED]**

NPU recurrence, per-token latency by context position:

| context | 100 | 500 | 1000 | 2000 | 4000 |
| :-- | --: | --: | --: | --: | --: |
| ms/token | 0.413 | 0.409 | 0.371 | 0.486 | 0.384 |

Flat. Real Qwen2.5-0.5B-int4 decode over the same range, measured with the KV
cache genuinely populated by a prefill:

| context | 64 | 256 | 1024 | 2048 | 4096 |
| :-- | --: | --: | --: | --: | --: |
| CPU ms/token | 17.4 | 22.7 | 28.0 | 35.6 | 34.5 |
| growth vs ctx=64 | 1.00x | 1.30x | 1.60x | **2.04x** | 1.98x |

The transformer roughly doubles by 2K context. The GPU curve is noisier, 13.3 ms
at ctx=256 rising to 24.1 ms at 2048 (about 1.8x), because each context used a
fresh infer request and the first is cold. The CPU curve is the clean one.

### 13.3 State size: the structural argument **[computed from architecture]**

Mamba-130M state is `24 layers * 1536 d_inner * 16 d_state`, constant:

| | memory | vs Mamba |
| :-- | --: | --: |
| Mamba-130M state (any context) | **1.12 MB** | 1x |
| Qwen-0.5B KV at 1K ctx | 12 MB | 10.7x |
| Qwen-0.5B KV at 4K ctx | 48 MB | 42.7x |
| Qwen-0.5B KV at 16K ctx | 192 MB | 170.7x |
| Qwen-0.5B KV at 64K ctx | 768 MB | 682.7x |

On a device where decode is bandwidth-bound and memory is shared with the CPU
and GPU over one 136 GB/s bus, this is the whole argument.

### 13.4 Full-depth Mamba compiles on the NPU, fused **[MEASURED]**

Fusing all layers into ONE graph matters: 24 separate dispatches at the ~0.22 ms
floor would cost about 5 ms/token in overhead alone. Fused, per token:

| config | CPU | GPU | NPU | state |
| :-- | --: | --: | --: | --: |
| 64x16, 1 layer (the repo toy) | 0.042 ms | 0.086 ms | 0.327 ms | 4 KB |
| 1536x16, 1 layer | 0.065 ms | 0.103 ms | 0.390 ms | 0.09 MB |
| 1536x16, 4 layers | 0.203 ms | 0.287 ms | 0.611 ms | 0.38 MB |
| **1536x16, 24 layers (Mamba-130M depth)** | 1.884 ms | 1.585 ms | **2.108 ms** | 2.25 MB |

Note what happens to the gap. On the toy the NPU is 7.8x slower than the CPU; at
full depth it is **1.12x** slower than the CPU and **1.33x** slower than the GPU.
Real work per dispatch finally amortises the floor. This is the first workload
measured in this document where the NPU is competitive.

### 13.5 What is NOT there - read before believing the above

The implementation is a correct SSM **kernel**, not a Mamba language model:

1. **B and C are fabricated, not loaded.** `B_bar = dt * ones` and
   `C = ones / sqrt(d_state)`. In Mamba both are *input-dependent* - that
   selectivity is the entire contribution of S6 over S4. What runs here is a
   linear time-invariant SSM using the Mamba A matrix. Fixing this is the single
   most important piece of work.
2. **`np.resize` from (1536,16) to (64,16)** tiles and truncates, scrambling
   which channel is which.
3. **No embedding, no LM head, no in_proj / conv1d / gating / out_proj**, and
   only layer 0 A_log. It cannot generate text.
4. **The 2.108 ms above is the scan only.** A real Mamba block is dominated by
   its projections (`d_model -> 2*d_inner` in, `d_inner -> d_model` out), which
   are most of the 130M parameters. Expect a full model to be several times this.
5. **Prefill is the hard problem.** Sequential recurrence at 2.1 ms x 4096 tokens
   is 8.6 s, which is unusable. This needs the chunked associative scan, covered
   in `docs/MAMBA_SSM_REFERENCE.md` chapters 8 and 12.

### 13.6 What to build toward

In order:

1. Export a real Mamba (Mamba-130M to start, then Mamba-2 or Falcon-Mamba-7B)
   to OpenVINO IR with static per-token shapes. No public Mamba NPU IR exists.
   This is both the work and the moat.
2. Wire real selective B and C. They are input-dependent but still fixed-shape
   per token, so they stay NPU-compatible.
3. Keep all layers fused into one graph; 13.4 shows why.
4. Implement the chunked parallel scan for prefill.

If that lands, the claim is: **a local language model running on the NPU with
unbounded context at constant memory, on hardware where transformers will not
compile at all.** That is worth building, and nobody has shipped it.

---

## 14. GATE 0 result: FP16 caps usable SSM context at roughly 10K tokens **[MEASURED]**

Section 13.6 proposed a long-context Mamba on the NPU: unbounded context at
constant memory. Before building it, the cheapest kill shot was run -- does the
recurrence survive tens of thousands of steps at the precision the NPU actually
computes in? It does not, and the failure is now precisely characterised.

### 14.1 The measurement

Real Mamba-130M layer-0 discretised `A_bar` (1536x16), 20,000 sequential steps,
each device running the identical OpenVINO graph, compared against an fp64
numpy reference:

| device | @500 | @5K | @10K | @20K | final max\|h\| |
| :-- | --: | --: | --: | --: | --: |
| CPU | 0.00% | 0.00% | 0.00% | **0.00%** | 1.1953 |
| GPU | 0.31% | 2.64% | 5.30% | **27.90%** | 1.3203 |
| NPU | 0.32% | 2.58% | 4.25% | **27.49%** | 1.3018 |

(fp64 reference final max\|h\| = 1.1956.)

**This is an FP16 problem, not an NPU problem.** The GPU drifts identically
(27.90% vs 27.49%) because both run fp16. Only the CPU, which computes in fp32,
is clean. The NPU's disadvantage is simply that it *cannot opt out* -- see 2.1,
`DEVICE_GOPS float32 = 0.0`.

Drift grows super-linearly: 0.32% -> 2.58% -> 4.25% -> 27.49%.

### 14.2 Root cause, precisely located

In the real layer, `A_bar` spans [0.0, 0.99999]. Nothing is exactly 1.0 in
fp64 -- but **163 of 24,576 values (0.66%) round to exactly 1.0 in fp16**.
Those channels become pure accumulators: `h = h + B*x`, no decay, forever.

FP16 has an 11-bit mantissa, so an accumulator hits **absorption**: once \|h\|
is large enough, small increments fall below half a ULP and are silently
discarded.

| \|h\| | fp16 ULP | increments below this are LOST |
| --: | --: | --: |
| 1 | 0.00098 | 0.00049 |
| 64 | 0.06250 | 0.03125 |
| 1024 | 1.00000 | 0.50000 |

Measured directly on a pure accumulator channel: with increments ~N(0, 0.01),
the first silently-absorbed update occurs at **step 630**.

The remaining 86.9% of channels have `A_bar < 0.9`, decay within tens of steps,
and self-correct. The drift is concentrated almost entirely in that 0.66%.

### 14.3 Mitigations tested

**fp32 state -- works, but is UNAVAILABLE on the NPU.** In numpy, keeping the
state in fp32 while weights stay fp16 roughly halves the error (17.8% -> 9.26%
at 65K in that harness). On device it does not survive: feeding a state whose
detail sits below fp16 ULP and reading it back,

| device | distinct sub-ULP values preserved (of 1,499) |
| :-- | --: |
| CPU | 1,385 |
| GPU | 3 |
| NPU | 3 |

The tensor is declared fp32 and is quantised to fp16 internally regardless.

**Clamping `A_bar` below 1.0 -- REFUTED.** The obvious fix is to clamp to the
largest fp16 value under 1.0 (1 - 2^-11 = 0.99951) so no channel is an unbounded
accumulator. Tested: it made things far worse (17.8% -> 93.9% at 65K).

The reason is instructive. Those 163 near-1.0 channels are the model's
**long-memory** channels -- clamping gives them a time constant of ~2048 steps
and destroys exactly the capability that makes long context work. It is a model
modification wearing the costume of a numerical fix.

### 14.4 Verdict and what it changes

**The "unbounded context at constant memory" pitch does not survive fp16 as
formulated.** Revised, measured limits:

| context | fp16 drift | usable? |
| :-- | --: | :-- |
| < 5K tokens | < 3% | yes |
| ~10K tokens | ~4-5% | marginal |
| 20K+ tokens | 27%+ | no |

The *memory* claim is untouched -- state is still 1.12 MB at any length. It is
the *fidelity* claim that breaks past roughly 10K tokens.

Whether 4-5% state drift changes generated tokens is still unmeasured; it needs
the full model with an LM head. Do not assume it is fatal, and do not assume it
is fine.

### 14.5 The idea this diagnosis produces

The drift is concentrated in **163 identifiable channels out of 24,576**. They
are known at weight-preparation time, before any inference runs.

So: **split the recurrence by channel.** Run the 24,413 well-conditioned
channels on the NPU in fp16, and the 163 accumulator channels on the CPU in
fp32. The CPU share is 0.66% of the work -- arithmetically trivial, and the CPU
was measured clean at 0.00% drift over 20K steps.

This is untested. It is the most promising remaining route to long-context SSM
on this NPU, it falls directly out of the failure analysis, and it is cheap to
try. Two risks to check: the extra dispatch per token (the ~0.22 ms floor
applies to the NPU half regardless), and whether splitting and recombining the
state costs more than it saves.

Alternatives if that fails: INT8 state with an explicit learned scale (the NPU
supports INT8 natively, and a controlled scale may condition better than fp16's
fixed exponent); periodic state recomputation from a checkpoint; or Mamba-2,
whose scalar-per-head `A` parameterisation may be better conditioned than
Mamba-1's per-channel diagonal.

---

## 15. Real production models: where the NPU actually wins **[MEASURED 2026-09-12]**

Sections 1 and 11-14 measured only dense f32/f16 matmul stacks. That is close to
the worst possible workload for this silicon: no INT8 path, no hardware weight
decompression, no convolution hardware. **Section 1's headline claim -- "the NPU
never wins on latency" -- was measured on the NPU's weak path only and is wrong
as stated.** This section replaces it with measurements on real production
models.

### 15.1 Five real models, five repeats each **[MEASURED]**

Fresh compile and fresh infer request per repeat, median of 60 inferences each,
seq len 64 where applicable. A ranking is only reported as trustworthy if the
same device wins all five repeats.

| model | precision | CPU | GPU | NPU | winner |
| :-- | :-- | --: | --: | --: | :-- |
| YOLO11n | int8 | 20.36 ms | 8.74 ms | **5.99 ms** | **NPU 5/5** |
| BGE-base | int8 | 31.28 ms | 4.94 ms | **4.29 ms** | **NPU 5/5** |
| Whisper-tiny encoder | fp16 | 244.75 ms | **8.49 ms** | 10.21 ms | GPU 5/5 |
| MiniLM-L6 | fp16 | 21.79 ms | **1.25 ms** | 1.99 ms | GPU 5/5 |
| MobileNetV3 | fp16 | 2.11 ms | 1.00 ms | 0.93 ms | **UNSTABLE** (GPU 3, NPU 2) |

**The NPU beats the GPU by 1.46x on YOLO11n and 1.15x on BGE-base, consistently.**
Against the CPU it is 3.4x and 7.3x faster respectively.

Run-to-run variation (coefficient of variation across the five repeats):

| model | CPU | GPU | NPU |
| :-- | --: | --: | --: |
| YOLO11n | 1.9% | 5.3% | 4.6% |
| BGE-base | 2.8% | 1.5% | 4.1% |
| Whisper enc | 1.4% | 3.1% | 3.5% |
| MobileNetV3 | 8.0% | **56.7%** | 8.3% |
| MiniLM-L6 | 2.2% | **25.3%** | 11.9% |

MobileNetV3 is unstable because the **GPU** swings wildly on very small graphs
(0.670-2.002 ms), not because of the NPU. Note also that on the two smallest
models the NPU is the *more* consistent device.

### 15.2 Is INT8 the cause? Partly, and the mechanism is NOT what it looks like

Both NPU winners are INT8; all three losers are fp16. That correlation is clean
but it is not proof, so it was tested directly.

**Weight-only INT8 compression gives the NPU no speedup.** Compressing MiniLM,
MobileNetV3 and Whisper with `nncf.compress_weights(INT8_SYM)` and re-running:

| model | CPU speedup | GPU speedup | NPU speedup |
| :-- | --: | --: | --: |
| MiniLM-L6 | 1.97x | 0.44x | **0.91x** |
| MobileNetV3 | 0.99x | 1.56x | **0.81x** |
| Whisper enc | 1.35x | 0.88x | **1.04x** |

The NPU got *slower* on two of three. Also note the accuracy cost: cosine
similarity against the fp16 output was 0.754 for MiniLM and **0.0002 for
MobileNetV3** -- weight-only compression destroyed that model outright. Do not
apply it blindly.

**The obvious explanation is wrong.** The natural theory is that Intel's models
quantise *activations* too (FakeQuantize / QDQ nodes), which is what switches the
DPU into INT8 mode. Checking the graphs:

| model | FakeQuantize nodes | int8 constants | NPU result |
| :-- | --: | --: | :-- |
| YOLO11n | 101 | 88 | wins |
| **BGE-base** | **0** | 150 | **wins** |
| MiniLM fp16 | 0 | 0 | loses |
| MiniLM compress_weights | 0 | 25 | loses |

**BGE-base has zero FakeQuantize nodes** -- it is weight-only INT8, structurally
the same as the compression that did not help MiniLM, and the NPU still wins on
it. So quantised activations are not the explanation.

**Full quantisation does not compile on this NPU.** Running a proper
`nncf.quantize()` with a 64-sample calibration set on MiniLM produces 36
FakeQuantize nodes and then fails:

```
[vpux-compiler] failed to legalize unresolved materialization from
  'tensor<128x384x1x3xf16>' to 'tensor<...!quant.uniform<u8:f16,...>>'
  at __module.transformer.layers.1.self_attn/aten::select/Gather/fq_input_0
```

**Practical rule: use Intel's pre-quantised INT8 models from the OpenVINO hub.
Do not expect to successfully quantise your own for this NPU.**

The most plausible remaining mechanism is weight *bandwidth*: BGE-base carries
108.9M weight elements against MiniLM's 22.3M, so halving weight bytes matters
far more for BGE. This is unproven and left open.

### 15.3 Energy: the NPU loses, and this settles a claim made all session

Sections 1, 5 and 11 all asserted that the NPU's real case is energy rather than
latency, and told the reader to measure it. It had never been measured. It has
now, on BGE-base int8 -- the workload where the NPU is the *fastest* device.

Method: differential package power via Intel RAPL through Windows PDH. Idle
measured before and after each busy window with a 25 s cooldown, three repeats
per device, interleaved in randomised order.

| device | energy per inference | latency | package delta |
| :-- | --: | --: | --: |
| **GPU** | **35.05 mJ** (+/- 1.01) | 4.022 ms | 8.7 W |
| NPU | 51.00 mJ (+/- 2.20) | **3.942 ms** | 12.9 W |
| CPU | 391.80 mJ (+/- 10.18) | 22.175 ms | 17.7 W |

Idle was stable at ~11 W across all nine runs, so the deltas are well clear of
the noise floor.

**The NPU is marginally faster and 45% more expensive in energy than the GPU.**
Both beat the CPU enormously (11x less energy, 5.6x faster).

> A first attempt at this measurement was invalid and reported itself as such:
> idle drifted 13.58-22.39 W between runs, a wider spread than the deltas being
> measured, and the GPU produced a physically impossible negative energy because
> its idle baseline was captured while the package was still hot. The cooldown,
> pre/post idle averaging and interleaving exist to fix that. If you repeat this,
> keep them.

### 15.4 The NPU's real advantage: it barely touches the host CPU **[MEASURED]**

The energy result prompted a hypothesis: perhaps the NPU submission path
busy-waits on a CPU core, and that host cost was being charged to the NPU in the
package rail.

**The hypothesis was refuted, and the truth is the opposite.** Host CPU time
consumed by this process per inference, same model, two repeats:

| device | wall ms/inference | host CPU ms/inference | cores busy |
| :-- | --: | --: | --: |
| CPU | 22.63 | 69.37 | 3.18 |
| GPU | 4.03 | 3.95 | **0.98** |
| **NPU** | **3.78** | **0.40** | **0.11** |

**The GPU driver burns a full CPU core (98%) to keep the GPU fed. The NPU uses
11% of one -- roughly 10x less host CPU for the same work, delivered slightly
faster.**

This is the strongest pro-NPU result in this document, and it is the one thing
the NPU does that neither other engine can:

- Run inference continuously while leaving essentially the whole CPU free.
- The GPU cannot make that claim. A "GPU offload" costs you a core.

It also corrects section 14. The claim there that "you cannot reach the NPU
without a CPU core" is too strong: inference needs only ~11% of a core. What
genuinely requires host CPU is **compilation** (`vpux-compiler`,
`COMPILATION_NUM_THREADS = 8`), which is why compiling failed under full
saturation. Compile early, cache the blob, and the runtime cost is tiny.

### 15.5 Corrected summary of where the NPU stands

| axis | verdict | evidence |
| :-- | :-- | :-- |
| Latency, dense fp16/fp32 | **loses** to CPU and GPU | section 1 |
| Latency, Intel INT8 models | **WINS** 1.15-1.46x vs GPU | 15.1, 5/5 stable |
| Latency, fp16 real models | loses to GPU | 15.1, 5/5 stable |
| Energy per inference | **loses** to GPU by 45% | 15.3 |
| **Host CPU freed** | **WINS decisively, ~10x vs GPU** | 15.4 |
| Foreground interference | better than GPU (+47.5% vs +123.8%) | 11.3 |
| Latency variance, small models | better than GPU | 15.1 |
| Quantising your own models | not viable, compiler fails | 15.2 |
| Long SSM recurrence | fp16 drift caps useful context ~10K | section 14 |

**The honest one-line case for this NPU: it runs Intel's pre-quantised INT8
models slightly faster than the GPU while using a tenth of the host CPU, at a
45% energy premium.** That is a real and narrow niche -- continuous background
inference on a machine whose CPU you want to keep free -- and it is not the
niche the project was built around.

---

## 16. SOLVED: the channel split restores long-context SSM on the NPU **[MEASURED 2026-09-12]**

Section 14 concluded that fp16 drift capped usable SSM context at roughly 10K
tokens, and proposed in 14.5 a fix that had not been tested: run the few
badly-conditioned channels on the CPU and the rest on the NPU. It works, and it
is close to free.

### 16.1 The affected set is tiny

`A_bar` is [1536 d_inner x 16 d_state]. The entries that round to exactly 1.0 in
fp16 -- the pure accumulators responsible for essentially all the drift:

| | count | share |
| :-- | --: | --: |
| entries rounding to 1.0 in fp16 | 163 of 24,576 | 0.66% |
| **rows containing at least one** | **19 of 1,536** | **1.24%** |

The split has to be by row, since a row is one d_inner channel. **19 rows go to
the CPU, 1,517 stay on the NPU.**

### 16.2 It works, and it holds to 64K steps

Relative error of the output against an fp64 reference, real Mamba-130M layer-0
weights, 65,536 sequential steps:

| variant | 1K | 4K | 16K | 32K | 64K |
| :-- | --: | --: | --: | --: | --: |
| all-NPU fp16 | 0.19% | 2.42% | 94.13% | 26.10% | **33.27%** |
| split, CPU side fp32 | 0.09% | 0.04% | 8.80% | 0.23% | **0.32%** |
| split, CPU side fp64 | 0.09% | 0.04% | 8.75% | 0.22% | **0.34%** |

**Drift at 64K steps falls from 33.27% to 0.32% -- roughly 100x better.**

Two readings worth noting:

- **fp32 is enough on the CPU side.** fp64 gives no measurable benefit (0.32% vs
  0.34%), so a real implementation can use ordinary fp32 there.
- **The 16K column is a measurement artifact, not an instability.** Relative
  error is inflated wherever the reference signal `y64` passes near zero. The
  32K and 64K columns (0.23%, 0.32%) show the actual trend, which is flat.

### 16.3 It costs nothing

| variant | ms/step |
| :-- | --: |
| all-NPU fp16 | 0.511 |
| split (CPU fp32) | 0.506 |
| split (CPU fp64) | 0.471 |

The split is **not slower**. The NPU graph shrinks from 1,536 to 1,517 rows,
which offsets the 19 rows of numpy work on the host. An earlier 20K-step run
measured 1.07x; at 65K it is within noise of parity.

This also fits section 15.4: NPU inference costs only ~11% of a host core, so
there is plenty of CPU headroom to absorb 19 rows of fp32 arithmetic per token.

### 16.4 What this changes

Section 14's verdict is superseded:

| context | all-NPU fp16 (section 14) | with channel split |
| :-- | :-- | :-- |
| < 5K | usable | clean (<0.1%) |
| 10K | marginal | clean |
| 20K | unusable (27%) | clean (2.3%) |
| 64K | unusable (33%) | **clean (0.32%)** |

**Long-context SSM on this NPU is viable after all.** The claim "64K context at
1.12 MB of constant state" survives, with one implementation requirement: a
19-row fp32 side-path on the CPU, computed at weight-preparation time.

Remaining caveats unchanged from 13.5 -- this is still an SSM kernel and not a
language model. B and C are fabricated rather than loaded, there is no
embedding, LM head or projection stack, and prefill still needs the chunked
scan. The numerical blocker is cleared; the model-building work is not.

---

## 17. What the "47 TOPS" number is actually worth **[MEASURED 2026-09-12]**

The driver reports `DEVICE_GOPS int8 = 46,694`, and that is the number on the
box. It is a theoretical peak: `6 tiles * 1024 MAC/cycle * 2 OP/MAC * ~1.9 GHz`
at 100% systolic utilisation. This section measures how much of it is reachable.

### 17.1 Method

Build workloads deliberately shaped to saturate the array -- large convolution
stacks with high channel counts and large batches, which is the workload class
NPUs exist for -- and compute achieved FLOP/s from known FLOP counts.
`PERFORMANCE_HINT: THROUGHPUT`. Conv FLOPs = `2*K*K*Cin*Cout*Hout*Wout*batch`.

### 17.2 Results

| workload | GFLOP | NPU TFLOP/s | % of NPU peak | GPU TFLOP/s | % of GPU peak |
| :-- | --: | --: | --: | --: | --: |
| b1 C64 56x56 x8 | 1.8 | 2.11 | 9.1% | 2.21 | 6.9% |
| b1 C128 56x56 x8 | 7.4 | 2.39 | 10.2% | 3.63 | 11.4% |
| b8 C128 56x56 x8 | 59.2 | 3.17 | 13.6% | 7.80 | 24.4% |
| b16 C128 56x56 x8 | 118.4 | 2.79 | 12.0% | 8.10 | 25.3% |
| **b16 C256 28x28 x12** | 177.6 | **5.05** | **21.6%** | **13.47** | **42.2%** |
| b32 C256 28x28 x12 | 355.1 | 5.02 | 21.5% | 12.69 | 39.7% |
| b256 N4096 matmul x8 | 68.7 | 3.23 | 13.8% | 5.65 | 17.7% |

### 17.3 Verdict

**Best achieved on the NPU: 5.05 TFLOP/s fp16 -- 21.6% of its own 23.35 TFLOP/s
fp16 peak.** Utilisation rises with workload size and then plateaus around 21%.

**The GPU reaches 42.2% of its peak -- roughly twice the utilisation efficiency.**

Stating this fairly, because the comparison is easy to abuse:

- These are **fp16** measurements. The 46.69 TOPS headline is an **INT8** figure.
  If INT8 scaled perfectly (2x), the NPU would reach roughly **10 TOPS INT8, or
  about 21% of the advertised number**. Constructing a truly INT8-saturating
  graph by hand was not attempted; 15.2 shows the quantisation toolchain does not
  cooperate on this device anyway.
- 21% of peak is **not scandalous in itself**. Real hardware rarely hits peak.
  What matters is the *comparison*: the GPU on the same machine, with the same
  compiler stack, reaches double the fraction of its own peak.
- The peak number assumes every MAC unit busy every cycle with zero memory
  stalls. Section 11.2 showed this SoC is memory-bandwidth-bound for real work,
  and all three engines share one 136 GB/s LPDDR5X bus.

**So the headline 47 TOPS should be read as a dimensional ceiling, not a
capability.** About a fifth of it is reachable on workloads designed to be
favourable, and the workloads people actually run -- batch-1 LLM decode -- are
bandwidth-bound, where TOPS is the wrong unit entirely.

### 17.4 Independent corroboration

This is not a contrarian local finding; it matches the public record as of
September 2026:

- Intel's own **NPU Acceleration Library is archived and end-of-life**,
  redirecting developers to OpenVINO GenAI.
- Published measurements of Llama-3.2-1B Q4_K_M on a Lunar Lake NPU report
  **2.07 tokens/s, slower than the CPU** on the same model class -- consistent
  with 11.2 here.
- The general industry critique is that TOPS does not predict LLM speed at all,
  because token generation is bounded by memory bandwidth and capacity rather
  than compute, and that headline figures often sum CPU + GPU + NPU.

### 17.5 One actionable lead

OpenVINO 2026.2 documentation recommends **`transformers==4.51.3`** for
generating models targeted at the Intel NPU, with newer versions supported only
in later releases. This machine has **transformers 4.57.6**.

That is a plausible cause of the LLM export failures in 11.1 -- both the dynamic
shape error and the "Found 80 duplicated names" compiler failure could come from
an export produced by an unsupported Transformers version. Worth testing in an
isolated virtual environment before concluding that LLM-on-NPU is unavailable
here. Do not downgrade the main environment; other work depends on 4.57.6.

---

## 18. Designing FOR the NPU: what can and cannot be derived **[MEASURED 2026-09-12]**

Everything so far has been about porting existing models TO this NPU. The better
question is what shape of model the silicon actually wants. That requires a
performance model -- a way to predict what will be fast before building it.

**Four candidate mechanisms were tested. All four were refuted.** The honest
conclusion is that a usable performance model for this device does not exist
yet, which is itself the most important finding in this section.

### 18.1 Refuted: quantised activations explain the INT8 wins

15.2. BGE-base has **zero** FakeQuantize nodes -- weight-only compression, the
same structure that gave MiniLM no benefit -- and the NPU still beats the GPU on
it consistently.

### 18.2 Refuted: SRAM residency

If NPU 4 has ~12 MB of on-die scratchpad, a model whose weights fit entirely
inside it should never stream from DRAM, and there should be a sharp cliff past
that size. Sweeping matmul stacks from 2 MB to 128 MB of weights at batch 8:

| weights | NPU ms | NPU TFLOP/s | GPU ms |
| --: | --: | --: | --: |
| 2.0 MB | 0.589 | 0.03 | 0.199 |
| 9.0 MB | 1.659 | 0.05 | 0.391 |
| **12.0 MB** | 2.237 | 0.04 | 0.322 |
| 16.0 MB | 2.655 | 0.05 | 0.369 |
| 36.0 MB | 5.015 | 0.06 | 0.561 |
| 128.0 MB | 14.067 | 0.08 | 2.738 |

**No cliff.** Efficiency is flat across the 12 MB boundary and in fact rises
slightly with size. Weight residency is not the limiting factor.

### 18.3 Refuted: the NPU is weight-bandwidth bound

The sweep above appeared to show NPU time scaling linearly with weight bytes at
a flat ~9 GB/s, which looked like a clean bandwidth limit. **That reading was
wrong, and it was an artifact of holding batch fixed at 8.**

Holding the weights fixed (16 MB) and varying only batch:

| batch | NPU GB/s | NPU TFLOP/s | GPU GB/s | GPU TFLOP/s |
| --: | --: | --: | --: | --: |
| 1 | 29.3 | 0.029 | 35.0 | 0.035 |
| 4 | 12.0 | 0.048 | 51.8 | 0.207 |
| 16 | 27.5 | 0.440 | 41.0 | 0.656 |
| 64 | 17.7 | 1.136 | 33.6 | 2.148 |
| 256 | 7.2 | 1.842 | 9.7 | 2.480 |
| 1024 | 2.3 | **2.401** | 4.3 | **4.444** |

A bandwidth-bound device holds GB/s constant as batch rises. A compute-bound one
holds TFLOP/s constant. **Neither happens.** GB/s swings from 29 to 2, TFLOP/s
climbs from 0.03 to 2.4 and then plateaus. The device is dispatch-dominated at
small batch and approaches a soft compute ceiling at large batch, with no clean
regime in between.

### 18.4 Refuted: halving weight bytes speeds up the NPU

The sharpest prediction from the bandwidth theory: INT8 halves the bytes, so it
should roughly double throughput. Same graphs, weight-only INT8:

| config | fp16 | int8 | speedup |
| :-- | --: | --: | --: |
| N1024 x8 b8 | 2.676 ms | 2.735 ms | **0.98x** |
| N1536 x8 b8 | 5.408 ms | 5.566 ms | **0.97x** |
| N2048 x8 b8 | 7.570 ms | 7.971 ms | **0.95x** |

Halving the weight bytes changes nothing. Weight volume is not the limit.

### 18.5 What this means

**The YOLO11n and BGE-base wins are real, reproducible (5/5), and mechanistically
unexplained.** Four plausible mechanisms have been tested and eliminated. Without
a performance model you cannot derive a good architecture for this chip -- you can
only search empirically.

That is a legitimate research programme. It is not a product plan, and it should
not be mistaken for one.

### 18.6 What CAN be derived, and it is not nothing

Hard constraints, all measured, all reliable:

| constraint | value | source |
| :-- | :-- | :-- |
| Static shapes only | dynamic IR will not compile | 11.1 |
| FP16 / INT8 only, no FP32 | `DEVICE_GOPS float32 = 0.0` | 2.1 |
| Dispatch floor | ~0.22 ms per call | 1 |
| Achievable fraction of peak | ~22% (GPU: 42%) | 17.2 |
| **Convolutions beat matmuls** | **5.05 vs 2.40 TFLOP/s** | 17.2, 18.3 |
| Host CPU per inference | **0.40 ms vs GPU's 3.95 ms** | 15.4 |
| Quantising your own models | toolchain fails | 15.2 |

From those, a model designed **for** this silicon should be:

1. **Convolutional or state-space, not attention-heavy.** Convolutions hit
   roughly 2x the TFLOP/s of matmuls here. This is the clearest architectural
   signal in the data.
2. **Fixed-shape end to end**, no dynamic control flow, no growing cache.
3. **Fused into one graph per invocation**, because six small calls cost six
   dispatch floors.
4. **Justified by the CPU-freeing property, not by speed.** 0.40 ms of host CPU
   per inference is the one advantage no other engine on this chip offers, and
   it is architecture-independent.

That points away from "a small LLM" and toward **continuous streaming perception**
-- always-on audio event detection, ambient activity understanding, sensor
fusion. Workloads that run for hours, where taking a tenth of a core instead of a
whole one is the entire product argument, and where nobody cares whether a single
inference took 4 ms or 6 ms.

The SSM work in sections 13-16 fits that profile exactly: fixed shape, constant
state, streaming input, runs forever.

---

## 19. The architecture this NPU actually wants **[MEASURED 2026-09-12]**

Section 18 concluded that no performance model exists for this device. This
section builds the beginning of one by testing architectures directly, and lands
on a concrete, measured recommendation.

### 19.1 The relevant mathematics

A discrete state-space model is `h_t = A h_{t-1} + B x_t`, `y_t = C h_t`.
Unrolling gives

```
y_t = sum_{k>=0} (C A^k B) x_{t-k}
```

**An SSM is exactly a convolution** whose kernel is `(CB, CAB, CA^2 B, ...)`.
The same weights have two exact representations -- recurrent (O(1) per step,
constant memory, strictly sequential) and convolutional (parallel over a window,
no carried state). This is the S4 / Mamba-2 duality, and it means "convolutional
or state-space" is a false choice: you pick the *representation* that suits the
hardware.

Per-output-token arithmetic, for `d` width, `N` state, `K` kernel, `L` layers:

| architecture | FLOP per token | receptive field |
| :-- | --: | :-- |
| SSM recurrent | `2*d*N` | **unbounded** |
| Dilated TCN | `2*L*K*d^2` | `1+(K-1)(2^L - 1)` |
| Attention | `2*d*T` | T, and **grows** |

At d=256, N=16, K=3, L=10: SSM = **8,192** FLOP/token, TCN = **3,932,160**.
**The SSM has a 480x arithmetic advantage, for more context.** On paper it is
not close.

### 19.2 The measurement inverts it

Measured cost per output token, both architectures, varying chunk size:

| chunk | SSM NPU | SSM CPU | TCN NPU | TCN CPU | TCN/SSM on NPU |
| --: | --: | --: | --: | --: | --: |
| 1 | 386.15 us | 47.85 | 272.55 us | 58.20 | 0.71x |
| 4 | 113.06 | 19.81 | 61.38 | 34.60 | 0.54x |
| 16 | 46.42 | 8.62 | 18.61 | 19.75 | 0.40x |
| 64 | 34.97 | 7.51 | 5.77 | 12.89 | 0.17x |
| **256** | 24.17 | 7.08 | **3.31** | 11.51 | **0.14x** |

**The TCN is 7x FASTER than the SSM on the NPU, despite doing 480x more
arithmetic.** The gap runs the wrong way and widens with chunk size.

**Why:** the recurrent SSM unrolls into `chunk` sequential, serially-dependent,
tiny elementwise operations. A systolic array cannot do anything with that. The
TCN is ten large convolutions -- exactly the shape the hardware exists for.

This is precisely the problem Mamba-2's SSD chunked algorithm was invented to
solve: convert the sequential scan into parallel matmuls. A naive unrolled
recurrence is the thing you must not build.

### 19.3 The design rule that replaces "count FLOPs"

> **On this NPU, FLOP count does not predict performance. Parallel graph
> structure does.** 3.9 MFLOP of convolution beats 8 KFLOP of sequential
> elementwise work by 7x.

This sharpens 18.6's "convolutions beat matmuls". The real variable is not the
operator type, it is whether the work is expressible as a few large parallel
tensor ops or as many small dependent ones.

### 19.4 The concrete recommendation

At chunk 256 the dilated TCN is the first workload in this entire document that
the NPU wins **on a graph designed here rather than shipped by Intel**:

| | NPU | CPU | GPU |
| :-- | --: | --: | --: |
| TCN, chunk 256, d=256, 10 layers | **3.31 us/token** | 11.51 | 1.58 |

**3.5x faster than the CPU.** The GPU is still quicker in wall-clock, but 15.4
applies: the NPU does it using 0.40 ms of host CPU per inference against the
GPU's 3.95 ms.

And receptive field is nearly free in this regime. From experiment K, holding
chunk at 256:

| layers | receptive field | NPU us/token | RF per us/token |
| --: | --: | --: | --: |
| 2 | 7 | 1.67 | 4.2 |
| 4 | 31 | 3.23 | 9.6 |
| 6 | 127 | 4.15 | 30.6 |
| 8 | 511 | 3.69 | 138.3 |
| **10** | **2,047** | **3.32** | **616.1** |

Receptive field grows 292x from 2 to 10 layers while cost roughly doubles,
because dispatch dominates. **Depth is close to free; buy context with layers.**

### 19.5 So: build this

A model designed for this silicon, derived from measurement rather than
preference:

- **Dilated causal convolution stack**, ~10 layers, kernel 3, dilation doubling.
  Receptive field ~2,000 timesteps at ~3.3 us/token.
- **Chunked at 256 timesteps**, one fused dispatch per chunk. That amortises the
  0.22 ms floor; the cost is 256 timesteps of latency before a chunk's outputs
  appear.
- **Fixed shape end to end**, no dynamic control flow.
- **If long memory beyond the receptive field is needed**, add an SSM in its
  *chunked parallel (SSD) form* -- never as an unrolled recurrence.
- **Justified on host CPU**, not wall-clock: a tenth of a core, continuously.

For a 16 kHz audio stream at 10 ms hops (100 frames/s), chunk 256 is 2.56 s of
latency -- fine for event detection and activity classification, useless for
interactive voice. That latency budget is the main design constraint this
architecture imposes, and it should drive which use cases are chosen.
