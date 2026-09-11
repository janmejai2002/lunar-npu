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

> **[MEASURED] The NPU is the slowest of the three engines at every workload
> size tested. It has a fixed ~0.22 ms dispatch floor. It never wins on latency.**

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
