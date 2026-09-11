# Chapter 2: Driver Stack, Compiler Internals & MLIR Dialects for Intel NPU 4

---

## 2.1 The NPU Driver & Runtime Architecture

Executing neural workloads on Intel Lunar Lake NPU 4 involves a multi-tier runtime stack spanning user-space libraries, hardware abstraction interfaces, compiler plugins, and kernel-mode device drivers.

```
+========================================================================================================+
|                                    APPLICATION & FRAMEWORK LAYER                                       |
|                  Python • OpenVINO GenAI • PyTorch (torch.compile) • C++ Applications                   |
+========================================================================================================+
                                                    |
                                                    v
+========================================================================================================+
|                                     OPENVINO RUNTIME (NPU PLUGIN)                                      |
|            Graph Optimization • Model Caching • Device Dispatcher • Precision Lowering                  |
+========================================================================================================+
                                                    |
                                                    v
+========================================================================================================+
|                                     VPUX COMPILER ENGINE (MLIR)                                        |
|      Dialect Lowers: IE -> VPU -> VPUIP -> VPURT • Tile Scheduler • Static Memory Allocator           |
|                Hardware Code Generation: DPU Microcode + SHAVE v4 ELF Binaries                         |
+========================================================================================================+
                                                    |
                                                    v
+========================================================================================================+
|                                  ONEAPI LEVEL ZERO DRIVER (UMD)                                        |
|         ze_intel_vpu.dll • Command Queues • Graph Submission Engine • Memory Descriptors               |
+========================================================================================================+
                                                    | (IOCTL / Ring Buffer)
                                                    v
+========================================================================================================+
|                                  KERNEL MODE DRIVER (KMD)                                              |
|            intel_vpu.sys (Windows Driver 32.0.100.4723) / intel_vpu.ko (Linux Driver)                  |
|               Power Island DVFS Management • Doorbell Registers • Interrupt Handling                   |
+========================================================================================================+
                                                    | (PCIe 00:0b.0 MMIO)
                                                    v
+========================================================================================================+
|                                 PHYSICAL INTEL AI BOOST NPU 4000                                       |
+========================================================================================================+
```

### 2.1.1 Kernel-Mode Driver (KMD)
On Windows 11, the NPU hardware is exposed as a root PCI device:
- **Hardware ID**: `PCI\VEN_8086&DEV_643E` (Intel NPU Accelerator).
- **Bus Address**: `Domain 0, Bus 0, Device 0x0b, Function 0`.
- **Driver Module**: `intel_vpu.sys` (Driver Version: `32.0.100.4723`).
The KMD handles low-level register mapping, Direct Memory Access (DMA) page tables, hardware interrupt handling, clock gating, and power island transitions (C0 active to C6 sleep).

### 2.1.2 User-Mode Driver (UMD) & Level Zero Extension
Intel exposes the NPU via the **oneAPI Level Zero** compute interface. Unlike GPUs which primarily utilize OpenCL or Level Zero Core compute commands, the NPU uses the **Level Zero Graph Extension** (`ze_graph_ext`):
- `zeGraphCreate2`: Accepts an offline-compiled `.blob` binary and maps it into NPU unified memory.
- `zeGraphGetArgumentProperties`: Inspects static input and output tensor bindings, strides, and memory alignments.
- `zeGraphSetArgumentValue`: Binds host/device memory pointers directly to graph input/output descriptors without staging buffers.
- `zeCommandListAppendGraphExecute`: Submits execution packets to the NPU hardware doorbell register.

---

## 2.2 Inside the VPUX Compiler Pipeline: The MLIR Dialects

At the core of Intel's OpenVINO NPU compilation is the **VPUX Compiler** (`vpux-compiler`). Built upon the LLVM **Multi-Level Intermediate Representation (MLIR)** infrastructure, the compiler progressively lowers high-level computational graphs through four specialized dialects:

```
                    High-Level OpenVINO / ONNX Graph
                                  |
                                  v
+-------------------------------------------------------------------+
| 1. Inference Engine (IE) Dialect                                   |
|    - Canonicalization, Constant Folding, Dead Code Elimination    |
|    - Layer Fusion (Conv + Add + ReLU -> FusedConv)                |
|    - Shape Inference & Upper Bound Verification                   |
+-------------------------------------------------------------------+
                                  |
                                  v
+-------------------------------------------------------------------+
| 2. VPU Dialect                                                    |
|    - Target hardware-independent VPU abstractions                 |
|    - Strategy Assignment: Tiling across 6 NCE tiles               |
|    - Split-Over-H (SOH), Split-Over-K (SOK), Split-Over-W (SOW)   |
|    - Precision Quantization & Scaling Adjustments                 |
+-------------------------------------------------------------------+
                                  |
                                  v
+-------------------------------------------------------------------+
| 3. VPUIP Dialect (Instruction-Level Physical)                     |
|    - Physical memory allocation (12MB SRAM vs LPDDR5X DRAM)       |
|    - DMA Channel Scheduling & Ping-Pong Double Buffering          |
|    - SHAVE DSP vector kernel dispatch bindings                    |
+-------------------------------------------------------------------+
                                  |
                                  v
+-------------------------------------------------------------------+
| 4. VPURT Dialect (VPU Runtime Execution)                          |
|    - Hardware Barrier Resolution & Dependency Graphs              |
|    - Register Task Descriptors                                    |
|    - Final Binary Code Generation (.blob serialization)           |
+-------------------------------------------------------------------+
```

### 2.2.1 Dialect 1: The `IE` Dialect
The `IE` dialect represents target-agnostic neural operations (e.g., `IE.Convolution`, `IE.MatMul`, `IE.SoftMax`). During this phase:
- **Upper Bound Analysis**: Every tensor must have known static upper bounds. If an operation has unbound dynamic shapes `[-1, -1]`, the compilation immediately aborts with:
  ```text
  [ERROR] Upper bounds are not specified for node '__module.embeddings/aten::add/Add'
  Got negative shape dim bound: '-1'
  ```
- **Operator Lowering**: High-level complex operators (e.g., Multi-Head Attention, RMSNorm) are decomposed into primitive vector and matrix building blocks.

### 2.2.2 Dialect 2: The `VPU` Dialect & Multi-Tile Partitioning
The `VPU` dialect assigns execution strategies across the **6 NCE tiles**. The compiler models the cost function of three primary tiling strategies:
1. **Split-Over-H (SOH)**: The spatial height dimension of the activation tensor is sliced into 6 horizontal bands, distributed across the 6 NCE tiles.
2. **Split-Over-K (SOK)**: The output channel dimension (filters) is sliced across tiles. Each tile computes a subset of output features for the complete spatial image.
3. **Split-Over-Batch (SOB)**: For batch sizes $>1$, batches are distributed across tiles.

### 2.2.3 Dialect 3: The `VPUIP` Dialect & Scratchpad SRAM Scheduling
This is the most critical phase for maximizing hardware performance. The `StaticMemoryScheduler` analyzes tensor lifespans and maps tensors into memory spaces:
- `@CMX_NN`: The **12 MB On-Die Scratchpad SRAM**. Tensors here achieve $>1.2\text{ TB/s}$ bandwidth.
- `@DDR`: The **On-Package LPDDR5X-8533 Unified Memory**. Tensors here achieve $136\text{ GB/s}$.

The compiler generates **DMA Task Descriptors** that pre-fetch future layer weights from `@DDR` into `@CMX_NN` while the DPU is concurrently executing the current layer from a parallel SRAM buffer (Ping-Pong execution).

### 2.2.4 Dialect 4: The `VPURT` Dialect & Hardware Barriers
NPU 4 does not use software threads or OS mutexes for synchronization. Instead, it utilizes **hardware barrier registers**:
- There are physical barrier registers shared across the 6 NCE tiles.
- A DPU task signals a barrier upon completing a matrix multiplication.
- The dependent SHAVE DSP task waits on that specific barrier before executing the non-linear activation (e.g., SiLU or Softmax).
- Barrier overhead is essentially zero (<5 clock cycles).

---

## 2.3 The Static Shape Imperative & Bypassing It

### 2.3.1 Why NPU 4 Mandates Static Shapes
Traditional CPUs and GPUs allocate memory dynamically using heap managers (`malloc`, `cudaMalloc`). On NPU 4:
- The 12 MB on-die SRAM has **no virtual memory translation** and **no dynamic memory paging**.
- The hardware DMA engine requires exact byte offsets, strides, and memory layouts pre-computed at compile time.
- If an input shape varies dynamically from token to token, the compiler cannot guarantee that intermediate activations will fit within the 12 MB SRAM, which would cause an unrecoverable hardware fault.

### 2.3.2 Architectural Strategies for Dynamic Workloads
To execute dynamic-length sequence workloads (such as autoregressive chat models or variable-length audio) on NPU 4, three advanced patterns must be used:

#### Strategy A: Static Bucket Ensembles
Pre-compile the model for discrete sequence buckets:
$$\mathcal{S} = \{32, 64, 128, 256, 512, 1024\}$$
At runtime, the host measures the sequence length $L$, selects the smallest bucket $B \ge L$, pads the tensor with zeros, and dispatches to the corresponding pre-cached static blob.

#### Strategy B: Chunked Streaming (Prefix-Tiled Execution)
Instead of passing variable-length audio or text, decompose the stream into static chunks:
- Whisper audio: Sliced into uniform **30.00-second Mel spectrogram chunks** (`[1, 80, 3000]`).
- Dense text embeddings: Sliced into uniform **64-token chunks** (`[1, 64]`).
- Screen frames: Normalized to static **640x640** or **1280x720** grids.

#### Strategy C: The Recurrent State-Space Approach (SSM)
As proven in Chapter 4, by replacing attention-based Transformers with State-Space Models (Mamba, RWKV), the recurrent hidden state has an **inherently constant dimension**:
$$h_t \in \mathbb{R}^{B \times D \times N}$$
Because $h_t$ never changes shape regardless of sequence length, **a single static NPU compilation serves all sequences from length 1 to length 1,000,000!**

---

## 2.4 Undocumented Hardware Properties & Tuning Knobs

Inspection of the driver properties reveals undocumented compiler parameters that unlock peak performance:

```python
import openvino as ov

core = ov.Core()

# 1. Hardware Clock Frequency Boost (NPU_TURBO)
# Forces the NCE tiles to operate at maximum boost frequency (1.95 GHz)
# bypassing aggressive power-saving down-clocking.
core.set_property("NPU", {"NPU_TURBO": True})

# 2. Hardware Quantize-Dequantize Optimization (NPU_QDQ_OPTIMIZATION)
# Fuses Quantize and Dequantize operators directly into DPU input/output pipelines,
# eliminating intermediate SHAVE DSP conversion passes.
core.set_property("NPU", {"NPU_QDQ_OPTIMIZATION": True})

# 3. Persistent Driver Disk Caching
# Compiles model once into pre-compiled binary blob (*.blob) and reloads in <5ms.
core.set_property("NPU", {"CACHE_DIR": "C:\\Users\\Janmejai\\.tools\\npu\\cache"})

# 4. Asynchronous Request Queue Depth
# Enables pipeline parallelism across 6 NCE tiles
core.set_property("NPU", {
    ov.properties.hint.performance_mode(): ov.properties.hint.PerformanceMode.LATENCY,
    ov.properties.hint.num_requests(): 1
})
```

---

## 2.5 The Compiled Binary Blob (`.blob`) Architecture

When OpenVINO saves a compiled NPU model to disk, it produces a binary file (`.blob`). Reverse-engineering the binary blob header exposes its physical execution plan:

```
+--------------------------------------------------------------------+
|                         NPU .BLOB BINARY FORMAT                    |
|                                                                    |
|  [0x00 - 0x1F] Header Magic ('VPUX') • Version • Target Arch (4000)|
|  [0x20 - 0x7F] Metadata Table: Input/Output Tensor Descriptors     |
|  [0x80 - 0xFF] Section Offsets & Alignment Maps                    |
|  ----------------------------------------------------------------  |
|  [Section 1]   Hardware Barrier Descriptors Table                  |
|  [Section 2]   DPU Task Descriptors (Systolic Matrix Microcode)    |
|  [Section 3]   SHAVE DSP Executable Bytecode (Vector SIMD Binaries)|
|  [Section 4]   DMA Transfer Queue Descriptors                      |
|  [Section 5]   Quantized Weight Payload (INT8 / FP16 Constants)    |
+--------------------------------------------------------------------+
```
Because the `.blob` file already contains pre-scheduled memory offsets, hardware barrier tables, and native SHAVE machine instructions, loading a `.blob` requires **zero recompilation, zero graph validation, and zero driver translation**. It is memory-mapped directly into the NPU's unified address space in **under 5 milliseconds**, enabling instantaneous cold starts.
