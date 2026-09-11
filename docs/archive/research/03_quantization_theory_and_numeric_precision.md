# Chapter 3: Quantization Theory, Numerical Precision & Low-Bit Calibration for Intel NPU 4

---

## 3.1 Mathematical Foundations of Hardware Quantization

Quantization is the formal process of mapping high-precision continuous representations (such as 32-bit floating-point $\mathbb{R}$) onto discrete, finite-range low-bit integer lattices (such as 8-bit $\mathbb{Z}_{256}$ or 4-bit $\mathbb{Z}_{16}$). On dedicated hardware like the Intel AI Boost NPU 4000, quantization is not merely a memory compression technique—it is the direct prerequisite for activating the high-throughput systolic array DPU engines.

### 3.1.1 Affine (Asymmetric) Quantization
For an arbitrary continuous tensor $X \in \mathbb{R}$ bounded within range $[\alpha, \beta]$, affine quantization maps real values $x \in X$ to integers $q \in [q_{\min}, q_{\max}]$:

$$q = \text{clip}\left( \left\lfloor \frac{x}{S} \right\rceil + Z, \ q_{\min}, \ q_{\max} \right)$$

Where:
- $\lfloor \cdot \rceil$ denotes the nearest rounding operator (typically Round-to-Nearest-Even).
- $S \in \mathbb{R}^+$ is the positive scalar **scale factor**:
  $$S = \frac{\beta - \alpha}{q_{\max} - q_{\min}}$$
- $Z \in \mathbb{Z}$ is the **zero-point**, representing the integer value to which real value $0.0$ maps:
  $$Z = \text{round}\left( \frac{-\alpha}{S} \right) + q_{\min}$$

The dequantization function that reconstructs the approximate real value $\hat{x}$ is:
$$\hat{x} = S \cdot (q - Z)$$

### 3.1.2 Symmetric Quantization
When the distribution of values is centered around zero, the dynamic range is chosen symmetrically: $[-\gamma, \gamma]$, where $\gamma = \max(|\alpha|, |\beta|)$. Under symmetric quantization:
$$Z = 0, \quad S = \frac{\gamma}{q_{\max}}$$

For signed 8-bit integers where $q \in [-128, 127]$, $q_{\max} = 127$, simplifying the mapping to:
$$q = \text{clip}\left( \left\lfloor \frac{x}{S} \right\rceil, -128, 127 \right), \quad \hat{x} = S \cdot q$$

---

## 3.2 Hardware Implications on Intel NPU 4 DPU Systolic Arrays

The choice between symmetric and asymmetric quantization profoundly impacts physical silicon execution on Lunar Lake's DPU matrix engines:

### 3.2.1 The Mathematical Cost of Asymmetric Zero-Points
Consider matrix multiplication between activation matrix $X \in \mathbb{R}^{M \times K}$ and weight matrix $W \in \mathbb{R}^{K \times N}$:
$$Y = X \cdot W$$

Under asymmetric quantization for both activations and weights:
$$\hat{X} = S_x (Q_x - Z_x), \quad \hat{W} = S_w (Q_w - Z_w)$$

The product becomes:
$$Y = S_x S_w \left[ \sum_{k=1}^K (Q_{x, ik} - Z_x)(Q_{w, kj} - Z_w) \right]$$

Expanding the inner algebraic summation:
$$\sum_{k=1}^K (Q_{x, ik} - Z_x)(Q_{w, kj} - Z_w) = \underbrace{\sum_{k=1}^K Q_{x, ik} Q_{w, kj}}_{\text{Core DPU MAC Engine}} - \underbrace{Z_w \sum_{k=1}^K Q_{x, ik}}_{\text{Dynamic Row Sum}} - \underbrace{Z_x \sum_{k=1}^K Q_{w, kj}}_{\text{Offline Column Sum}} + \underbrace{K \cdot Z_x Z_w}_{\text{Constant}}$$

#### Hardware Consequences on NPU 4:
1. **Core Systolic Array**: The term $\sum Q_x Q_w$ maps directly into the 2D MAC grid with 100% computational efficiency.
2. **Offline Column Sum**: $Z_x \sum Q_w$ and $K Z_x Z_w$ are static and can be folded into the bias vector during offline compilation by `vpux-compiler`.
3. **Dynamic Row Sum**: The term $Z_w \sum Q_x$ is dynamically dependent on runtime activations. If $Z_w \ne 0$, the NPU must allocate extra SHAVE DSP vector cycles to compute row reductions across activations, creating a pipeline synchronization stall between the DPU and SHAVE units.
4. **Architectural Rule for NPU 4**: **Weights must ALWAYS be quantized symmetrically ($Z_w = 0$)**. This eliminates the dynamic row-sum term entirely:
   $$Y = S_x S_w \left[ \sum_{k=1}^K Q_{x, ik} Q_{w, kj} - Z_x \sum_{k=1}^K Q_{w, kj} \right]$$

---

## 3.3 The Activation Outlier Challenge in LLMs & SmoothQuant Adaptation

When deploying Transformer architectures or Large Language Models on NPU 4, standard Post-Training Quantization (PTQ) to INT8 often causes catastrophic perplexity degradation. This failure is driven by **activation outliers**.

```
Activation Distribution (Hidden States)      Weight Distribution
       |                                              |
       |  Outlier Peak (x > 120.0)                    |
       |  |                                           |
       |  |                                        ---+--- (Gaussian Bounded)
   ----+--+----+---- (Standard Tokens [-2, 2])        |
       |                                              |
```

In modern transformer models (e.g., Qwen, LLaMA, Mistral), activations across certain specific channels exhibit values up to $100\times$ larger than the mean token distribution. If uniform per-tensor quantization is applied:
- The scale factor $S = \frac{120.0}{127} \approx 0.94$.
- 99.9% of normal tokens (with values between $-1.0$ and $+1.0$) are quantized to integer bins $0, +1$, or $-1$, destroying all semantic representational capacity.

### 3.3.1 Mathematical Formulation of SmoothQuant for NPU 4
SmoothQuant resolves this by exploiting an architectural invariant: **weights are easy to quantize, while activations are difficult**. SmoothQuant applies a per-channel scaling transformation to migrate quantization difficulty from activations into weights:

$$Y = X W = (X \cdot \text{diag}(s)^{-1}) \cdot (\text{diag}(s) \cdot W) = \hat{X} \hat{W}$$

Where $s \in \mathbb{R}^K$ is a per-channel smoothing factor:
$$s_j = \frac{\max(|X_j|)^\alpha}{\max(|W_j|)^{1-\alpha}}$$

- $X_j$ represents the activation values across channel $j$.
- $W_j$ represents the weights corresponding to channel $j$.
- $\alpha \in [0, 1]$ is the migration hyperparameter. For Intel NPU 4, empirical calibration establishes **$\alpha = 0.5$** as the Pareto-optimal balance point.

Once smoothed:
- The activation dynamic range is compressed by up to $10\times$.
- Activations can be quantized symmetrically into INT8 with zero loss in precision.
- The diagonal matrix $\text{diag}(s)^{-1}$ is folded directly into the preceding LayerNorm or RMSNorm weights, incurring **0 additional runtime FLOPs**.

---

## 3.4 Sub-Byte Quantization: INT4, W4A16 & W8A8 Precision Schemes

Intel NPU 4 supports flexible precision schemes across its DPU and SHAVE processing engines:

| Precision Scheme | Weight Precision | Activation Precision | Memory Footprint | Accuracy Impact | NPU Hardware Execution |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **FP16** | 16-bit Float | 16-bit Float | 100% (Baseline) | 0.0% Loss | Native SHAVE + DPU FP16 Mode |
| **W8A8 (INT8)** | 8-bit Signed Int | 8-bit Signed Int | 50% Reduction | <0.2% Degradation | Maximum DPU Throughput (46.7 TOPS) |
| **W4A16** | 4-bit Packed Int | 16-bit Float | 75% Reduction | <0.5% Degradation | DMA Hardware Unpack -> FP16 MAC |
| **W4A8** | 4-bit Packed Int | 8-bit Signed Int | 75% Reduction | ~1.0% Degradation | DPU Sub-Byte Integer Acceleration |
| **BitNet (1.58b)**| $\{-1, 0, +1\}$ | 8-bit Signed Int | 80% Reduction | Minimal (if trained)| Addition-Only Integer Accumulation |

### 3.4.1 Weight-Only 4-bit Packing (W4A16)
In memory-bandwidth-bound autoregressive decoding (such as single-token SLM generation), performance is constrained by weight streaming speed from LPDDR5X memory rather than arithmetic compute.
- In **W4A16**, weights are stored as packed 4-bit nibbles (two weights per byte).
- During inference, the hardware DMA engine streams weights across the on-package bus at **136 GB/s**, effectively achieving the equivalent bandwidth of **272 GB/s** uncompressed FP16.
- Weights are unpacked on-the-fly inside the NCE tile scratchpad memory before dispatch to the compute units.

---

## 3.5 NNCF Post-Training Quantization Pipeline for NPU 4

The recommended toolchain for calibrating neural networks for Intel Lunar Lake NPU is Intel's **Neural Network Compression Framework (NNCF)**:

```python
import nncf
import openvino as ov

# Step 1: Initialize OpenVINO Model
core = ov.Core()
model = core.read_model("unquantized_model.xml")

# Step 2: Define Calibration Dataset (100-300 representative domain samples)
def transform_fn(data_item):
    return {"input_ids": data_item["input_ids"], "attention_mask": data_item["attention_mask"]}

calibration_dataset = nncf.Dataset(sample_data, transform_fn)

# Step 3: Configure Advanced NPU Quantization Parameters
quantized_model = nncf.quantize(
    model,
    calibration_dataset,
    model_type=nncf.ModelType.TRANSFORMER,
    preset=nncf.QuantizationPreset.PERFORMANCE,  # Enforces Symmetric INT8 for Weights & Activations
    fast_bias_correction=True,
    subset_size=256,
    smooth_quant_alpha=0.5,                       # Activates SmoothQuant for NPU activation outliers
    target_device=nncf.TargetDevice.NPU           # Enforces NPU 4 hardware constraints and static layouts
)

# Step 4: Save NPU-Optimized IR
ov.save_model(quantized_model, "quantized_model_npu.xml")
```

By applying SmoothQuant ($\alpha = 0.5$) combined with Symmetric Per-Channel weight quantization, models achieve **3.8x to 5.2x faster inference** on Intel AI Boost NPU 4000 compared to unquantized FP16 baselines while preserving $>99.5\%$ task accuracy.
