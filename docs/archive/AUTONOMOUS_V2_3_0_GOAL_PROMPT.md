# Autonomous Execution Goal Prompt: Lunar NPU v2.3.0 "Make Everything Real"

> **Instructions for the User**:  
> To launch the next phase of autonomous development in a new chat session, copy the entire prompt inside the code block below and send it to the agent.  
> It is pre-configured with all operational constraints, session continuity rules, gap analysis mandates, and reality contracts.

---

```markdown
/goal Lunar NPU v2.3.0: Grounding Theoretical Research into Physical Silicon & Real Operational Code ("Make Everything Real")

### 1. MANDATORY CONTEXT RESTORATION & FIRST ACTION
You are continuing development on **Lunar NPU** (`janmejai2002/lunar-npu`) on Windows 11 Home 64-bit / Intel Core Ultra 7 258V (Intel AI Boost NPU 4000 @ 47 TOPS INT8 + Intel Arc 140V Xe2 GPU).

Before writing code or proposing changes, you MUST immediately:
1. Inspect `CONTEXT_HANDOFF.md` to load the current system state, architecture, and passing test baseline (**128/128 pass**).
2. Inspect `docs/RESEARCH_VS_REALITY_GAP_ANALYSIS.md` to understand the audit findings: 45.7% Real, 25.7% Isolated/Synthetic, 28.6% Unimplemented.
3. Inspect `docs/JOBS_TO_BE_DONE_V2_3_0.md` to load the prioritized engineering jobs and acceptance criteria.
4. In your very first reply, summarize the project's exact state in 2-3 concise bullets and state the immediate P0 job you are tackling. Do NOT ask generic "How can I help you?" questions.

---

### 2. CORE MISSION & ANTI-HALLUCINATION CONTRACT
In v2.2.0, Lunar NPU established its real Operational Transmission Layer (`indexer.py`, named pipe `\\.\pipe\lunar_silicon_guard` with HMAC receipts, DirectComposition Ghost HUD with `WDA_EXCLUDEFROMCAPTURE = 0x11`).

Your mission in v2.3.0 is to **eliminate all synthetic simulations and make the core algorithms real, physical, and integrated into active consumer workflows.**

You are permanently bound by `<RULE[anti_hallucination_and_reality_anchor]>`:
1. **NO SPEC-ONLY EXPANSIONS**: Never write 50-page architecture treatises or speculative math proofs. Build working, verifiable code.
2. **ZERO SYNTHETIC DATA**: Never use `np.random.randn()` or mock uniform arrays in production code to simulate product behavior or declare victory. Run operations on real project files, real code tokens, real OS audio streams, and real desktop frames.
3. **THE TRANSMISSION TEST**: Writing an isolated algorithm is only 20% of the work. An algorithm is NOT complete until it has real inputs, an active consumer (agent/IDE/shell), and observable utility (tokens saved, calls blocked, latency cut). If it only talks to its own test file, label it `[ISOLATED EXPERIMENT - NOT INTEGRATED]`.
4. **TELEMETRY IS NOT A PRODUCT**: Dials and synthetic benchmark loops are not completed milestones. Products do work for the user.
5. **BRUTAL TRUTH REPORTING**: In every status update, explicitly report items under:
   - `[REAL & WORKING IN PRODUCTION]`
   - `[ISOLATED TEST PASS - Synthetic math only]`
   - `[UNIMPLEMENTED / PLACEHOLDER]`

---

### 3. TECHNICAL SPECIFICATIONS & ENVIRONMENT CONSTRAINTS
- **OS & Shell**: Windows 11 Home 64-bit, Windows PowerShell exclusively. Never use raw bashisms or `cd` navigation.
- **Python**: 3.13.4, `venv/` (installed in editable mode via `pyproject.toml`).
- **Accelerator Target**: Intel AI Boost NPU 4000 (OpenVINO NPU plugin, FP16/INT8 compiled graphs) + Intel Arc 140V Xe2 GPU (Level Zero / OpenVINO GPU plugin).
- **Test Baseline**: **128/128 tests currently pass**. You must never regress existing tests. Run tests via `rtk pytest tests/` or `python -m pytest tests/`.
- **Active Endpoints**:
  - Lunar Studio API & HUD: `http://127.0.0.1:8899`
  - Silicon Guard Named Pipe: `\\.\pipe\lunar_silicon_guard`
  - CLI: `lunar` (in PATH)

---

### 4. PHASED SPRINT EXECUTION ROADMAP

Execute the following Sprints in strict sequence. Do not jump ahead until each sprint's acceptance criteria and automated tests are satisfied.

#### SPRINT 1 (P0): Ground Mamba-2 SSM in Real Pretrained Weights & Fast BPE Tokenizer
- **Target File**: `lunar_core/mamba_ssm.py`
- **Actions**:
  1. Replace the `np.random.uniform` weight generator with a real weight loader supporting Hugging Face safetensors or OpenVINO IR blobs (e.g. `state-spaces/mamba-130m` or `tiiuae/falcon-mamba-7b` INT8).
  2. Integrate Hugging Face `tokenizers` (Fast BPE) to ingest real strings and output physical token IDs.
  3. Load actual $A, B, C, \Delta$ discretization matrices from pretrained weights.
  4. Ensure $O(1)$ recurrent state persistence across multi-turn sessions.
- **Verification**: `lunar mamba "def fibonacci(n):"` produces coherent code tokens; write and pass `tests/test_mamba_real_weights.py`.

#### SPRINT 2 (P0): Compile Micro-LoRA Adjoint Backward Graph into OpenVINO NPU IR
- **Target File**: `lunar_core/micro_lora.py`
- **Actions**:
  1. Express the Adjoint Backward Pass ($\nabla \mathbf{B}, \nabla \mathbf{A}$) as OpenVINO `ov.Model` operations instead of CPU `np.matmul`.
  2. Compile both forward and backward graphs to `NPU` (or dual-target `NPU` forward + `GPU` backward).
  3. Implement fused `SRAMAdamW` parameter updates without host DRAM re-allocation.
  4. Train on real user corrections from session logs.
- **Verification**: `lunar lora --steps 50` runs with <5% host CPU utilization; write and pass `tests/test_lora_npu_backward.py`.

#### SPRINT 3 (P1): Native DirectX DXGI Desktop Duplication Capture Engine
- **Target Files**: `lunar_core/dxgi_capture.py` (NEW), `lunar_core/vision.py`
- **Actions**:
  1. Implement `IDXGIOutputDuplication` capture via Windows `d3d11.dll` and `dxgi.dll` using `ctypes`.
  2. Map Direct3D staging textures directly into NumPy/OpenVINO memory buffers in <3ms.
  3. Wire captured frames directly into the YOLO11n INT8 OpenVINO pipeline.
  4. Preserve `PIL.ImageGrab` only as fallback for headless/RDP environments.
- **Verification**: `python -m lunar_core.dxgi_capture --bench` demonstrates 60+ FPS capture; write and pass `tests/test_dxgi_capture.py`.

#### SPRINT 4 (P1): Heterogeneous Latent Consistency Model (LCM) 4-Step Diffusion Engine
- **Target File**: `lunar_core/diffusion.py` (NEW)
- **Actions**:
  1. Implement `LunarHeterogeneousDiffusion`:
     - Text Encoder (CLIP INT8) on `NPU`.
     - 4-step LCM Denoiser (INT8 OpenVINO IR) on `NPU`.
     - VAE Latent Decoder (FP16 OpenVINO IR) on `GPU` (Arc 140V Xe2).
  2. Expose CLI command `lunar sketch "<prompt>" --out <path>`.
  3. Expose MCP tool `lunar_diffuse`.
- **Verification**: `lunar sketch "code review icon" --out test.png` outputs a valid 512x512 image in <1.5s; write and pass `tests/test_diffusion.py`.

#### SPRINT 5 (P1): Native Windows WASAPI IAudioClient COM Loopback Hook
- **Target File**: `lunar_core/audio.py`
- **Actions**:
  1. Implement a native COM WASAPI loopback capture client (`AUDCLNT_STREAMFLAGS_LOOPBACK`) via `ctypes`.
  2. Capture live system audio, downsample in-memory to 16kHz mono float32, and feed directly to Whisper INT8 on the NPU.
  3. Add energy-based Voice Activity Detection (VAD) to prevent idle processing.
- **Verification**: Audio playing through system speakers is transcribed into real text; write and pass `tests/test_audio_loopback_real.py`.

#### SPRINT 6 (P2): 10,000-Bit Hyperdimensional Computing (HDC) Memory Engine
- **Target File**: `lunar_core/hdc.py` (NEW)
- **Actions**:
  1. Implement $D=10,000$ binary hypervector engine packed into 1,250 bytes (`np.packbits`).
  2. Implement XOR binding, circular shift permutation, and majority bundling.
  3. Expose fast associative memory search via `lunar_core/studio.py` and CLI `lunar hdc`.
- **Verification**: Recall latency <150µs; write and pass `tests/test_hdc.py`.

---

### 5. SELF-HEALING & VERIFICATION GATE
At the conclusion of each sprint:
1. Run `rtk pytest tests/` or `python -m pytest tests/`.
2. Confirm that the total test count increases (e.g. 128 -> 135 -> 142) and that **100% of tests pass cleanly**.
3. Verify that the Lunar Studio HUD at `http://127.0.0.1:8899/api/health` returns healthy status.
4. Update `CONTEXT_HANDOFF.md` and `docs/RESEARCH_VS_REALITY_GAP_ANALYSIS.md` with the updated Reality Scorecard and verified benchmark latencies.

When ALL 6 sprints are complete, verified on physical hardware, and all tests pass with zero synthetic mocks, output the termination marker:
`<!-- GOAL_COMPLETE -->`
```
