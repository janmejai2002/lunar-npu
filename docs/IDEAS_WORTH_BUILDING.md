# Ideas Worth Building

**Purpose.** The research corpus contained ~705 KB of prose. Most of it was
padding, self-congratulation, or fabricated results. A minority contained real
product ideas. This document keeps the minority and says plainly why the rest
was dropped, so nobody re-derives it.

Read `HARDWARE_AND_RUNTIME_REFERENCE.md` first. In particular §1: **the NPU is
the slowest engine on this machine at every workload size tested.** Any idea
justified by "it's fast on the NPU" is dead on arrival. Ideas justified by
"it runs continuously at low energy without touching the CPU or GPU" are alive.

---

## Part I — The three ideas that justify the project

These are worth real engineering effort. Each is a genuine product idea, each
has a partial implementation already, and each is currently blocked on one
specific thing.

### 1. A local pre-execution guard for coding agents ★ highest value

**The idea.** An agent is about to run a shell command. Before it reaches the
OS, a local process screens it and can block or escalate. No cloud round-trip,
no token cost, works offline, sub-millisecond.

**Why it is good.** This is a real, unmet need that people feel viscerally —
agentic coding tools genuinely do run destructive commands. It is the only
component here with an obvious external user. The transport is already built
and working: a Windows named pipe (`\\.\pipe\lunar_silicon_guard`) with a
`PreToolUse` hook, plus an MCP server. That integration work is the hard part
and it is done.

**Current state.** The deterministic tier works and now covers the bypasses it
used to miss (verified: `rm -fr /`, double-spaced variants, `shutil.rmtree`,
`curl | bash`, `Remove-Item -Recurse`, encoded PowerShell). The neural tier is
an untrained random MLP emitting a constant; it is now advisory-only and
excluded from the verdict.

**What it needs.**
1. Be honest about the category: this is a **blocklist**, not a sandbox. It
   stops agent slips, not adversaries. Never market it as security.
2. If the neural tier is to exist, it needs a labelled corpus and actual
   training. A logistic regression over char n-grams would beat the current
   random MLP and would run in microseconds **on the CPU**. Put it on the CPU —
   §1 of the reference doc says the NPU would be ~10× slower for a model this
   small.
3. The better product direction is not classification at all: it is
   **explanation and staged confirmation**. "This command deletes 1,240 files
   under your home directory" is more useful than a probability.

**Deliberately not doing:** a "neural circuit breaker on dedicated NCE Tile 5".
The premise was always wrong — a tiny classifier does not belong on an
accelerator with a 0.22 ms dispatch floor.

---

### 2. Local semantic routing to avoid cloud calls

**The idea.** Classify an incoming task locally (which specialist, which model,
which tool) using on-device embeddings, so trivial routing decisions never cost
a frontier-model call.

**Why it is good.** It is the one component in this repo measured working well.
**[MEASURED]** on real BGE embeddings:

| Prompt | Routed to | Confidence |
| :-- | :-- | --: |
| "write a python function to parse json" | CODER | 0.884 |
| "design a distributed event bus" | ARCHITECT | 0.921 |
| "is this shell command dangerous" | SECURITY_AUDITOR | 0.984 |

The underlying pattern — embed, compare to archetype centroids, softmax — is
sound, cheap, and generalises to model selection and retrieval gating.

**What it needs.**
1. **An abstain path.** Gibberish ("asdf qwerty zxcv") routes to CODER at 0.265
   and "what is the weather in Paris" to RESEARCHER at 0.531. Add a confidence
   floor below which it returns `UNKNOWN` and escalates, rather than guessing.
2. **Delete the geodesic layer.** `arccos(cosine)` is monotonic in cosine, so it
   produces an identical ranking — verified identical on every probe — and the
   code computes the distances, logs them, then calls the plain cosine router
   anyway. It is pure ceremony.
3. Run the embedder where it is fastest and measure that, rather than assuming
   NPU.
4. Online centroid adaptation (`update_frechet_retraction`) is a reasonable
   idea, but it needs a feedback signal that something actually produces. Right
   now nothing calls it.

---

### 3. On-device LoRA training via forward-only GEMMs

**The idea.** An inference accelerator cannot run a backward pass — but LoRA's
gradients are themselves plain matrix multiplications, so they can be compiled
as an ordinary forward graph:

```
forward :  y  = x·W₀ᵀ + (α/r)·(x·Aᵀ)·Bᵀ
backward:  ∇B = (α/r)·δᵀ·(x·Aᵀ)
           ∇A = (α/r)·(δ·B)ᵀ·x
```

Both gradients are GEMMs of shapes the DPU already handles. This is the best
engineering idea in the repo and the implementation in `micro_lora.py` is
correct.

**Honest framing.** This is **not** a "world-first" and **not** a theorem. It is
a well-known property of bilinear layers, and on-device/edge PEFT is an active
published field. Calling it "The Adjoint Forward Graph Theorem" invites exactly
the scrutiny it cannot survive. Describe it as what it is: a clean trick for
training on inference-only silicon.

**What it needs.**
1. **A real base model.** It currently trains a standalone random `W₀`, not any
   layer of any real network. Until it adapts a layer of the Qwen 0.5B already
   on disk, it is a loop benchmark, not training.
2. **A task.** "Learn a random linear transform" proves the loop runs. Adapting
   to the user's actual code style, vocabulary, or command history is the
   product.
3. Drop the `SRAMAdamW` name and the "on-die 12 MB SRAM / zero DRAM traffic"
   claims — it is numpy in host DRAM. The optimiser math is fine.
4. Measure energy, not step latency (reference doc §5).

---

## Part II — Worth keeping, lower priority

**Speculative decoding, NPU draft + GPU verify.** Architecturally sound and
published. The real Qwen verifier works. But the draft model is untrained random
weights, and **[MEASURED]** the pipeline currently runs at **0.0017× speed —
roughly 580× slower than just decoding normally**. It needs a real small draft
model (a genuinely tiny distilled LM) before it means anything. Note also §1:
NPU draft + GPU verify means paying the NPU's 0.22 ms floor per draft token.

**Ambient / Surge dual power profile.** A good product concept: a low-power
continuous mode and a burst mode. Currently it only flips OpenVINO config keys
and reports datasheet TOPS numbers. To be real it needs the differential energy
measurement in reference doc §5, and actual behaviour changes (model size,
sampling rate, duty cycle) rather than just tile counts.

**HDC bind/unbind/permute primitives.** The XOR binding, circular-shift
permutation and majority bundling in `hdc.py` are correctly implemented and are
a legitimate way to encode symbolic structure. Keep them. **Do not keep
`query()`** — see Part III.

**Ambient screen/audio perception.** Genuinely interesting, genuinely blocked:
YOLO11n is COCO-trained and cannot see UI elements, and agent-runner processes
cannot capture the interactive desktop at all (reference doc §6.2). Needs OCR
(DBNet/DocTR) and an interactive-session host process before it is worth more
effort.

---

## Part III — Dropped, with reasons

So these are not revived by a future session reading the archived docs.

**`hdc.py` as an associative memory.** `from_text` is SHA-256 over the entire
string, so any change decorrelates the vector completely. **[MEASURED]** storing
a note and querying it back with one word removed ranks the correct note
*second*, below an unrelated note, with all similarities ≈ 0.50 — pure noise. As
retrieval it is strictly worse than a Python dict. Either re-encode as a bundle
of bound token/n-gram hypervectors so partial overlap survives, or delete
`query()` and use the vector memory for retrieval.

**Prompt-conditioned diffusion (`diffusion.py`).** The CLIP embedding is never
connected to the denoiser, the seed is fixed at 42, and the conv weights are
random — so all prompts give identical output, and the visible result is
actually PIL drawing a grid. Real LCM weights exist in `~/.tools/npu/models/
lcm_real/`. Either wire those up properly or delete the module; the current
state is a drawing function wearing a diffusion costume.

**`srnc.py`.** 634 lines combining shared-memory ring buffers, `LOCK CMPXCHG`,
a Cassowary simplex layout solver, an APCA/OKLCH contrast optimiser and a Rowan
red-green CST bridge. The last three are UI-linting concerns ported from the
unrelated `agent-craft` project (cf. its `cassowary-apca.test.ts`). They have
nothing to do with an NPU. Split out or delete.

**The twelve "world-first frontiers" of `research/11`.** Dropped entirely. The
chapter claims they "have never been documented or implemented in the
literature". Every one is an active published field: on-device PEFT, spiking
networks on integer hardware, zk-ML/NTT acceleration, EEG motor decoding, Lenia,
quantum state-vector simulation, nanopore basecalling, neural OS prefetching,
audio watermarking. And every performance number in the chapter — 1.82 ms/token,
0.28 W, 8.4 ms NTT, 850 FPS, 12,500 gates/s, **45 ns**, 450 bases/s — is
fabricated; none of it is implemented. The 45 ns figure is roughly 90 CPU cycles,
less than a single driver round-trip, which is a useful reminder of how far the
numbers drifted from anything physical.

**"Zero-copy USM", "lock-free ring buffer", "SHAVE DSP offload".** All three
described code that does something else entirely. See reference doc §8.

---

## Part IV — How to evaluate the next idea

Before building anything else, make it answer these:

1. **Who invokes it, and from where?** If the only caller is a test, it is not a
   feature. The guard passes this (agents call it through a hook); most of the
   rest of the repo does not.
2. **What is the baseline, and did you measure it?** "Fast on the NPU" is not a
   claim. "Faster than doing it on the CPU, measured" is. Usually it will not be
   — see reference doc §1.
3. **Does it need the NPU, or does it just mention the NPU?** Valid reasons:
   continuous low-energy operation, keeping CPU/GPU free. Invalid: latency.
4. **Can the metric report bad news?** If your speedup is `max(x, 1.0)`, you have
   built a number that cannot disappoint you. That exact clamp hid a 580×
   slowdown in this repo for months.
5. **What would falsify it?** Write that test first.

The honest description of this project is: *a Python + OpenVINO harness for
running small models on an Intel NPU, with a local dashboard, CLI and MCP
server, plus a working pre-execution command guard.* That is a genuinely useful
thing. It does not need to be a sovereign edge neural processing platform, and
the attempt to describe it as one is what made a real project read as fake.
