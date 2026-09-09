# Contributing to Lunar

Thank you for your interest in contributing to **Lunar**! Lunar is an open-source platform dedicated to unlocking physical hardware acceleration on Intel Lunar Lake (47 TOPS NPU) for ambient intelligence, speculative decoding, and edge neural processing.

---

## Code of Conduct

We are committed to providing a welcoming, inclusive, and harassment-free environment for everyone. Please read and follow our [Code of Conduct](CODE_OF_CONDUCT.md).

---

## Getting Started

### 1. Fork & Clone
```bash
git clone https://github.com/janmejai2002/lunar-npu.git
cd lunar-npu
```

### 2. Environment Setup
We recommend Python 3.10 to 3.13. Using a virtual environment:
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
```

### 3. Hardware Requirements
- **Primary Target**: Intel Core Ultra 200V series ("Lunar Lake") with Intel AI Boost NPU (Architecture 4000, driver >= 1004723).
- **Secondary Targets**: Intel Core Ultra 100 series ("Meteor Lake"), Arrow Lake, or CPU fallback.
- **Runtime**: OpenVINO Runtime 2025.0+.

---

## Development Workflow

### Running Tests
All tests are located in `tests/` and use `pytest`:
```powershell
python -m pytest -p no:httpbin -v
```

### Running Hardware Benchmarks
To benchmark local NPU latency and throughput:
```powershell
python benchmarks/run_benchmarks.py
```

### Code Style & Architecture Principles
1. **Zero Hallucination**: Code must be accompanied by deterministic tests.
2. **Minimal Dependencies**: Core runtime depends solely on `openvino` and `numpy`. Avoid adding large frameworks to core.
3. **Static Tensor Shapes**: When compiling computational graphs for the Intel NPU, specify static input and state shapes (`[1, dim]` or `[1, seq_len]`). Dynamic dimensions degrade or fail compilation on the NPU compiler plugin.
4. **Clean Types**: Use standard Python type annotations (`typing`).

---

## Conventional Commits

We follow the [Conventional Commits](https://www.conventionalcommits.org/) specification:
- `feat(component)`: A new feature or recipe implementation
- `fix(component)`: A bug fix (please also log in `docs/BUG_LEDGER.md`)
- `perf(component)`: Performance optimizations (e.g. latency, memory footprint)
- `docs(component)`: Documentation or research compendium updates
- `test(component)`: Adding or refactoring unit tests
- `refactor(component)`: Code changes that neither fix bugs nor add features

---

## Pull Request Checklist

Before submitting your PR, verify:
- [ ] All unit tests pass: `pytest -p no:httpbin tests/ -v`
- [ ] No temporary files or caches (`__pycache__`, `.pyc`, logs) are tracked
- [ ] New features include corresponding unit tests under `tests/`
- [ ] Documentation is updated if CLI options or SDK interfaces changed
- [ ] PR description describes the "Why" and provides benchmark numbers if touching performance

Thank you for advancing the edge neural processing frontier!
