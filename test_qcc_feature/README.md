# QCC Feature Tests

This directory contains integration and acceptance tests for compiler features developed
on top of the [Catalyst](https://github.com/PennyLaneAI/catalyst) quantum compiler stack.
Each subdirectory is a self-contained test suite for one feature.

## Directory Structure

```
test_qcc_feature/
├── README.md               ← this file
├── qasm3/                  ← OpenQASM 3 translation pipeline tests
└── do_qaoa/                ← Differential-Optimization QAOA tests
```

> **Adding a new feature?** Follow the [Contributor Guide](#contributor-guide) at the bottom
> of this file and add a row to the table below.

---

## Features

| Folder | Feature | Quick run | Needs build? |
|--------|---------|-----------|--------------|
| [qasm3/](qasm3/) | OpenQASM 3 translation pipeline | `bash test_qcc_feature/qasm3/run_all_tests.sh` | Yes (`quantum-translate`) |
| [do_qaoa/](do_qaoa/) | Differential-Optimization QAOA | `python test_qcc_feature/do_qaoa/test_all.py --skip-mlir` | Optional (`quantum-opt` for phases 1–2) |

---

## Feature Summaries

### qasm3 — OpenQASM 3 Translation Pipeline

Converts quantum circuits through the full translation chain:

```
Qiskit QuantumCircuit → Catalyst MLIR → quantum-opt (canonicalize) → quantum-translate → OpenQASM 3.0
```

**Source**: `mlir/lib/Target/OpenQASM3/TranslateToQASM3.cpp`, `qiskit_importer_standalone.py`

**Test files**

| File | What it tests |
|------|--------------|
| `test_translation.py` | Legacy end-to-end runner across all 59 `.qasm` circuit files |
| `test_qasm3_translation_pytest.py` | Pytest suite — Bell states, GHZ, rotations, mid-circuit measurement, conditionals, edge cases, output format |
| `test_random_circuits_pytest.py` | Randomised circuits — stress, reproducibility, scalability, parametric sweeps |
| `run_all_tests.sh` | Shell wrapper that runs legacy + pytest suites with dependency checks |
| `random_circuit_generator.py` | Helper that generates random Clifford / medium / QV / small circuits |
| `qasm3_circuits/` | 59 hand-crafted QASM 2.0 input circuits |
| `random_circuits/` | Pre-generated random circuits (Clifford, medium, QV, small) |

**Run**

```bash
# All suites (from repo root)
bash test_qcc_feature/qasm3/run_all_tests.sh

# Pytest only
pytest test_qcc_feature/qasm3/test_qasm3_translation_pytest.py -v

# Random circuit stress tests
pytest test_qcc_feature/qasm3/test_random_circuits_pytest.py -v
```

**Build prerequisite**: `quantum-opt` and `quantum-translate` must be built:

```bash
make -C mlir dialects   # builds both binaries under mlir/build/bin/
```

**Last verified**: 59/59 legacy circuits pass, 28/28 pytest cases pass.

---

### do_qaoa — Differential-Optimization QAOA

Implements and validates the DO-QAOA algorithm (Sang et al., arXiv:2602.21689v1).
Partitions a MaxCut Hamiltonian into hotspot and peripheral qubits, transfers landscape
parameters across sub-problems, and achieves >10,000× shot-count speedup over FrozenQubits.

**Source**: `frontend/catalyst/api_extensions/doqaoa.py`

**Test files**

| File | What it tests |
|------|--------------|
| `test_phase1.py` | 4-qubit MaxCut MLIR round-trip through `quantum-opt` |
| `test_phase2.py` | Pearson r > 0.999 on 10 Erdős-Rényi graphs (landscape correlation) |
| `test_phase3.py` | 10-node BA graph, m=2 — shots ≤ 130,000, full-opt count = 1 |
| `test_phase4.py` | 12-node BA graph, m=3 — `do_qaoa()` API, shots ≤ 230,000, energy < 0 |
| `test_phase5.py` | All 25 acceptance criteria (C1–C11) + 6 benchmark figures |
| `test_acceptance_criteria.py` | C1–C11 standalone (no figures) — imported by `test_phase5.py` |
| `test_all.py` | Master runner for phases 1–5 |

**Run**

```bash
# Phases 3–5 only (no build required)
python test_qcc_feature/do_qaoa/test_all.py --skip-mlir

# All phases including MLIR round-trip (requires quantum-opt)
python test_qcc_feature/do_qaoa/test_all.py

# Single phase
python test_qcc_feature/do_qaoa/test_phase3.py
```

**Build prerequisite** (phases 1 & 2 only): `quantum-opt` must be built:

```bash
make -C mlir dialects   # binary lands at mlir/build/bin/quantum-opt
```

**Last verified**: phases 3, 4, 5 — all 25 acceptance criteria pass. Phase 3: 662 shots
(49,502× speedup). Phase 4: 508 shots (64,508× speedup). Phase 5 wall-clock speedup: 165×.

---

## Environment Setup

All tests assume the `catalyst` conda environment and are run from the **repo root**
(`/home/ubuntu/catalyst`).

```bash
conda activate catalyst

# Optional: enable Qiskit Aer for semantic simulation validation (qasm3 tests)
pip install qiskit-aer
```

The mlir_core Python bindings path is resolved automatically by each test's `conftest.py`
or module-level path setup — no manual `PYTHONPATH` export is needed.

---

## Contributor Guide

### Adding a new feature test suite

**1. Create your subdirectory**

```
test_qcc_feature/
└── your_feature/          ← name it after the feature (snake_case)
    ├── test_all.py        ← master runner (required)
    ├── test_*.py          ← individual test files
    └── README.md          ← feature-level README (required)
```

**2. Path convention**

All path navigation must be `__file__`-relative. Your tests sit two levels below the repo
root (`test_qcc_feature/your_feature/`), so use:

```python
import pathlib
ROOT = pathlib.Path(__file__).parent.parent.parent   # → repo root
```

Never hardcode absolute paths or CWD-relative strings like `"frontend/..."`.

**3. Write a `test_all.py` master runner**

Model it after [do_qaoa/test_all.py](do_qaoa/test_all.py). It should:
- Accept `-v` / `--verbose` flags
- Accept `--skip-<subsystem>` flags for steps that need optional build artifacts
- Print a clear PASS / FAIL / SKIP summary at the end
- Exit with code 0 on full pass, 1 on any failure

Minimal template:

```python
#!/usr/bin/env python3
import argparse, os, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))   # repo root

def run_script(name, verbose=False):
    path = os.path.join(HERE, name)
    result = subprocess.run(
        [sys.executable, path],
        capture_output=not verbose, text=True, cwd=ROOT,
    )
    output = "" if verbose else (result.stdout + result.stderr)
    return result.returncode == 0, output

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    phases = ["test_phase1.py"]   # list your test scripts here
    results = []
    for phase in phases:
        passed, output = run_script(phase, args.verbose)
        if not args.verbose and output:
            print(output)
        results.append((phase, passed))

    print("\nSUMMARY")
    all_passed = all(p for _, p in results)
    for name, passed in results:
        print(f"  {'PASS' if passed else 'FAIL'}  {name}")
    sys.exit(0 if all_passed else 1)

if __name__ == "__main__":
    main()
```

**4. Write a `README.md` for your feature**

Include at minimum:
- One-line description of what the feature does
- Table of test files and what each one tests
- How to run (quick command + full command)
- Any build prerequisites

**5. Update this file**

Add a row to the [Features](#features) table:

```markdown
| [your_feature/](your_feature/) | Short description | `python test_qcc_feature/your_feature/test_all.py` | Yes/No |
```

And add a summary section under [Feature Summaries](#feature-summaries) following the
same structure as the existing entries.

**6. Verify**

```bash
# Make sure your tests run cleanly from the repo root
python test_qcc_feature/your_feature/test_all.py
```
