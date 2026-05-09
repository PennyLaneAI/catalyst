# DO-QAOA Test Suite

## Files

| File | What it does |
|------|-------------|
| `test_phase1.py` | 4-qubit MaxCut MLIR round-trip — generates `quantum.freeze_partition` module and verifies it through `quantum-opt` |
| `test_phase2.py` | Pearson r > 0.999 on 10 Erdős-Rényi graphs — validates landscape correlation using exact QAOA statevector |
| `test_phase3.py` | 10-node BA graph, m=2 — runs full DO-QAOA training schedule, asserts shots ≤ 130,000 |
| `test_phase4.py` | 12-node BA graph, m=3 — calls `do_qaoa()` API, asserts shots ≤ 230,000 and energy < 0 |
| `test_acceptance_criteria.py` | Acceptance criteria C1–C11 from the paper (Pearson r, ARG, shot budget, CNOT count, speedup, etc.) |
| `test_all.py` | Master runner — runs all of the above in one command |

## How to Run

All commands from the repo root (`do-qaoa/`).

```bash
# Run everything (phases 3, 4 + acceptance criteria; skips phases needing quantum-opt)
python test_do_qaoa/test_all.py --skip-mlir

# Run everything including MLIR phases (requires quantum-opt on PATH)
python test_do_qaoa/test_all.py

# Skip slow acceptance tests (C3, C4, C5, C10)
python test_do_qaoa/test_all.py --skip-mlir --fast

# Run a single phase
python test_do_qaoa/test_phase3.py
python test_do_qaoa/test_phase4.py

# Run acceptance criteria only
python test_do_qaoa/test_acceptance_criteria.py
```
