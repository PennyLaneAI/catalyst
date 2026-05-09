# DO-QAOA Test Suite

## Files

| File | What it does |
|------|-------------|
| `test_phase1.py` | 4-qubit MaxCut MLIR round-trip — generates `quantum.freeze_partition` module, verifies it through `quantum-opt`, saves `benchmark_results/phase1_milestone.png` |
| `test_phase2.py` | Pearson r > 0.999 on 10 Erdős-Rényi graphs — validates landscape correlation, saves `benchmark_results/phase2_milestone.png` |
| `test_phase3.py` | 10-node BA graph, m=2 — runs full DO-QAOA training schedule, asserts shots ≤ 130,000, saves `benchmark_results/phase3_milestones.png` |
| `test_phase4.py` | 12-node BA graph, m=3 — calls `do_qaoa()` API, asserts shots ≤ 230,000 and energy < 0, saves `benchmark_results/phase4_milestones.png` |
| `test_phase5.py` | Acceptance criteria C1–C11 + benchmark figure generation (Tasks 1–5), saves 6 PNGs to `benchmark_results/` |
| `test_acceptance_criteria.py` | Acceptance criteria C1–C11 standalone (no figures) — used internally by `test_phase5.py` |
| `test_all.py` | Master runner — runs phases 1–5 in one command |

## How to Run

All commands from the repo root (`do-qaoa/`).

```bash
# Run phases 3, 4, 5 (no build required — skips MLIR phases 1 and 2)
python test_do_qaoa/test_all.py --skip-mlir

# Run a single phase
python test_do_qaoa/test_phase3.py
python test_do_qaoa/test_phase4.py
python test_do_qaoa/test_phase5.py

# Run acceptance criteria only (no figures)
python test_do_qaoa/test_acceptance_criteria.py
```

## Running Phase 1 and Phase 2 (requires quantum-opt)

Phase 1 and 2 test the MLIR compiler backend and need `quantum-opt` to be built first.

**Step 1 — Initialize submodules**
```bash
git submodule update --init --recursive
```

**Step 2 — Build LLVM/MLIR**
```bash
make -C mlir llvm
```

**Step 3 — Build dependencies**
```bash
make -C mlir stablehlo enzyme
```

**Step 4 — Build quantum-opt**
```bash
make -C mlir dialects
```

The binary will be at `mlir/build/bin/quantum-opt`. This takes 3–5 hours on first build.

**Step 5 — Run all tests including phase 1 and 2**
```bash
python test_do_qaoa/test_all.py
```
