# DO-QAOA Acceptance Tests — How to Run

**Project:** Doubly Optimized QAOA (Sang et al., arXiv:2602.21689v1)
**Implementation:** PennyLane Catalyst

---

## Quick Start (one command)

```bash
python test_acceptance_criteria.py
```

Expected output:
```
Ran 25 tests in ~45s

OK
```

---

## What the tests cover

All 11 acceptance criteria from the implementation plan:

| Criterion | What is tested | Target |
|---|---|---|
| C1  | Pearson r (m=1, no coefficients) — MaxCut spin-flip symmetry | r = 1.000 |
| C2  | Pearson r (m=1, with coefficients) — large free-qubit bias | \|r\| > 0.999 |
| C3  | ARG on power-law (BA) graphs, m=3 | median ≤ 30% |
| C4  | Total shots budget for m=1, 2, 3 | ≤ 170k / 250k |
| C5  | ARG on Erdős-Rényi graphs, m=3 | median ≤ 40% |
| C6  | FrozenQubits reference shots → speedup ≥ 262× | ≥ 262× |
| C7  | Within-graph r(\|S\|, ΔB) for m=3 over 10 ER graphs | mean > 0.79 |
| C8  | CNOT count per circuit call equals FrozenQubits | = 2 × \|edges\| |
| C9  | Landscape overlap q for concentrated graphs | q > 0.80 |
| C10 | Wall-clock speedup vs FrozenQubits | ≥ 10× |
| C11 | Platform support + module imports | macOS arm64 / Linux x86_64 |

---

## Prerequisites

Make sure you are inside the project virtual environment:

```bash
cd do-qaoa
source venv/bin/activate      # macOS / Linux
```

Verify dependencies are installed:

```bash
python -c "import pennylane, networkx, numpy; print('OK')"
```

---

## Run options

### Run all 25 tests (recommended)
```bash
python test_acceptance_criteria.py
```

### Run with verbose output (shows each test name)
```bash
python -m unittest test_acceptance_criteria -v
```

### Run a single criterion
```bash
# Example: only C8 (CNOT count)
python -m unittest test_acceptance_criteria.TestC8_CNOTCount -v
```

### Run individual benchmark scripts (standalone)
```bash
python phase5_task1_power_law_benchmark.py   # ARG power-law
python phase5_task2_landscape_correlation.py  # landscape r
python phase5_task3_compiler_pass_timing.py   # compile time
python phase5_task4_shot_regression.py        # shot budget
python phase5_task5_wallclock_profiling.py    # wall-clock speedup
```

### Regenerate all milestone figures
```bash
python generate_phase5_milestones.py
# Produces: phase5_task1_arg.png, phase5_task2_landscape.png,
#           phase5_task3_timing.png, phase5_task4_shots.png,
#           phase5_task5_wallclock.png, phase5_milestones.png
```

---

## Expected results

```
test_within_graph_r_equals_one     (C1)  ... ok
test_monotonic_increase_with_bias  (C2)  ... ok
test_path_graphs_mean_r            (C2)  ... ok
test_reference_4qubit_path         (C2)  ... ok
test_median_arg_m3                 (C3)  ... ok
test_shot_budget_m1                (C4)  ... ok
test_shot_budget_m2                (C4)  ... ok
test_shot_budget_m3                (C4)  ... ok
test_median_arg_er_m3              (C5)  ... ok
test_frozen_qubits_shot_formula    (C6)  ... ok
test_mean_within_graph_r_m3        (C7)  ... ok
test_ba12_m3                       (C8)  ... ok
test_ba8_m1                        (C8)  ... ok
test_ba8_m2                        (C8)  ... ok
test_er10_m1                       (C8)  ... ok
test_er10_m3                       (C8)  ... ok
test_concentrated_graphs_mean_q    (C9)  ... ok
test_er_m3_grand_mean              (C9)  ... ok
test_k3_m1_q_gt_080                (C9)  ... ok
test_wallclock_speedup_ge_10x      (C10) ... ok
test_doqaoa_module_loads           (C11) ... ok
test_numpy_available               (C11) ... ok
test_pennylane_version             (C11) ... ok
test_platform_supported            (C11) ... ok
test_result_type                   (C11) ... ok

----------------------------------------------------------------------
Ran 25 tests in ~45s

OK
```

---

## File overview

```
test_acceptance_criteria.py      ← main test file (run this)
phase5_task1_power_law_benchmark.py
phase5_task2_landscape_correlation.py
phase5_task3_compiler_pass_timing.py
phase5_task4_shot_regression.py
phase5_task5_wallclock_profiling.py
acceptance_c2_pearson_m1_with_coeff.py
acceptance_c5_er_arg_m3.py
acceptance_c8_cnot_count.py
acceptance_c9_landscape_overlap.py
generate_phase5_milestones.py    ← generates all figures
phase5_milestones.png            ← combined milestone figure
```
