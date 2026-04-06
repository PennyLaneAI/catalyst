# DO-QAOA Implementation Progress
**Project:** Doubly Optimized QAOA into PennyLane Catalyst (QCC 2.0)
**Reference:** Sang et al., arXiv:2602.21689v1 (2026)
**Timeline:** Mar 30 – May 1, 2026 (5 weeks)
**Key metric:** 200–300× reduction in quantum shots

---

## Project Overview

DO-QAOA reduces O(2^m) training sessions to O(K≈1) by exploiting landscape
similarity across frozen sub-problems via divide-and-conquer. The core idea:

1. Freeze m hotspot qubits → produce 2^m sub-problems
2. Cluster sub-problem energy landscapes → K similarity groups (K≈1 for sparse graphs)
3. Optimise only the representative sub-circuit
4. Transfer parameters to all others via the Bias-Aware Transfer Rule
5. Pick the bitstring with minimum expectation value

---

## Phase 1 — IR Extension & Dialect Design
**Duration:** Mar 30 – Apr 6, 2026
**Status:** COMPLETE ✅

---

### Task 1 — Define 5 types + 5 ops in TableGen ODS

**Files:**
- `catalyst/mlir/include/Quantum/IR/QuantumTypes.td`
- `catalyst/mlir/include/Quantum/IR/QuantumOps.td`

**5 new types** added to the `quantum` dialect:

| Mnemonic | Parameters | Purpose |
|---|---|---|
| `!quantum.partition<N, m>` | numQubits, m | Carries total qubit count N and frozen qubit count m. Entry point of DO-QAOA. |
| `!quantum.cluster_map<K>` | k | Holds K cluster assignments for 2^m sub-problem landscapes. |
| `!quantum.circuit_ref` | — | Opaque index identifying the representative sub-circuit per cluster. |
| `!quantum.params` | — | Variational parameter buffer θ = (γ₁..γₚ, β₁..βₚ). |
| `!quantum.bitstring` | — | Final binary node assignment with minimum ⟨H⟩. |

`PartitionType` and `ClusterMapType` have integer template parameters (e.g. `<10, 3>`, `<2>`) with custom `assemblyFormat`. The other three are parameter-less and print as bare mnemonics.

**5 new ops** — all carry the `[Pure]` trait (no side effects):

| Op | Inputs | Output | Role |
|---|---|---|---|
| `quantum.freeze_partition` | `hotspot_count: i32`, `hotspot_indices: array<i32>` | `!quantum.partition<N,m>` | Annotates which qubits are frozen |
| `quantum.landscape_cluster` | `!quantum.partition`, `k: i32` | `!quantum.cluster_map<K>` | Groups 2^m landscapes into K clusters |
| `quantum.select_representative` | `!quantum.cluster_map` | `!quantum.circuit_ref` | Picks the representative sub-circuit |
| `quantum.bias_transfer` | `!quantum.params`, `B_rep`, `B_target`, `threshold` (f64) | `!quantum.params` | Direct copy or warm-start based on ΔB |
| `quantum.aggregate_min` | `Variadic<!quantum.params>` | `!quantum.bitstring` | Selects minimum expectation value result |

`freeze_partition` has `let hasVerifier = 1` enabling the C++ verifier in Task 2.

---

### Task 2 — C++ Verifier for FreezePartitionOp

**File:** `catalyst/mlir/lib/Quantum/IR/QuantumOps.cpp`

`FreezePartitionOp::verify()` enforces 3 invariants at compile time:

1. `hotspot_count == len(hotspot_indices)` — count attribute must match actual array length
2. `hotspot_count <= numQubits` — cannot freeze more qubits than the circuit has
3. `hotspot_count == type parameter m` — the attribute and the type must agree

Uses `mlir::cast<PartitionType>(...)` (modern MLIR syntax).

---

### Task 3 — Type System: LLVM Lowering

**File:** `catalyst/mlir/lib/Quantum/Transforms/quantum_to_llvm.cpp`

5 conversion rules added to `QIRTypeConverter`:

| DO-QAOA type | LLVM type | Rationale |
|---|---|---|
| `!quantum.partition<N, m>` | `!llvm.struct<(i32, i32)>` | Two fields: numQubits and m |
| `!quantum.cluster_map<K>` | `!llvm.struct<(i32)>` | One field: k |
| `!quantum.circuit_ref` | `i64` | Opaque index — same pattern as ObservableType |
| `!quantum.params` | `!llvm.ptr` | Pointer to heap f64 buffer |
| `!quantum.bitstring` | `!llvm.ptr` | Pointer to heap i8 buffer |

---

### Task 4 — Graph Metadata IR Representation

**Files:**
- `catalyst/mlir/include/Quantum/IR/QuantumAttrDefs.td`
- `catalyst/mlir/lib/Quantum/IR/QuantumAttrs.cpp`

Two new MLIR attributes for the J_ij adjacency and edge-weight matrix:

**`DenseGraphAttr`** — small graphs (N ≤ 64)
```
#quantum.dense_graph<4, dense<[[0.0, -0.5, 0.0, -0.5], ...]> : tensor<4x4xf64>>
```
Stores the full symmetric NxN f64 weight matrix as a `DenseElementsAttr`.
Verifier checks: rank-2 tensor, element type f64, shape matches numNodes.

**`SparseGraphAttr`** — large graphs (N > 64)
```
#quantum.sparse_graph<100, 3, [0, 0, 1], [1, 2, 2], dense<[-0.5, -0.5, -0.5]> : tensor<3xf64>>
```
COO upper-triangle encoding: row indices, col indices, 1D weight `DenseElementsAttr`.
Verifier checks: array lengths equal numEdges, indices in bounds, strict upper-triangle (i < j).

Both attributes call `verify()` inside `parse()` before `::get()` to emit graceful errors instead of assertion crashes.

---

### Task 5 — FileCheck Unit Tests (34 tests)

**Files:** `catalyst/mlir/test/Quantum/`

| File | Tests | What it covers |
|---|---|---|
| `DOQAOADialectTest.mlir` | 14 | Round-trip parse/print for all 5 ops and type combinations |
| `DOQAOAVerifierTest.mlir` | 3 | `freeze_partition` verifier — all 3 invariant violations |
| `DOQAOALoweringTest.mlir` | 6 | LLVM type lowering — each type individually and all together |
| `DOQAOAGraphMetadataTest.mlir` | 4 | Dense/sparse attribute round-trip |
| `DOQAOAGraphMetadataVerifierTest.mlir` | 7 | Dense/sparse attribute verifier errors |

All integrated into Catalyst's existing lit test structure.

---

### Task 6 — doqaoa_partition() Python Decorator

**File:** `catalyst/frontend/catalyst/api_extensions/doqaoa.py`

**`DOQAOAConfig` dataclass** — 6 validated fields:

| Field | Default | Purpose |
|---|---|---|
| `m` | required | Number of frozen hotspot qubits (≥1) |
| `bias_threshold` | 0.3 | ΔB cutoff — below this, direct copy; above, warm-start |
| `warmstart_epochs` | 10 | Adam optimisation epochs for warm-start branch |
| `init_strategy` | `"shortcut"` | Analytic p=1 init `(-π/6, -π/8)` or `"random"` |
| `k_max` | None | Max landscape clusters (None = auto) |
| `landscape_grid_size` | 16 | Grid resolution for energy sampling (16→256 evals) |

**`DOQAOAPartitionCallable`** wraps a QNode and exposes:
- `.hotspot_indices` — selected frozen qubit indices
- `.config` — the `DOQAOAConfig` used
- `.num_qubits` — total qubit count from the underlying device
- `__call__` — delegates to the original QNode unchanged

**`doqaoa_partition(qnode, *, graph, config)`** works as both decorator and decorator factory:
```python
config = DOQAOAConfig(m=2)
G = nx.barabasi_albert_graph(10, 2)

@doqaoa_partition(graph=G, config=config)
@qml.qnode(dev)
def circuit(params):
    ...

print(circuit.hotspot_indices)  # [0, 1, 4] — highest degree nodes
```

---

### Task 7 — Graph Analysis Utilities (NetworkX)

**File:** `catalyst/frontend/catalyst/api_extensions/doqaoa.py`

**`degree_centrality_sort(graph)`**
Sorts all nodes by NetworkX degree centrality, highest first. Identifies which
qubits most influence the energy landscape — nodes with most edges create the
most variation across sub-problems (Section 1.1 of arXiv:2602.21689v1).

**`select_hotspot_indices(graph, m)`**
Calls `degree_centrality_sort`, picks the top-m nodes, returns them sorted
ascending for deterministic qubit ordering.

Example on a Barabási-Albert graph:
```python
G = nx.barabasi_albert_graph(10, 2, seed=42)
select_hotspot_indices(G, m=3)  # → [0, 1, 4]  (hub nodes)
```

---

### Task 8 — PennyLane Hamiltonian Bridge

**File:** `catalyst/frontend/catalyst/api_extensions/doqaoa.py`

**`extract_coupling_matrix(H)`**
Parses a PennyLane Hamiltonian into:
- `J`: dict `(i, j) → coefficient` for ZZ (quadratic) terms
- `h`: dict `i → coefficient` for Z (linear) terms

Handles `PauliZ`, `PauliZ @ PauliZ`, `Prod`, and `SProd` operator forms.
Covers both MaxCut (pure ZZ, h=0) and general Ising Hamiltonians.

**`compute_bias(H, N)`**
Computes `B = (1/N) Σ|hᵢ|`. Used by the Bias-Aware Transfer Rule:
if `|B_target − B_rep| < threshold` → direct copy, else warm-start.

**`hamiltonian_to_graph_attrs(H, N, sparse_threshold=64)`**
Full MLIR bridge. Takes a PennyLane Hamiltonian, returns two attribute strings:

- **H_quad**: `#quantum.dense_graph<N, …>` for N ≤ 64, `#quantum.sparse_graph<N, E, …>` for N > 64
- **H_lin**: `dense<[h_0, …, h_{N-1}]> : tensor<Nxf64>`

```python
H = qml.Hamiltonian([-0.5,-0.5,-0.5,-0.5], [Z0Z1, Z1Z2, Z2Z3, Z3Z0])
h_quad, h_lin = hamiltonian_to_graph_attrs(H, num_qubits=4)
# h_quad → '#quantum.dense_graph<4, dense<[[0.0,-0.5,...]]> : tensor<4x4xf64>>'
# h_lin  → 'dense<[0.0, 0.0, 0.0, 0.0]> : tensor<4xf64>'
```

Exported in `api_extensions/__init__.py`: `DOQAOAConfig`, `doqaoa_partition`, `hamiltonian_to_graph_attrs`.

---

## Phase 1 Milestone — 4-qubit MaxCut Round-Trip

**Files:**
- `milestone_maxcut.py` — generates the MLIR module and runs round-trip
- `MaxCut4QubitMilestone.mlir` — generated MLIR output
- `milestone_plot.py` — generates the visual proof figure
- `phase1_milestone.png` — visual proof (see below)

**Circuit:** 4-node cycle graph (0-1-2-3-0), QAOA p=1 ansatz
**Hamiltonian:** H = −0.5·(Z₀Z₁ + Z₁Z₂ + Z₂Z₃ + Z₃Z₀)
**Config:** m=2, K=1, threshold=0.3, init=shortcut

**What the Python script does:**
1. `doqaoa_partition` decorator selects hotspot qubits `[0, 1]` via degree centrality
2. `hamiltonian_to_graph_attrs` converts H to `#quantum.dense_graph<4, tensor<4×4×f64>>`
3. Generates full MLIR module with all 5 DO-QAOA ops chained
4. Runs `quantum-opt` round-trip — verifies parse → print → re-parse is lossless

**Generated MLIR (key section):**
```mlir
func.func @maxcut_4qubit_doqaoa(%params_rep: !quantum.params) {
  %partition = quantum.freeze_partition {
      hotspot_count   = 2 : i32,
      hotspot_indices = array<i32: 0, 1>,
      h_quad = #quantum.dense_graph<4, dense<[[0.0,-0.5,0.0,-0.5],...]> : tensor<4x4xf64>>,
      h_lin  = dense<[0.0, 0.0, 0.0, 0.0]> : tensor<4xf64>
  } : !quantum.partition<4, 2>

  %cluster_map = quantum.landscape_cluster(
      %partition : !quantum.partition<4, 2>) {k = 1 : i32} : !quantum.cluster_map<1>

  %circuit_ref = quantum.select_representative(
      %cluster_map : !quantum.cluster_map<1>) : !quantum.circuit_ref

  %params_out = quantum.bias_transfer(%params_rep : !quantum.params)
      {B_rep = 0.0 : f64, B_target = 0.0 : f64, threshold = 3.0e-01 : f64}
      : !quantum.params

  %bitstring = quantum.aggregate_min(
      %params_out : !quantum.params) : !quantum.bitstring

  func.return
}
```

**Milestone result: PASS**

---

## Visual Proof

![Phase 1 Milestone](phase1_milestone.png)

**Panel descriptions:**
- **Top left:** 4-node cycle graph. Purple ★ = hotspot qubits [0,1] selected by degree centrality. Green edges with weight −0.5.
- **Top middle:** J_ij coupling matrix heatmap. Blue = −0.5 coupling. Purple column/row highlights = hotspot qbit columns. All diagonal and non-edge entries = 0.
- **Top right:** Bias-Aware Transfer Rule decision. ΔB = 0.0 < θ = 0.3 → Direct copy branch. Zero extra training sessions.
- **Bottom left:** Full 5-op DO-QAOA pipeline with `quantum-opt round-trip: PASS` badge.
- **Bottom right:** Test results bar chart — 56/56 checks passing across all test suites.

---

## Phase 1 — Complete Test Results

```
DOQAOADialectTest.mlir              14/14  PASS   (round-trip parse/print)
DOQAOAVerifierTest.mlir              3/3   PASS   (freeze_partition verifier)
DOQAOALoweringTest.mlir              6/6   PASS   (LLVM type lowering)
DOQAOAGraphMetadataTest.mlir         4/4   PASS   (dense/sparse attrs)
DOQAOAGraphMetadataVerifierTest.mlir 7/7   PASS   (attr verifier errors)
Python API smoke tests              22/22  PASS   (config, decorator, bridge)
──────────────────────────────────────────────────────────────────────────
Total                               56/56  PASS   0 failures
```

---

## Phase 1 — All Modified/Created Files

### MLIR (C++ / TableGen)
| File | Change |
|---|---|
| `catalyst/mlir/include/Quantum/IR/QuantumTypes.td` | Added 5 DO-QAOA types |
| `catalyst/mlir/include/Quantum/IR/QuantumOps.td` | Added 5 DO-QAOA ops |
| `catalyst/mlir/include/Quantum/IR/QuantumAttrDefs.td` | Added `DenseGraphAttr` + `SparseGraphAttr` |
| `catalyst/mlir/lib/Quantum/IR/QuantumOps.cpp` | Added `FreezePartitionOp::verify()` |
| `catalyst/mlir/lib/Quantum/IR/QuantumAttrs.cpp` | New — parse/print/verify for graph attrs |
| `catalyst/mlir/lib/Quantum/IR/CMakeLists.txt` | Registered `QuantumAttrs.cpp` |
| `catalyst/mlir/lib/Quantum/Transforms/quantum_to_llvm.cpp` | Added 5 LLVM type converter rules |

### Tests
| File | Tests |
|---|---|
| `catalyst/mlir/test/Quantum/DOQAOADialectTest.mlir` | 14 |
| `catalyst/mlir/test/Quantum/DOQAOAVerifierTest.mlir` | 3 |
| `catalyst/mlir/test/Quantum/DOQAOALoweringTest.mlir` | 6 |
| `catalyst/mlir/test/Quantum/DOQAOAGraphMetadataTest.mlir` | 4 |
| `catalyst/mlir/test/Quantum/DOQAOAGraphMetadataVerifierTest.mlir` | 7 |

### Python
| File | Change |
|---|---|
| `catalyst/frontend/catalyst/api_extensions/doqaoa.py` | New — full Python API |
| `catalyst/frontend/catalyst/api_extensions/__init__.py` | Exported 3 new symbols |

### Milestone
| File | Purpose |
|---|---|
| `milestone_maxcut.py` | Generates + round-trips MaxCut MLIR |
| `MaxCut4QubitMilestone.mlir` | Generated MLIR module |
| `milestone_plot.py` | Generates visual proof figure |
| `phase1_milestone.png` | Visual proof — all 4 panels |

---

---

## Phase 2 — Landscape Analysis
**Duration:** Apr 6–13, 2026
**Status:** Tasks 1–8 COMPLETE ✅ | Phase 2 done

---

### Task 1 — LandscapeOverlapAnalysis Pass

**Files:**
- `catalyst/mlir/include/Quantum/Transforms/Passes.td`
- `catalyst/mlir/lib/Quantum/Transforms/LandscapeOverlapAnalysis.cpp`
- `catalyst/mlir/test/Quantum/DOQAOALandscapeOverlapTest.mlir`

**What it does:**

For each `quantum.freeze_partition` op in a `func.FuncOp`:

1. **Extract J_ij** from `h_quad` — handles both `DenseGraphAttr` (N ≤ 64) and `SparseGraphAttr` (N > 64). Also extracts linear bias vector `h_lin`.

2. **Enumerate 2^m sub-problems** — each sub-problem k fixes the m hotspot qubits to spin values ±1 via the bit representation of k.

3. **Evaluate E_k(γ,β) on a 16×16 grid** using the QAOA p=1 closed-form Ising expression:
   - Two-body contribution from free-free edges: `J/2 · sin(4β) · sin(2γJ)`
   - Single-body contribution from free qubits with effective bias: `h_eff · (−sin(2β) · cos(2γ·h_eff))`
   - Effective bias `h_eff[u]` accumulates the field from frozen neighbours: `h[u] + Σ_v J[u,v]·s_v`
   - Constant offset from frozen-frozen coupling

4. **L2-normalise** each landscape vector (gridSize² = 256 elements).

5. **Compute pairwise cosine similarity S_kl** (Eq. 2.6 of arXiv:2602.21689v1):
   `S_kl = v_k · v_l` (since vectors are already normalised)

6. **Compute mean overlap** `q = mean(S_kl)` over all C(2^m, 2) pairs.

7. **Annotate** the `freeze_partition` op with:
   - `landscape_overlap_q : f64` — mean cosine similarity
   - `recommended_k : i32` — 1 if `q ≥ overlap_threshold` (default 0.9), else 2^m

**Pass options:**

| Option | Default | Description |
|---|---|---|
| `--grid-size` | 16 | Grid points per axis (total = gridSize²) |
| `--overlap-threshold` | 0.9 | Cosine similarity above which K=1 |

**Real physics observed (closed-form approximation, initial implementation):**

| Circuit | m | q | recommended_k | Why |
|---|---|---|---|---|
| 4-cycle MaxCut (pure ZZ) | 2 | −0.257 | 4 | Anti-correlated landscapes |
| Ising with strong bias (h=2.0) | 1 | +0.992 | 1 | Near-identical landscapes |

**FileCheck tests:** 5 tests in `DOQAOALandscapeOverlapTest.mlir` — all PASS

---

### Task 2 — Exact/Sample Energy Eval Backend (EnergyEval)

**Status:** COMPLETE ✅

**Files:**
- `catalyst/mlir/include/Quantum/Transforms/EnergyEval.h`
- `catalyst/mlir/lib/Quantum/Transforms/EnergyEval.cpp`
- `catalyst/mlir/lib/Quantum/Transforms/LandscapeOverlapAnalysis.cpp` (rewired)
- `catalyst/mlir/test/Quantum/DOQAOAEnergyEvalTest.mlir`

**What it does:**

Replaces the closed-form landscape approximation with a full QAOA p=1 statevector simulation, giving physically correct energy expectations:

**Exact path (N_free ≤ 20 qubits):**

1. Freeze m hotspot qubits to ±1 based on subproblem index k.
2. Compute effective bias `h_eff[u] = h[u] + Σ_v J[u,v]·s_v` for all free qubits.
3. Build the free-free J sub-matrix (N_free × N_free).
4. Initialise statevector `|ψ⟩ = |+⟩^⊗N_free`, amplitude = 1/√(2^N_free).
5. Apply cost unitary C(γ): `|z⟩ → exp(−iγ·E_z)|z⟩` where E_z is the Ising energy of basis state z.
6. Apply mixer B(β): product of RX(2β) gates qubit-by-qubit (stride-loop over the statevector).
7. Compute `⟨H⟩ = Σ_z |ψ_z|² · E_z`.

**Sample path (N_free > 20 qubits):**

- Draw 512 independent samples from Born marginal `P(z_u=1) ≈ sin²(β)` (valid for |+⟩ init at p=1).
- Score each sample with Ising energy, return mean.
- Sufficient for landscape *shape* comparison (cosine similarity), not required to be exact.

**Caching:**

- Thread-local `unordered_map<string, vector<double>>` keyed by `"<topologyHex>_k{k}_g{gridSize}"`.
- Topology key = hex-encoded upper-triangle J + h values (non-zero entries only).
- `flushCache()` called at the start of each `FuncOp` by `LandscapeOverlapAnalysis`.
- Cache hit skips the full statevector computation.

**Corrected real physics (exact statevector backend):**

| Circuit | m | N_free | q | recommended_k | Physics |
|---|---|---|---|---|---|
| 4-cycle MaxCut | 2 | 2 | 0.586 | 4 | Landscapes differ across 4 sub-problems |
| 4-cycle MaxCut | 1 | 3 | 1.000 | 1 | Freezing one qubit → symmetric bias flip → identical landscapes |
| Ising, strong bias (h=2.0) | 1 | 3 | 1.000 | 1 | h >> J → landscapes dominated by bias, near-identical |
| Path graph (sparse) | 2 | 2 | 1.000 | 1 | Sparse, symmetric structure → similar landscapes |

**Key physics insight:** For m=1, freezing qubit 0 to +1 vs −1 flips the effective bias on its neighbours symmetrically. If the graph is symmetric (same edge weights from qubit 0 to both affected neighbours), the two landscapes are identical (q=1) and only K=1 training session is needed.

**FileCheck tests:** 5 tests in `DOQAOAEnergyEvalTest.mlir` + 5 updated tests in `DOQAOALandscapeOverlapTest.mlir` — all PASS

| Test file | Test | Checks |
|---|---|---|
| `DOQAOAEnergyEvalTest.mlir` | `@exact_path_m2` | N_free=2, exact path, attributes written |
| `DOQAOAEnergyEvalTest.mlir` | `@exact_strong_bias_k1` | Strong bias h=5 → `recommended_k = 1` |
| `DOQAOAEnergyEvalTest.mlir` | `@exact_sparse_path` | SparseGraphAttr path handles correctly |
| `DOQAOAEnergyEvalTest.mlir` | `@cache_hit_identical_graphs` | Two identical ops → both get same `landscape_overlap_q` |
| `DOQAOAEnergyEvalTest.mlir` | `@no_hquad_skipped` | Missing attr → pass skips, no attribute written |
| `DOQAOALandscapeOverlapTest.mlir` | `@maxcut_4cycle_m2` | `landscape_overlap_q` written, `recommended_k = 4` |
| `DOQAOALandscapeOverlapTest.mlir` | `@maxcut_4cycle_m1` | `recommended_k = 1` (corrected: q≈1.0 ≥ threshold) |
| `DOQAOALandscapeOverlapTest.mlir` | `@ising_with_bias` | `recommended_k = 1` |
| `DOQAOALandscapeOverlapTest.mlir` | `@no_h_quad` | Pass skips gracefully |
| `DOQAOALandscapeOverlapTest.mlir` | `@sparse_path_graph` | Sparse path, attributes written |

**Phase 2 test totals:**

```
DOQAOALandscapeOverlapTest.mlir   5/5   PASS
DOQAOAEnergyEvalTest.mlir         5/5   PASS
──────────────────────────────────────────────
Phase 2 total                    10/10  PASS
```

---

---

### Task 3 — Phase Transition Detector

**Status:** COMPLETE ✅

**Files modified:**
- `catalyst/mlir/include/Quantum/Transforms/Passes.td` — added `scThreshold` option
- `catalyst/mlir/lib/Quantum/Transforms/LandscapeOverlapAnalysis.cpp` — BFS diameter + s_eff
- `catalyst/mlir/test/Quantum/DOQAOAPhaseTransitionTest.mlir` — 5 new tests (10 CHECK lines)

**What it does:**

In the DO-QAOA paper, the "concentrated" vs "fragmented" landscape regime is controlled by an effective parameter s. Above the phase transition threshold sc ≈ 0.6, all sub-problem landscapes are similar and parameter transfer works. Below sc, landscapes diverge and DO-QAOA may not achieve K=1.

This detector estimates s from graph structure (cheap, no energy eval needed) as a pre-flight diagnostic.

**BFS diameter computation:**

```
Adjacency: edge (u,v) exists if |J[u*N+v]| > 1e-12
BFS from every node → eccentricity (max distance)
Diameter = max eccentricity across all source nodes
Disconnected pairs → distance treated as N-1 (conservative)
```

**Heuristic formula (Sang et al.):**

```
s_eff = 2 / (1 + diameter)
```

| Graph type | Diameter | s_eff | Regime |
|---|---|---|---|
| Complete K4 | 1 | 1.000 | Concentrated (s > sc=0.6) ✓ |
| 4-cycle | 2 | 0.667 | Concentrated ✓ |
| Path P4 | 3 | 0.500 | **Fragmented** — warning emitted |
| Path P5 | 4 | 0.400 | **Fragmented** — warning emitted |

**Pass option added:**

```
--sc-threshold=<double>  (default 0.6)
```

**Attribute written:**

```
s_eff : f64   -- on every freeze_partition op (always, regardless of regime)
```

**Warning emitted (fragmented regime):**

```
warning: doqaoa-landscape-overlap: fragmented landscape regime
  (s_eff=0.500 < sc=0.600, diameter=3);
  DO-QAOA parameter transfer may not achieve K=1
```

**FileCheck tests:** 5 tests in `DOQAOAPhaseTransitionTest.mlir` — tested with two check prefixes:

| Prefix | Runs | Purpose |
|---|---|---|
| `ATTR` | `2>/dev/null` (IR only) | Verify `s_eff` attribute written on all 5 ops |
| `WARN` | `2>&1` (combined) | Verify warning text for the 3 fragmented-regime cases |

| Test | Graph | Diameter | s_eff | Warning? |
|---|---|---|---|---|
| `@complete_graph_k4` | K4 complete | 1 | 1.000 | No |
| `@cycle_4_concentrated` | 4-cycle | 2 | 0.667 | No |
| `@path_5_fragmented` | Path P5 | 4 | 0.400 | Yes — "s_eff=0.400" + "diameter=4" |
| `@path_4_fragmented` | Path P4 | 3 | 0.500 | Yes — "diameter=3" |
| `@sparse_path_fragmented` | Sparse P4 | 3 | 0.500 | Yes — "sc=0.600" |

**Phase 2 running test totals:**

```
DOQAOALandscapeOverlapTest.mlir   5/5   PASS
DOQAOAEnergyEvalTest.mlir         5/5   PASS
DOQAOAPhaseTransitionTest.mlir   5/5   PASS  (ATTR + WARN prefixes)
──────────────────────────────────────────────
Phase 2 total                    15/15  PASS
```

---

### Tasks 4–8 — Clustering, Bias Shift, Threshold, Shortcut Init, Basin Analysis

**Status:** COMPLETE ✅

**Files created/modified:**
- `catalyst/mlir/lib/Quantum/Transforms/LandscapeOverlapAnalysis.cpp` — K-means + elbow (Task 4)
- `catalyst/mlir/include/Quantum/Transforms/EnergyEval.h` — `computeBias()` API
- `catalyst/mlir/lib/Quantum/Transforms/EnergyEval.cpp` — `computeBias()` implementation
- `catalyst/mlir/lib/Quantum/Transforms/BiasShiftAnalysis.cpp` — new pass (Tasks 5+7+8)
- `catalyst/mlir/include/Quantum/Transforms/Passes.td` — `doqaoa-bias-shift` pass
- `catalyst/mlir/lib/Quantum/Transforms/CMakeLists.txt` — registered BiasShiftAnalysis
- `doqaoa_threshold_sweep.py` — Python sweep utility (Task 6)
- `catalyst/mlir/test/Quantum/DOQAOAClusterTest.mlir` — 5 tests
- `catalyst/mlir/test/Quantum/DOQAOABiasShiftTest.mlir` — 4 tests × 3 prefixes

---

**Task 4 — K-means + Elbow Cluster Assignment**

Added to `LandscapeOverlapAnalysis.cpp`:
- **K-means++ init**: seed centroid + D²-weighted sampling for subsequent centroids
- **Lloyd's algorithm**: 100 iterations, convergence check
- **Elbow**: first K where cumulative variance explained ≥ 90%
- **Shortcut**: if q ≥ overlap_threshold → skip K-means (K=1, all in cluster 0)

New attributes on `freeze_partition`:
- `cluster_k : i32` — number of clusters found
- `cluster_assignments : array<i32, 2^m>` — which cluster each sub-problem belongs to

Physics observed:
| Case | q | cluster_k | assignments |
|---|---|---|---|
| Complete K4 m=1 | ≈1.0 | 1 | [0, 0] |
| 4-cycle m=1 | ≈1.0 | 1 | [0, 0] |
| 4-cycle m=2 | 0.586 | **2** | [1, 0, 0, 1] |
| Sparse path m=2 | ≈1.0 | 1 | [0, 0, 0, 0] |

The 4-cycle m=2 correctly clusters into K=2 (symmetric pairs {k=0,k=3} and {k=1,k=2}).

---

**Task 5 — BiasShiftAnalysis Pass**

New pass `doqaoa-bias-shift`:
1. For each freeze_partition (already annotated by doqaoa-landscape-overlap):
   - Compute `B_k = (1/N_free) Σ|h_eff[i]|` for every sub-problem k
   - Pick representative of each cluster = member with minimum B_k
   - Compute `ΔB_k = |B_k − B_rep|` for all k
2. Writes: `b_values`, `bias_shifts`, `representatives`

---

**Task 6 — Threshold Calibration Utility**

`doqaoa_threshold_sweep.py`:
- Benchmark suite: Complete K8, Cycle C6, Path P6, Star S6, Biased Cycle
- Sweeps ΔB threshold 0.00→0.80 in steps of 0.01
- ARG(θ) = fraction of sub-problems eligible for direct copy at threshold θ
- Produces `doqaoa_threshold_sweep.png` (2-panel: ARG curve + ΔB box plot)

```
Graph                  ARG@0.1   ARG@0.3*   ARG@0.5    (* paper default)
-------------------------------------------------------
  Complete K8           33.3%    100.0%   100.0%
  Cycle C6             100.0%    100.0%   100.0%
  Path P6              100.0%    100.0%   100.0%
  Star S6              100.0%    100.0%   100.0%
  Biased Cycle          33.3%    100.0%   100.0%
```

Paper default θ=0.3 achieves 100% ARG on all benchmark graphs.

Pass option added: `--bias-threshold=<double>` (default 0.3).

---

**Task 7 — Shortcut Initialisation**

Written by BiasShiftAnalysis into every annotated `freeze_partition`:
- `init_gamma = −π/6 ≈ −0.5236 rad` — physics-informed p=1 starting point
- `init_beta  = −π/8 ≈ −0.3927 rad`

These are the universal shortcut values for Ising QAOA p=1 from Sang et al.

---

**Task 8 — Parameter Concentration / Basin Analysis**

Also in BiasShiftAnalysis:
1. Calls `energy::buildLandscapeVector(graph, repK, gridSize)` — **cache hit** from doqaoa-landscape-overlap (cache not flushed between passes in the same pipeline)
2. Finds argmin of the landscape vector → converts to `(basin_gamma, basin_beta)` via grid formula
3. If `|basin_gamma − init_gamma| > basinTol` OR `|basin_beta − init_beta| > basinTol` → warning

Pass option: `--basin-tol=<double>` (default 0.2).

Example warning (tight tol=0.001 on 4-cycle):
```
warning: doqaoa-bias-shift: basin centre (2.304, 0.524) deviates from
  shortcut (-0.524, -0.393) by (2.827, 0.916) > tol=0.001;
  shortcut init may be suboptimal for this graph
```

---

**Phase 2 complete test totals:**

```
DOQAOALandscapeOverlapTest.mlir      5/5   PASS
DOQAOAEnergyEvalTest.mlir            5/5   PASS
DOQAOAPhaseTransitionTest.mlir       5/5   PASS  (ATTR + WARN)
DOQAOAClusterTest.mlir               5/5   PASS  (ATTR)
DOQAOABiasShiftTest.mlir            4×3 = 12/12 PASS  (ATTR + BASIN + GUARD)
─────────────────────────────────────────────────
Phase 2 total                       32/32  PASS
```

---

## Phase 2 Milestone — Pearson r > 0.999 on Erdős-Rényi Graphs

**Status:** PASS ✅

**Files:**
- `milestone_erdos_renyi.py` — generates ER graphs, runs full mlir-opt pipeline, computes Pearson r
- `milestone_plot_phase2.py` — dedicated 6-panel dark-theme visual proof (same style as Phase 1)
- `phase2_milestone.png` — saved figure

**Deliverable:** Working mlir-opt pipeline that takes a `freeze_partition` module, runs `LandscapeOverlapAnalysis + BiasShiftAnalysis`, and annotates all sub-problems with `cluster_id` and `bias_shift`. Validated on 10-node Erdős-Rényi graphs.

**What the script does:**

1. Generates 10 random connected Erdős-Rényi G(10, 0.5) graphs (seed=42)
2. For each graph: selects m=1 hotspot via degree centrality
3. Builds two landscape vectors (k=0: hotspot=+1, k=1: hotspot=−1) using exact QAOA p=1 statevector (mirrors EnergyEval.cpp)
4. Computes Pearson r between the two N_free=9 qubit landscape vectors (16×16=256 grid points)
5. Generates MLIR module with `freeze_partition` + runs `quantum-opt --doqaoa-landscape-overlap --doqaoa-bias-shift`
6. Parses IR output to verify `cluster_k`, `bias_shifts` are annotated
7. Saves 6-panel plot

**Results:**

```
Graph          r       q_IR  cluster_k  bias_shift[1]  Status
----------------------------------------------------------------------
    1    1.000000    1.000000           1          0.0000  PASS ✓
    2    1.000000    1.000000           1          0.0000  PASS ✓
    3    1.000000    1.000000           1          0.0000  PASS ✓
    4    1.000000    1.000000           1          0.0000  PASS ✓
    5    1.000000    1.000000           1          0.0000  PASS ✓
    6    1.000000    1.000000           1          0.0000  PASS ✓
    7    1.000000    1.000000           1          0.0000  PASS ✓
    8    1.000000    1.000000           1          0.0000  PASS ✓
    9    1.000000    1.000000           1          0.0000  PASS ✓
   10    1.000000    1.000000           1          0.0000  PASS ✓
----------------------------------------------------------------------
Pearson r:  min=1.000000  mean=1.000000  max=1.000000
Graphs with r > 0.999: 10/10

MILESTONE PASS ✓  —  all 10 graphs r > 0.999
```

**Why r = 1.000 for all ER graphs with m=1?**

For m=1 (single frozen qubit), the two sub-problems k=0 and k=1 differ only in the sign of the frozen spin. In a zero-bias Ising model (h=0), flipping the hotspot spin flips the effective bias on its neighbours: `h_eff[u] → −h_eff[u]`. The QAOA energy landscape satisfies `E_{k=1}(γ,β) = E_{k=0}(−γ,β)` — a simple reflection. After L2-normalisation, the two vectors are identical up to the sign of γ, which means their dot product (cosine similarity) is exactly 1 and Pearson r = 1. This confirms DO-QAOA's core claim: K=1 training session suffices for sparse/random graphs with m=1.

**Plot panels:**
- Top left: Pearson r per graph (all above 0.999 threshold)
- Top middle: Landscape vectors for k=0 vs k=1 on graph #10
- Top right: Graph #10 topology (orange = hotspot node)
- Bottom left: r distribution histogram
- Bottom middle: Scatter plot lv0 vs lv1 (perfect diagonal → r=1)
- Bottom right: Pipeline summary + MILESTONE PASS badge

![Phase 2 Milestone](phase2_milestone.png)

---

## Phase 2 — Complete Test Results

```
DOQAOALandscapeOverlapTest.mlir      5/5   PASS
DOQAOAEnergyEvalTest.mlir            5/5   PASS
DOQAOAPhaseTransitionTest.mlir       5/5   PASS  (ATTR + WARN)
DOQAOAClusterTest.mlir               5/5   PASS  (ATTR)
DOQAOABiasShiftTest.mlir            4×3 = 12/12 PASS  (ATTR + BASIN + GUARD)
Milestone: 10/10 ER graphs r > 0.999  PASS
─────────────────────────────────────────────────
Phase 2 total                       32/32 + 10/10 milestone  PASS
```

---

---

## Phase 3 — Parameter Transfer Pass

**Status:** IN PROGRESS | **Started:** Apr 6, 2026

### Task 1 — RepresentativeSelectionPass ✅

**Pass:** `--doqaoa-representative-selection` | **File:** `RepresentativeSelectionPass.cpp`

For each sub-problem k in every `freeze_partition` op (requires bias-shift annotations):
- `k == representatives[cluster]` → `is_representative[k]=1`, `transfer_modes[k]=0` (full optimisation)
- `bias_shifts[k] < threshold` → `transfer_modes[k]=1` (direct copy, zero cost)
- `bias_shifts[k] >= threshold` → `transfer_modes[k]=2` (warm start)

**New pass option:** `--bias-threshold=0.3`

| Attribute | Type | Meaning |
|---|---|---|
| `is_representative` | `array<i32, 2^m>` | 1 for cluster reps |
| `transfer_modes` | `array<i32, 2^m>` | 0=rep, 1=copy, 2=warm_start |
| `direct_copy_count` | `i32` | total sub-problems eligible for copy |
| `warm_start_count` | `i32` | total sub-problems needing warm start |

**Tests:** `DOQAOARepresentativeSelectionTest.mlir` — 5×3 = 15/15 PASS (ATTR + MODES + GUARD)

---

### Task 2 — TrainingScheduleRewriterPass ✅

**Pass:** `--doqaoa-training-schedule` | **File:** `TrainingScheduleRewriterPass.cpp`

Lowers 2^m sub-problems into a three-phase execution plan encoded as MLIR attributes:

| Phase | Mode | Sub-problems | Epoch budget |
|---|---|---|---|
| 1 — Full optimisation | 0 (representative) | K | `fullEpochs` (default 100) |
| 2 — Warm-start | 2 | non-reps with large ΔB | `warmstartEpochs` (default 20) |
| 3 — Direct transfer | 1 | non-reps with small ΔB | 0 (free) |

**New pass options:** `--full-epochs=100`, `--warmstart-epochs=20`

| Attribute | Type | Meaning |
|---|---|---|
| `training_schedule` | `array<i32, 2^m>` | sub-problem indices in execution order |
| `schedule_phase_ends` | `array<i32: 3>` | `[K, K+warm_count, 2^m]` — phase boundaries |
| `schedule_epochs` | `array<i32, 2^m>` | epoch budget per sub-problem (original k order) |
| `schedule_sources` | `array<i32, 2^m>` | cluster rep to copy params from (original k order) |

**Physics verified:**
- 4-cycle m=1 (K=1): schedule=[0,1], phase_ends=[1,1,2], epochs=[100,0], sources=[0,0]
- 4-cycle m=2 (K=2): schedule=[0,1,2,3], phase_ends=[2,2,4], epochs=[100,100,0,0]
- Sparse path m=2 (K=1): schedule=[0,1,2,3], phase_ends=[1,1,4], epochs=[100,0,0,0]
- threshold=0.0 (all warm-start): 4-cycle m=1 → phase_ends=[1,2,2], epochs=[200,30]

**Tests:** `DOQAOATrainingScheduleTest.mlir` — 5×3 = 15/15 PASS (SCHED + WARM + GUARD)

---

## Upcoming

| Phase | Task | Status |
|---|---|---|
| **Phase 3** | Task 3 — WarmStartOptimiser | Next |
| **Phase 4** | Frontend API & Integration | Planned |
| **Phase 5** | Benchmarking | Planned |
