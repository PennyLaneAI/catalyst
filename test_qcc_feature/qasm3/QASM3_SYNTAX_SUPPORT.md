# OpenQASM 3.0 Syntax Support in QCC — Dynamic-Circuit Priorities

Reference: [OpenQASM 3.0 specification](https://github.com/openqasm/openqasm)

This document lists the OpenQASM 3.0 syntax essential for **dynamic circuits**
(circuits where classical measurement results influence subsequent quantum
operations), assigns priorities, and records the implementation status in the
QCC pipeline:

```
Qiskit QuantumCircuit → qiskit_importer_standalone.py → Catalyst MLIR
    → quantum-opt (canonicalize) → quantum-translate --mlir-to-qasm3 → OpenQASM 3.0
```

Priorities: **P0** = core dynamic-circuit semantics (feedforward), **P1** =
needed for the standard dynamic-circuit patterns in the official spec examples
(teleport, QEC, repeat-until-success), **P2** = advanced/rare constructs.

## P0 — Essential core (implemented ✅)

| Syntax | Example | Status | Representation in MLIR |
|--------|---------|--------|------------------------|
| Mid-circuit measurement | `c[0] = measure q[0];` | ✅ | `quantum.measure` → (`i1`, `!quantum.bit`) |
| Classical register declaration | `bit[n] c;` | ✅ | `creg_name`/`creg_idx`/`creg_size` attributes on `quantum.measure`; declarations are pre-scanned and emitted at the top of `main` |
| Feedforward if/else | `if (c[0]) { x q[1]; } else { z q[1]; }` | ✅ | `scf.if` on the measurement `i1` |
| Condition expressions | `!c[0]`, `c[0] && c[1]`, `c[0] \|\| c[1]`, `c[0] == 1` | ✅ | `arith.xori` / `arith.andi` / `arith.ori` / `arith.cmpi eq\|ne`, translated recursively by `buildCondExpr` |
| Reset | `reset q[0];` | ✅ | `quantum.custom {gate_name="reset"}` marker |
| Anonymous bit fallback | `bit m_1; m_1 = measure q[0];` | ✅ | measurement without creg attributes (loose Qiskit clbits) |

## P1 — Standard dynamic-circuit patterns (implemented ✅)

| Syntax | Example | Status | Representation in MLIR |
|--------|---------|--------|------------------------|
| While loop (repeat-until-success) | `while (c[0]) { h q[0]; c[0] = measure q[0]; }` | ✅ | `scf.while`; loop-carried values are `[cond bits...] + [qubits...]`; the before region recomputes the boolean and forwards args via `scf.condition` |
| Multi-bit register condition | `if ((!c[0] && c[1])) ...` for `c == 2` | ✅ | importer AND-folds per-bit tests into one `i1` feeding a single `scf.if` |
| Barrier | `barrier q[0], q[1];` | ✅ | `quantum.custom {gate_name="barrier"}` marker |
| For loop (constant bounds) | `for int i in [0:1:5] { ... }` | ✅ (pre-existing) | `scf.for` with qubit iter_args |
| Builtin `U` gate | `U(θ, φ, λ) q[0];` | ✅ | Qiskit `u` → gate name mapped to builtin `U` (plain `u` is not in stdgates.inc) |
| User-defined gate inlining | `gate post q { }` | ✅ (inlined) | importer inlines the parameter-bound Qiskit `.definition` body; empty bodies vanish (identity). The `gate` definition itself is not re-emitted (see P2) |

**Acceptance:** the official spec example `openqasm3_official_example/teleport.qasm`
(reset, barrier, `U`, two feedforward ifs, custom identity gate) round-trips
through the full pipeline and re-parses as valid QASM3
(`TestDynamicCircuits::test_official_teleport_roundtrip`).

## P2 — Not implemented (documented for future work)

| Syntax | Notes |
|--------|-------|
| `switch (c) { case 0: ... }` | Qiskit `switch_case`; would map to `scf.index_switch` |
| `break;` / `continue;` | no natural `scf` counterpart; needs restructuring |
| Classical types `int[n]`, `uint[n]`, `float[n]`, `angle[n]`, `bool`, `complex` | measurement results are raw `i1` SSA values; no classical storage model |
| Classical arithmetic / casts (`int[2](flags)`, `a += b`) | blocks `qec.qasm`, `rus.qasm`, `adder.qasm` |
| `def` subroutines with typed params/returns | translator emits only a stub `def name() { ... }` for non-main funcs |
| Re-emitting user `gate` definitions | definitions are inlined instead (semantically equivalent, loses abstraction) |
| Gate modifiers `ctrl @`, `negctrl @`, `inv @`, `pow(k) @` | importer flattens controlled ops to named gates (`cx`, ...) |
| `input` / `output` variables | |
| Timing: `delay`, `duration`, `stretch`, `box`, `defcal`, `durationof` | hardware-level; out of compiler scope for now |
| `let` aliasing, array views | |
| `gphase` | `quantum.gphase` exists in the dialect but is not translated |
| `while` on classical expressions other than register-bit tests | condition must be derived from measured bits |

## Implementation notes & caveats

- **reset/barrier markers**: represented as `quantum.custom` ops so no dialect
  change is needed. Verified to survive
  `apply-transform-sequence, canonicalize, merge-rotations` as long as their
  results are consumed by later ops. A *trailing* reset/barrier (nothing after
  it on that qubit) is dead code under `canonicalize` (CustomOp is
  `NoMemoryEffect`) and gets dropped — harmless for observable behavior. The
  hardening path is a dedicated side-effecting `quantum.reset` op in
  `QuantumOps.td`.
- **creg attributes**: `quantum.measure` is side-effecting and has no
  canonicalizer, so the `creg_*` attributes survive the pass pipeline. If a
  future pass drops them, the translator falls back to anonymous `bit m_N`
  (still-correct output), except inside `while` bodies where the loop-carry
  depends on the register name — the translator prints a warning in that case.
- **scf.while arg pruning**: `canonicalize` prunes unused values forwarded by
  `scf.condition`, so after-region block args must be mapped from
  `scf.condition`'s forwarded operands, never positionally from the loop
  inits. `WhileLoopTest.mlir` covers both the full-forwarding and pruned forms.
- **`if (c0 == 1)` vs `== true`**: the spec's own teleport example compares a
  single `bit` to `1`, but `qiskit_qasm3_import` only accepts `== true` for
  single bits. Our emitted conditions use bare `c[0]` / `!c[0]` forms, which
  both accept.

## Test entry points

```bash
# MLIR-level lit tests (ResetBarrierTest, CRegMeasureTest, ConditionExprTest, WhileLoopTest, ...)
/home/ubuntu/catalyst/mlir/llvm-project/build/bin/llvm-lit mlir/build/test/OpenQASM

# Python end-to-end (TestDynamicCircuits covers all P0/P1 features + teleport acceptance)
python3 -m pytest test_qcc_feature/qasm3/test_qasm3_translation_pytest.py -v
```
