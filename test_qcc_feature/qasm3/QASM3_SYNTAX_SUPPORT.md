# OpenQASM 3.0 Syntax Support in QCC — Dynamic-Circuit Priorities

Reference: [OpenQASM 3.0 specification](https://github.com/openqasm/openqasm)

## Frontend: hybrid QASM3 loader (`qasm3_frontend.load_qasm3`)

qiskit's QASM3 importer (`qiskit_qasm3_import`) supports only a subset of the
grammar (no `const`, `def`, `uint`, casts, arrays, `let`, gate modifiers).
`load_qasm3()` in [qasm3_frontend.py](../../qasm3_frontend.py) tries qiskit
first (best dynamic-circuit support), then falls back to a partial evaluator
on the reference `openqasm3` parser (full grammar) that lowers static
constructs at compile time: const folding, classical types & casts,
`def` subroutine inlining (measured-bit returns bind to the assignment
target), user gate definitions, gate modifiers (`ctrl @`/`negctrl @`/`inv @`/
`pow(k) @`), `for` unrolling, `let` aliases, register broadcast, and
`input` values supplied via `load_qasm3(..., inputs={...})`.

**Official spec example coverage: 11/21** translate end-to-end
(adder, cphase, inverseqft1, inverseqft2, qec, qft, qpt, rb, rus, teleport,
varteleport). The other 10 live in
`openqasm3_official_example/unsupported/` (see its README) and fail
*cleanly* with the blocker named:

| Reason | Files |
|---|---|
| `extern` functions (need a runtime) | gateteleport, scqec, vqe, (also t1) |
| Hardware timing (`stretch`/`duration`/`delay`/`box`) | alignment, dd, t1 |
| `defcal` pulse grammar | defcal |
| Runtime classical arithmetic on measurement results | ipe (angle accumulator feeds gate params) |
| Out-of-bounds bugs in the spec's own example text | arrays (`my_defined_uints[4]` on size 4), msd (`scratch[3]` on `qubit[3]`) |

No existing library covers these: PennyLane has no QASM3 import, Catalyst's
frontend only *generates* QASM, and Braket's OpenQASM interpreter is
simulation-oriented (no reset/extern/feedforward).

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

## P2 — Handled at compile time by the hybrid frontend

These constructs never reach the MLIR pipeline: `qasm3_frontend` evaluates
them while building the qiskit circuit (so the emitted QASM3 contains their
*effects*, not the constructs themselves):

| Syntax | How it is lowered |
|--------|-------------------|
| `const`, classical types `int[n]`/`uint[n]`/`float[n]`/`angle[n]`/`bool` | compile-time values; expressions, casts and math functions evaluated |
| Casts in conditions (`int[2](flags) == 1`) | lowered to register comparisons on the underlying creg |
| `def` subroutines | inlined; `return b` / `return measure q` binds to the caller's assignment target so measurements land in the right register |
| User `gate` definitions | built as qiskit gates and inlined by the importer (definition not re-emitted; loses abstraction, keeps semantics) |
| Gate modifiers `ctrl @`, `negctrl @`, `inv @`, `pow(k) @` | applied via qiskit `.control()`/`.inverse()`/`.power()` then decomposed |
| `for` over ranges / discrete sets, classical `while` | unrolled (cap 10000) |
| `let` aliasing, array types, slices, `sizeof` | compile-time evaluation |
| `input` declarations | values supplied via `load_qasm3(..., inputs={...})` |
| `gphase` | folded into qiskit `global_phase` |
| `bit[n] x = "..."` initializers before a runtime `while` | one loop iteration is peeled so the lost initializer cannot skip the loop |

## P2 — Not implemented (frontend raises a clear error)

| Syntax | Notes |
|--------|-------|
| `switch (c) { case 0: ... }` | Qiskit `switch_case`; would map to `scf.index_switch` |
| `break;` / `continue;` | no natural `scf` counterpart; needs restructuring |
| *Runtime* classical arithmetic on measurement results (`c <<= 1`, bool algebra on bits, measuring into `angle[n]`) | no representation in qiskit or the MLIR pipeline; blocks `ipe.qasm`, `msd.qasm` |
| `extern` functions | require a runtime implementation by definition; blocks `gateteleport`, `scqec`, `vqe`, `t1` |
| `output` variables | declared but treated as plain registers |
| Timing: `delay`, `duration`, `stretch`, `box`, `defcal`, `durationof` | hardware-level; out of compiler scope; blocks `alignment`, `dd`, `t1`, `defcal` |

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
- **Register-comparison reconstruction**: the translator rebuilds whole-
  register comparisons (`if (c == 2)`, `while (flags != 0)`) from the
  importer's bit-level AND-folds when all conjuncts test the same creg —
  matching spec style and keeping the output parseable by qiskit (whose
  importer rejects `&&`). Note qiskit's importer still rejects `!=` register
  comparisons; that form is validated against the reference `openqasm3`
  grammar instead.

## Test entry points

```bash
# MLIR-level lit tests (ResetBarrierTest, CRegMeasureTest, ConditionExprTest, WhileLoopTest, ...)
/home/ubuntu/catalyst/mlir/llvm-project/build/bin/llvm-lit mlir/build/test/OpenQASM

# Python end-to-end (TestDynamicCircuits covers all P0/P1 features + teleport acceptance)
python3 -m pytest test_qcc_feature/qasm3/test_qasm3_translation_pytest.py -v
```
