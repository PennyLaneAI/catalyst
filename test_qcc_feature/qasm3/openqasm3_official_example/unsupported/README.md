# Unsupported Official OpenQASM 3 Examples

These files from the [official OpenQASM 3 specification examples](https://github.com/openqasm/openqasm)
are **grammatically valid** (the reference `openqasm3` parser accepts all of
them) but cannot be tested through the QCC pipeline, because they use
constructs that have **no concrete circuit representation** — no
circuit-building frontend (qiskit, Braket, or our `qasm3_frontend` AST
evaluator) can turn them into an executable `QuantumCircuit`.

They are kept out of the parent folder so batch runs
(`python qcc_simulator.py --batch openqasm3_official_example/`) contain only
testable circuits. Running one of these files individually reports
**SKIP** with the reason below (never a raw parse error).

| File | Blocker | Detail |
|------|---------|--------|
| `gateteleport.qasm` | `extern` | `extern vote(...)` — external classical function with no body; requires a runtime implementation by definition |
| `scqec.qasm` | `extern` | `extern zfirst(...)` / `send(...)` — real-time decoder callbacks |
| `vqe.qasm` | `extern` | `extern get_parameter(...)` — runtime parameter fetch from a classical optimizer |
| `t1.qasm` | timing + `extern` | `duration`, `durationof`, `delay`, stretch-based scheduling |
| `alignment.qasm` | timing | `stretch`, `delay` — hardware scheduling, not gate semantics |
| `dd.qasm` | timing | `stretch`, `duration`, `box`, `durationof` — dynamical-decoupling scheduling |
| `defcal.qasm` | pulse grammar | `defcalgrammar "openpulse"` / `defcal` — calibration-level definitions |
| `ipe.qasm` | runtime classical arithmetic | measurement results accumulate into an `angle[n]` (`c <<= 1`) that feeds gate parameters (`inv @ phase(c)`); gate parameters computed from measurements at runtime cannot be represented in a static circuit |
| `arrays.qasm` | bug in the example itself | classical-only demo; writes out of bounds: `my_defined_uints[a - 1]` with `a = 5` on a size-4 array |
| `msd.qasm` | bug in the example itself + runtime classical arithmetic | `cz scratch[3], scratch[2]` on a `qubit[3]` parameter (indices 0–2); also computes a `bool` from measured bits at runtime |

Notes:

- "extern", timing, and pulse constructs describe **hardware/runtime
  behavior**, not circuits — supporting them is out of scope for a
  translation pipeline and would be equally impossible for any other
  circuit-level frontend.
- The two "bug in the example" files are genuinely erroneous as executable
  programs in the upstream spec repository; our frontend reports
  `Index out of bounds` instead of crashing.
- The pytest suite (`test_qasm3_translation_pytest.py::TestHybridFrontend::
  test_official_example_unsupported_reason`) locks in that each of these
  files fails **cleanly** with the reason named above.
- See `../../QASM3_SYNTAX_SUPPORT.md` for the full syntax-support matrix.
