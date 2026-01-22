# Release 0.15.0 (development release)

<h3>New features since last release</h3>

<h3>Improvements 🛠</h3>

* The default mcm_method for the finite-shots setting (dynamic one-shot) no longer silently falls
  back to single-branch statistics in most cases. Instead, an error message is raised pointing out
  alternatives, like explicitly selecting single-branch statistics.
  [(#2398)](https://github.com/PennyLaneAI/catalyst/pull/2398)

  Importantly, single-branch statistics only explores one branch of the MCM decision tree, meaning
  program outputs are typically probabilistic and statistics produced by measurement processes are
  conditional on the selected decision tree path.

* Two new verifiers were added to the `quantum.paulirot` operation. They verify that the Pauli word
  length and the number of qubit operands are the same, and that all of the Pauli words are legal.
  [(#2405)](https://github.com/PennyLaneAI/catalyst/pull/2405)

<h3>Breaking changes 💔</h3>

* Dropped support for NumPy 1.x following its end-of-life. NumPy 2.0 or higher is now required.
  [(#2407)](https://github.com/PennyLaneAI/catalyst/pull/2407)

<h3>Deprecations 👋</h3>

<h3>Bug fixes 🐛</h3>

* Fixed the angle conversion when lowering `qec.ppr` and `qec.ppr.arbitrary` operations to
  `__catalyst__qis__PauliRot` runtime calls. The PPR rotation angle is now correctly multiplied
  by 2 to match the PauliRot convention (`PauliRot(φ) == PPR(φ/2)`).
  [(#2414)](https://github.com/PennyLaneAI/catalyst/pull/2414)

<h3>Internal changes ⚙️</h3>

* The quantum dialect MLIR and TableGen source has been refactored to place type and attribute
  definitions in separate file scopes.
  [(#2329)](https://github.com/PennyLaneAI/catalyst/pull/2329)

* Added lowering of `qec.ppm`, `qec.ppr`, and `quantum.paulirot` to the runtime CAPI and QuantumDevice C++ API.
  [(#2348)](https://github.com/PennyLaneAI/catalyst/pull/2348)
  [(#2413)](https://github.com/PennyLaneAI/catalyst/pull/2413)

* Added LLVM conversion patterns to lower QEC dialect operations to their corresponding runtime
  CAPI calls.
  This includes `qec.ppr` and `qec.ppr.arbitrary` (lowered to `__catalyst__qis__PauliRot`),
  `qec.ppm` (lowered to `__catalyst__qis__PauliMeasure`). This enables device execution of QEC
  operations through the Catalyst runtime.
  [(#2389)](https://github.com/PennyLaneAI/catalyst/pull/2389)

* A new compiler pass `unroll-conditional-ppr-ppm` for lowering conditional PPR and PPMs
  into normal PPR and PPMs with SCF dialect to support runtime execution.
  [(#2390)](https://github.com/PennyLaneAI/catalyst/pull/2390)

* New qubit-type specializations have been added to Catalyst's MLIR type system. These new qubit
  types include `!quantum.bit<logical>`, `!quantum.bit<qec>` and `!quantum.bit<physical>`. The
  original `!quantum.bit` type continues to be supported and used as the default qubit type.
  [(#2369)](https://github.com/PennyLaneAI/catalyst/pull/2369)

* The upstream MLIR `Test` dialect is now available via the `catalyst` command line tool.
  [(#2417)](https://github.com/PennyLaneAI/catalyst/pull/2417)

<h3>Documentation 📝</h3>

* Updated the Unified Compiler Cookbook to be compatible with the latest versions of PennyLane and Catalyst.
  [(#2406)](https://github.com/PennyLaneAI/catalyst/pull/2406)

* Updated the changelog and builtin_passes.py to link to https://pennylane.ai/compilation/pauli-based-computation instead.
  [(#2409)](https://github.com/PennyLaneAI/catalyst/pull/2409)

<h3>Contributors ✍️</h3>

This release contains contributions from (in alphabetical order):
Ali Asadi,
Joey Carter,
Sengthai Heng,
Jeffrey Kam,
Mudit Pandey,
Andrija Paurevic,
David D.W. Ren
Paul Haochen Wang.
