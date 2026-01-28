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

<h3>Breaking changes 💔</h3>

* (Compiler integrators only) The versions of StableHLO/LLVM/Enzyme used by Catalyst have been updated.
  [(#2415)](https://github.com/PennyLaneAI/catalyst/pull/2415)
  [(#2416)](https://github.com/PennyLaneAI/catalyst/pull/2416)
  [(#2444)](https://github.com/PennyLaneAI/catalyst/pull/2444)

  - The StableHLO version has been updated to
  [v1.13.7](https://github.com/openxla/stablehlo/tree/v1.13.7).
  - The LLVM version has been updated to
  [commit 8f26458](https://github.com/llvm/llvm-project/tree/8f264586d7521b0e305ca7bb78825aa3382ffef7).
  - The Enzyme version has been updated to
  [v0.0.238](https://github.com/EnzymeAD/Enzyme/releases/tag/v0.0.238).

<h3>Deprecations 👋</h3>

<h3>Bug fixes 🐛</h3>

<h3>Internal changes ⚙️</h3>

* The quantum dialect MLIR and TableGen source has been refactored to place type and attribute
  definitions in separate file scopes.
  [(#2329)](https://github.com/PennyLaneAI/catalyst/pull/2329)

* Added `PauliMeasure` and `PauliRot` to the runtime CAPI and QuantumDevice C++ API.
  [(#2348)](https://github.com/PennyLaneAI/catalyst/pull/2348)

* Added LLVM conversion patterns to lower QEC dialect operations to their corresponding runtime
  CAPI calls.
  This includes `qec.ppr` and `qec.ppr.arbitrary` (lowered to `__catalyst__qis__PauliRot`),
  `qec.ppm` (lowered to `__catalyst__qis__PauliMeasure`). This enables device execution of QEC
  operations through the Catalyst runtime.
  [(#2389)](https://github.com/PennyLaneAI/catalyst/pull/2389)

* A new compiler pass `unroll-conditional-ppr-ppm` for lowering conditional PPR and PPMs
  into normal PPR and PPMs with SCF dialect to support runtime execution.
  [(#2390)](https://github.com/PennyLaneAI/catalyst/pull/2390)

<h3>Documentation 📝</h3>

* Updated the Unified Compiler Cookbook to be compatible with the latest versions of PennyLane and Catalyst.
  [(#2406)](https://github.com/PennyLaneAI/catalyst/pull/2406)

<h3>Contributors ✍️</h3>

This release contains contributions from (in alphabetical order):
Ali Asadi
Sengthai Heng,
Jeffrey Kam,
Mehrdad Malekmohammadi,
Mudit Pandey.
