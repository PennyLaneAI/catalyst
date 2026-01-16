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

<h3>Deprecations 👋</h3>

<h3>Bug fixes 🐛</h3>

<h3>Internal changes ⚙️</h3>

* A new compiler pass `unroll-conditional-ppr-ppm` for lowering conditional PPR and PPMs
  into normal PPR and PPMs with SCF dialect to support runtime execution.
  [(#2390)](https://github.com/PennyLaneAI/catalyst/pull/2390)

* Added LLVM conversion patterns to lower QEC dialect operations to their corresponding runtime
  CAPI calls. 
  This includes `qec.ppr` and `qec.ppr.arbitrary` (lowered to `__catalyst__qis__PauliRot`),
  `qec.ppm` (lowered to `__catalyst__qis__PauliMeasure`). This enables device execution of QEC
  operations through the Catalyst runtime.
  [(#2389)](https://github.com/PennyLaneAI/catalyst/pull/2389)

* The quantum dialect MLIR and TableGen source has been refactored to place type and attribute
  definitions in separate file scopes.
  [(#2329)](https://github.com/PennyLaneAI/catalyst/pull/2329)

<h3>Documentation 📝</h3>

<h3>Contributors ✍️</h3>

This release contains contributions from (in alphabetical order):

Sengthai Heng,
Jeffrey Kam