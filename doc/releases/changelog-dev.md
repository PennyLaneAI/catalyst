# Release 0.15.0 (development release)

<h3>New features since last release</h3>

<h3>Improvements 🛠</h3>

<h3>Breaking changes 💔</h3>

<h3>Deprecations 👋</h3>

<h3>Bug fixes 🐛</h3>

<h3>Internal changes ⚙️</h3>

* Added LLVM conversion patterns to lower QEC dialect operations to their corresponding runtime
  CAPI calls. 
  This includes `qec.ppr` and `qec.ppr.arbitrary` (lowered to `__catalyst__qis__PauliRot`),
  `qec.ppm` (lowered to `__catalyst__qis__PauliMeasure`). This enables device execution of QEC
  operations through the Catalyst runtime.
  [(#XXXX)](https://github.com/PennyLaneAI/catalyst/pull/XXXX)

<h3>Documentation 📝</h3>

<h3>Contributors ✍️</h3>

This release contains contributions from (in alphabetical order):

Sengthai Heng,
