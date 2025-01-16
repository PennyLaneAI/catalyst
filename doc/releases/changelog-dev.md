# Release 0.11.0-dev

<h3>New features since last release</h3>

<h3>Improvements 🛠</h3>

* The pattern rewriting in the `quantum-to-ion` lowering pass has been updated to use MLIR's dialect
  conversion infrastructure.
  [(#1442)](https://github.com/PennyLaneAI/catalyst/pull/1442)

<h3>Breaking changes 💔</h3>

<h3>Deprecations 👋</h3>

<h3>Bug fixes 🐛</h3>

<h3>Internal changes ⚙️</h3>

* The `get_c_interface` method has been added to the OQD device, which enables retrieval of the C++
  implementation of the device from Python. This allows `qjit` to accept an instance of the device
  and connect to its runtime.
  [(#1420)](https://github.com/PennyLaneAI/catalyst/pull/1420)

* `from_plxpr` now uses the `qml.capture.PlxprInterpreter` class for reduced code duplication.
  [(#1398)](https://github.com/PennyLaneAI/catalyst/pull/1398)

* The error messages for invalid measurements in :func:`qml.adjoint() <pennylane.adjoint>` and
  :func:`qml.ctrl() <pennylane.ctrl>` regions have been improved.
  [(#1425)](https://github.com/PennyLaneAI/catalyst/pull/1425)

* To better align with the semantics of `**QubitResult()` functions like `getNonCtrlQubitResults()`,
  `ValueRange` return types have been replaced with `ResultRange` and `Value` return types with
  `OpResult`. This change ensures clearer intent and usage. The `matchAndRewrite` function of
  `ChainedUUadjOpRewritePattern` has also been improved by using `replaceAllUsesWith` instead of for
  loops.
  [(#1426)](https://github.com/PennyLaneAI/catalyst/pull/1426)

<h3>Documentation 📝</h3>

<h3>Contributors ✍️</h3>

This release contains contributions from (in alphabetical order):

Christina Lee,
Mehrdad Malekmohammadi,
Sengthai Heng
