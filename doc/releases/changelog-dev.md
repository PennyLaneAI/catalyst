# Release 0.11.0 (development release)

<h3>New features since last release</h3>

<h3>Improvements 🛠</h3>

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

* Replace `ValueRange` with `ResultRange` and `Value` with `OpResult` to better align with the semantics of `**QubitResult()` functions like `getNonCtrlQubitResults()`. This change ensures clearer intent and usage. Improve the `matchAndRewrite` function by using `replaceAllUsesWith` instead of for loop.
  [(#1426)](https://github.com/PennyLaneAI/catalyst/pull/1426)

<h3>Documentation 📝</h3>

<h3>Contributors ✍️</h3>

This release contains contributions from (in alphabetical order):

Christina Lee
Mehrdad Malekmohammadi
Sengthai Heng
