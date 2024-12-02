# Release 0.10.0-dev (development release)

<h3>New features since last release</h3>

<h3>Improvements 🛠</h3>

* Replace pybind11 with nanobind for C++/Python bindings in the frontend and in the runtime.
  [(#1173)](https://github.com/PennyLaneAI/catalyst/pull/1173)
  [(#1293)](https://github.com/PennyLaneAI/catalyst/pull/1293)

  Nanobind has been developed as a natural successor to the pybind11 library and offers a number of
  [advantages](https://nanobind.readthedocs.io/en/latest/why.html#major-additions), in particular,
  its ability to target Python's [stable ABI interface](https://docs.python.org/3/c-api/stable.html)
  starting with Python 3.12.

* Catalyst now uses the new compiler API to compile quantum code from the python frontend.
  Frontend no longer uses pybind11 to connect to the compiler. Instead, it uses subprocess instead.
  [(#1285)](https://github.com/PennyLaneAI/catalyst/pull/1285)

* Add a MLIR decomposition for the gate set {"T", "S", "Z", "Hadamard", "RZ", "PhaseShift", "CNOT"}
  to the gate set {RX, RY, MS}. It is useful for trapped ion devices. It can be used thanks to
  `quantum-opt --ions-decomposition`.
  [(#1226)](https://github.com/PennyLaneAI/catalyst/pull/1226)

* qml.CosineWindow is now compatible with QJIT.
  [(#1166)](https://github.com/PennyLaneAI/catalyst/pull/1166)

* All PennyLane templates are tested for QJIT compatibility.
  [(#1161)](https://github.com/PennyLaneAI/catalyst/pull/1161)

* Decouple Python from the Runtime by using the Python Global Interpreter Lock (GIL) instead of
  custom mutexes.
  [(#624)](https://github.com/PennyLaneAI/catalyst/pull/624)

  In addition, executables created using :func:`~.debug.compile_executable` no longer require
  linking against Python shared libraries after decoupling Python from the Runtime C-API.
  [(#1305)](https://github.com/PennyLaneAI/catalyst/pull/1305)

* Improves the readability of conditional passes in pipelines
  [(#1194)](https://github.com/PennyLaneAI/catalyst/pull/1194)

<h3>Breaking changes 💔</h3>

* The `toml` module has been migrated to PennyLane with an updated schema for declaring device
  capabilities. Devices with TOML files using `schema = 2` will not be compatible with the latest
  Catalyst. See [Custom Devices](https://docs.pennylane.ai/projects/catalyst/en/stable/dev/custom_devices.html)
  for updated instructions on integrating your device with Catalyst and PennyLane
  [(#1275)](https://github.com/PennyLaneAI/catalyst/pull/1275)

* Handling for the legacy operator arithmetic (the `Hamiltonian` and `Tensor` classes in PennyLane)
  is removed.
  [(#1308)](https://github.com/PennyLaneAI/catalyst/pull/1308)

<h3>Bug fixes 🐛</h3>

* Fix bug introduced in 0.8 that breaks nested invocations of `qml.adjoint` and `qml.ctrl`.
  [(#1301)](https://github.com/PennyLaneAI/catalyst/issues/1301)

* Fix a bug in `debug.compile_executable` which would generate incorrect stride information for
  array arguments of the function, in particular when non-64bit datatypes are used.
  [(#1338)](https://github.com/PennyLaneAI/catalyst/pull/1338)

<h3>Deprecations 👋</h3>

<h3>Internal changes ⚙️</h3>

* Catalyst no longer depends on or pins the `scipy` package, instead OpenBLAS is sourced directly
  from [`scipy-openblas32`](https://pypi.org/project/scipy-openblas32/) or Accelerate is used.
  [(#1322)](https://github.com/PennyLaneAI/catalyst/pull/1322)
  [(#1328)](https://github.com/PennyLaneAI/catalyst/pull/1328)

* The `QuantumExtension` module (previously implemented with pybind11) has been removed. This module
  was not included in the distributed wheels and has been deprecated to align with our adoption of
  Python's stable ABI, which pybind11 does not support.
  [(#1187)](https://github.com/PennyLaneAI/catalyst/pull/1187)

* Remove Lightning Qubit Dynamic plugin from Catalyst.
  [(#1227)](https://github.com/PennyLaneAI/catalyst/pull/1227)
  [(#1307)](https://github.com/PennyLaneAI/catalyst/pull/1307)
  [(#1312)](https://github.com/PennyLaneAI/catalyst/pull/1312)

* `catalyst-cli` and `quantum-opt` are compiled with `default` visibility, which allows for MLIR plugins to work.
  [(#1287)](https://github.com/PennyLaneAI/catalyst/pull/1287)

* Sink patching of autograph's allowlist.
  [(#1332)](https://github.com/PennyLaneAI/catalyst/pull/1332)
  [(#1337)](https://github.com/PennyLaneAI/catalyst/pull/1337)

* The `sample` and `counts` measurement primitives now support dynamic shot values across catalyst,
  although at the PennyLane side, the device shots still is constrained to a static integer literal.

  To support this, `SampleOp` and `CountsOp` in mlir no longer carry the shots attribute, since integer attributes are tied to literal values and must be static.
  `DeviceInitOp` now takes in an optional SSA argument for shots, which sample and counts operations will retrieve when converting to runtime CAPI calls.

  [(#1170)](https://github.com/PennyLaneAI/catalyst/pull/1170)
  [(#1310)](https://github.com/PennyLaneAI/catalyst/pull/1310)

<h3>Documentation 📝</h3>

* A new tutorial going through how to write a new MLIR pass is available. The tutorial writes an
  empty pass that prints hello world. The code of the tutorial is at
  [a separate github branch](https://github.com/PennyLaneAI/catalyst/commit/ba7b3438667963b307c07440acd6d7082f1960f3).
  [(#872)](https://github.com/PennyLaneAI/catalyst/pull/872)

<h3>Contributors ✍️</h3>

This release contains contributions from (in alphabetical order):

Astral Cai,
Joey Carter,
David Ittah,
Erick Ochoa Lopez,
Mehrdad Malekmohammadi,
William Maxwell
Romain Moyard,
Raul Torres,
Paul Haochen Wang.
