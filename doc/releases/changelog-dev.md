# Release 0.10.0-dev (development release)

<h3>New features since last release</h3>

<h3>Improvements 🛠</h3>

* Catalyst now uses the new compiler API to compile quantum code from the python frontend.
  Frontend no longer uses pybind11 to connect to the compiler. Instead, it uses subprocess instead.
  [(#1285)](https://github.com/PennyLaneAI/catalyst/pull/1285)

* Replace pybind11 with nanobind for C++/Python bindings in the frontend.
  [(#1173)](https://github.com/PennyLaneAI/catalyst/pull/1173)

  Nanobind has been developed as a natural successor to the pybind11 library and offers a number of
  [advantages](https://nanobind.readthedocs.io/en/latest/why.html#major-additions), in particular,
  its ability to target Python's [stable ABI interface](https://docs.python.org/3/c-api/stable.html)
  starting with Python 3.12.

* Add a MLIR decomposition for the gate set {"T", "S", "Z", "Hadamard", "RZ", "PhaseShift", "CNOT"} to
  the gate set {RX, RY, MS}. It is useful for trapped ion devices. It can be used thanks to
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

<h3>Deprecations 👋</h3>

<h3>Internal changes ⚙️</h3>

* The `QuantumExtension` module (previously implemented with pybind11) has been removed. This module
  was not included in the distributed wheels and has been deprecated to align with our adoption of
  Python's stable ABI, which pybind11 does not support.
  [(#1187)](https://github.com/PennyLaneAI/catalyst/pull/1187)

* Remove Lightning Qubit Dynamic plugin from Catalyst.
  [(#1227)](https://github.com/PennyLaneAI/catalyst/pull/1227)
  [(#1307)](https://github.com/PennyLaneAI/catalyst/pull/1307)
  [(#1312)](https://github.com/PennyLaneAI/catalyst/pull/1312)

<h3>Documentation 📝</h3>

* A new tutorial going through how to write a new MLIR pass is available. The tutorial writes an empty pass that prints hello world. The code of the tutorial is at [a separate github branch](https://github.com/PennyLaneAI/catalyst/commit/ba7b3438667963b307c07440acd6d7082f1960f3).
  [(#872)](https://github.com/PennyLaneAI/catalyst/pull/872)

<h3>Bug fixes 🐛</h3>

<h3>Contributors ✍️</h3>

This release contains contributions from (in alphabetical order):

Joey Carter,
Erick Ochoa Lopez,
Mehrdad Malekmohammadi,
William Maxwell
Romain Moyard,
Raul Torres,
Paul Haochen Wang.
