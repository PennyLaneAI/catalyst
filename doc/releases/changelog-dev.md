:orphan:

# Release 0.10.0-dev (development release)

<h3>New features since last release</h3>

<h3>Improvements 🛠</h3>

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

<h3>Breaking changes 💔</h3>

<h3>Deprecations 👋</h3>

<h3>Documentation 📝</h3>

<h3>Bug fixes 🐛</h3>

<h3>Contributors ✍️</h3>

This release contains contributions from (in alphabetical order):

Joey Carter,
Romain Moyard.