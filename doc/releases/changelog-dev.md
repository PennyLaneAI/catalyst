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

<h3>Breaking changes 💔</h3>

<h3>Deprecations 👋</h3>

<h3>Documentation 📝</h3>

* A new tutorial going through how to write a new MLIR pass is available. The tutorial writes an empty pass that prints hello world. The code of the tutorial is at [a separate github branch](https://github.com/PennyLaneAI/catalyst/commit/a857655b2f7afef6de19cdc1faaa226243e0bb58).
  [(#872)](https://github.com/PennyLaneAI/catalyst/pull/872)

<h3>Bug fixes 🐛</h3>

<h3>Contributors ✍️</h3>

This release contains contributions from (in alphabetical order):
Joey Carter,
Paul Haochen Wang.
