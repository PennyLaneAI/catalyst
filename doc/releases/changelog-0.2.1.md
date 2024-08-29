# Release 0.2.1

<h3>Bug fixes</h3>

* Add missing OpenQASM backend in binary distribution, which relies on the latest version of the
  AWS Braket plugin for PennyLane to resolve dependency issues between the plugin, Catalyst, and
  PennyLane. The Lightning-Kokkos backend with Serial and OpenMP modes is also added to the binary
  distribution.
  [#198](https://github.com/PennyLaneAI/catalyst/pull/198)

* Return a list of decompositions when calling the decomposition method for control operations.
  This allows Catalyst to be compatible with upstream PennyLane.
  [#241](https://github.com/PennyLaneAI/catalyst/pull/241)

<h3>Improvements</h3>

* When using OpenQASM-based devices the string representation of the circuit is printed on
  exception.
  [#199](https://github.com/PennyLaneAI/catalyst/pull/199)

* Use ``pybind11::module`` interface library instead of ``pybind11::embed`` in the runtime for
  OpenQasm backend to avoid linking to the python library at compile time.
  [#200](https://github.com/PennyLaneAI/catalyst/pull/200)

<h3>Contributors</h3>

This release contains contributions from (in alphabetical order):

Ali Asadi,
David Ittah.
