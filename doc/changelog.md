# Release 0.2.0-dev

<h3>New features</h3>

* Bring feature parity to the Lightning-Kokkos backend simulator.
  [#55](https://github.com/PennyLaneAI/catalyst/pull/55)

<h3>Improvements</h3>

* Build the runtime against qir-stdlib v0.1.0.
  [#58](https://github.com/PennyLaneAI/catalyst/pull/58)

* Replace input-checking assertions with exceptions.
  [#67](https://github.com/PennyLaneAI/catalyst/pull/67)

* Lift heap allocations for quantum op results from the runtime into the MLIR compiler core. This
  allows all memref buffers to be memory managed in MLIR using the
  [MLIR bufferization infrastructure](https://mlir.llvm.org/docs/Bufferization/).
  [#63](https://github.com/PennyLaneAI/catalyst/pull/63)

<h3>Breaking changes</h3>

<h3>Bug fixes</h3>

* Fix memory leaks from data that flows back into the Python environment.
  [#54](https://github.com/PennyLaneAI/catalyst/pull/54)

* Fix memory leaks resulting from partial bufferization at the MLIR level. This fix makes the
  necessary changes to reintroduce the ``-buffer-deallocation`` pass into the MLIR pass pipeline.
  The pass guarantees that all allocations contained within a function (that is allocations that are
  not returned from a function) are also deallocated.

  This fixes a large majority of leaks in many typical quantum functions.
  [#61](https://github.com/PennyLaneAI/catalyst/pull/61)

<h3>Contributors</h3>

This release contains contributions from (in alphabetical order):

Ali Asadi,
Erick Ochoa Lopez,
Sergei Mironov.

# Release 0.1.1

<h3>New features</h3>

* Adds support for interpreting control flow operations.
  [#31](https://github.com/PennyLaneAI/catalyst/pull/31)

<h3>Improvements</h3>

* Adds fallback compiler drivers to increase reliability during linking phase. Also adds support for a
  CATALYST_CC environment variable for manual specification of the compiler driver used for linking.
  [#30](https://github.com/PennyLaneAI/catalyst/pull/30)

<h3>Breaking changes</h3>

<h3>Bug fixes</h3>

* Fixes the Catalyst image path in the readme to properly render on PyPI.

<h3>Contributors</h3>

This release contains contributions from (in alphabetical order):

Ali Asadi,
Erick Ochoa Lopez.

# Release 0.1.0

Initial public release.

<h3>Contributors</h3>

This release contains contributions from (in alphabetical order):

Ali Asadi,
Sam Banning,
David Ittah,
Josh Izaac,
Erick Ochoa Lopez,
Sergei Mironov,
Isidor Schoch.
