# Release 0.2.0-dev

<h3>New features</h3>

* Add a Backprop operation with Bufferiation
  [#107](https://github.com/PennyLaneAI/catalyst/pull/107)

* Add support for ``else if`` chains for ``@cond`` conditionals
  [#104](https://github.com/PennyLaneAI/catalyst/pull/104)

* Add the end-to-end support for multiple backend devices. The compilation flag
  ``ENABLE_LIGHTNING_KOKKOS=ON`` builds the runtime with support for PennyLane's
  ``lightning.kokkos``. Both ``lightning.qubit`` and ``lightning.kokkos`` can be
  chosen as available backend devices from the frontend.
  [#89](https://github.com/PennyLaneAI/catalyst/pull/89)

* Add support for ``var`` of general observables
  [#124](https://github.com/PennyLaneAI/catalyst/pull/124)

<h3>Improvements</h3>

* Improving error handling by throwing descriptive and unified expressions for runtime
  errors and assertions.
  [#92](https://github.com/PennyLaneAI/catalyst/pull/92)

* Improve interface for adding and re-using flags to quantum-opt commands.
  These are called pipelines, as they contain multiple passes.
  [#38](https://github.com/PennyLaneAI/catalyst/pull/38)

* Improve python compatibility by providing a stable signature for user generated functions.
  [#106](https://github.com/PennyLaneAI/catalyst/pull/106)

* Handle C++ exceptions without unwinding the whole stack.
  [#99](https://github.com/PennyLaneAI/catalyst/pull/99)

* Support constant negative step sizes in ``@for_loop`` loops.
  [#129](https://github.com/PennyLaneAI/catalyst/pull/129)

<h3>Breaking changes</h3>

<h3>Bug fixes</h3>

* Fix a bug in the mapping from logical to concrete qubits for mid-circuit measurements.
  [#80](https://github.com/PennyLaneAI/catalyst/pull/80)

* Fix a bug in the way gradient result type is inferred.
  [#84](https://github.com/PennyLaneAI/catalyst/pull/84)

* Fix a memory regression and reduce memory footprint by removing unnecessary temporary buffers.
  [#100](https://github.com/PennyLaneAI/catalyst/pull/100)

* Provide a new abstraction to the ``QuantumDevice`` interface in the runtime called ``MemRefView``.
  C++ implementations of the interface can iterate through and directly store results into the
  ``MemRefView`` independant of the underlying memory layout. This can eliminate redundant buffer
  copies at the interface boundaries, which has been applied to existing devices.
  [#109](https://github.com/PennyLaneAI/catalyst/pull/109)

* Reduce memory utilization by transferring ownership of buffers from the runtime to Python instead
  of copying them. This includes adding a compiler pass that copies global buffers into the heap
  as global buffers cannot be transferred to Python.
  [#112](https://github.com/PennyLaneAI/catalyst/pull/112)

* Temporary fix of use-after-free and dependency of uninitialized memory.
  [#121](https://github.com/PennyLaneAI/catalyst/pull/121)

<h3>Contributors</h3>

This release contains contributions from (in alphabetical order):

Ali Asadi,
David Ittah,
Jacob Mai Peng,
Romain Moyard,
Erick Ochoa Lopez.

# Release 0.1.2

<h3>New features</h3>

* Add an option to print verbose messages explaining the compilation process.

  [#68](https://github.com/PennyLaneAI/catalyst/pull/68)

* Allow ``catalyst.grad`` to be used on any traceable function (within a qjit context).
  This means the operation is no longer resticted to acting on ``qml.qnode``s only.
  [#75](https://github.com/PennyLaneAI/catalyst/pull/75)


<h3>Improvements</h3>

* Work in progress on a Lightning-Kokkos backend:

  Bring feature parity to the Lightning-Kokkos backend simulator.
  [#55](https://github.com/PennyLaneAI/catalyst/pull/55)

  Add support for variance measurements for all observables.
  [#70](https://github.com/PennyLaneAI/catalyst/pull/70)

* Build the runtime against qir-stdlib v0.1.0.
  [#58](https://github.com/PennyLaneAI/catalyst/pull/58)

* Replace input-checking assertions with exceptions.
  [#67](https://github.com/PennyLaneAI/catalyst/pull/67)

* Perform function inlining to improve optimizations and memory management within the compiler.
  [#72](https://github.com/PennyLaneAI/catalyst/pull/72)

<h3>Breaking changes</h3>

<h3>Bug fixes</h3>

* Several fixes to address memory leaks in the compiled program:

  Fix memory leaks from data that flows back into the Python environment.
  [#54](https://github.com/PennyLaneAI/catalyst/pull/54)

  Fix memory leaks resulting from partial bufferization at the MLIR level. This fix makes the
  necessary changes to reintroduce the ``-buffer-deallocation`` pass into the MLIR pass pipeline.
  The pass guarantees that all allocations contained within a function (that is allocations that are
  not returned from a function) are also deallocated.
  [#61](https://github.com/PennyLaneAI/catalyst/pull/61)

  Lift heap allocations for quantum op results from the runtime into the MLIR compiler core. This
  allows all memref buffers to be memory managed in MLIR using the
  [MLIR bufferization infrastructure](https://mlir.llvm.org/docs/Bufferization/).
  [#63](https://github.com/PennyLaneAI/catalyst/pull/63)

  Eliminate all memory leaks by tracking memory allocations at runtime. The memory allocations
  which are still alive when the compiled function terminates, will be freed in the
  finalization / teardown function.
  [#78](https://github.com/PennyLaneAI/catalyst/pull/78)

* Fix returning complex scalars from the compiled function.
  [#77](https://github.com/PennyLaneAI/catalyst/pull/77)

<h3>Contributors</h3>

This release contains contributions from (in alphabetical order):

Ali Asadi,
David Ittah,
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
