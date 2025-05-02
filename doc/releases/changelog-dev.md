# Release 0.12.0 (development release)

<h3>New features since last release</h3>

<h3>Improvements 🛠</h3>

* The behaviour of measurement processes executed on `null.qubit` with QJIT is now more in line with
  their behaviour on `null.qubit` *without* QJIT.
  [(#1598)](https://github.com/PennyLaneAI/catalyst/pull/1598)

  Previously, measurement processes like `qml.sample()`, `qml.counts()`, `qml.probs()`, etc.
  returned values from uninitialized memory when executed on `null.qubit` with QJIT. This change
  ensures that measurement processes on `null.qubit` always return the value 0 or the result
  corresponding to the '0' state, depending on the context.

<h3>Breaking changes 💔</h3>

* (Device Developers Only) The `QuantumDevice` interface in the Catalyst Runtime plugin system
  has been modified, which requires recompiling plugins for binary compatibility.
  [(#1680)](https://github.com/PennyLaneAI/catalyst/pull/1680)

  As announced in the [0.10.0 release](https://docs.pennylane.ai/projects/catalyst/en/stable/dev/release_notes.html#release-0-10-0),
  the `shots` argument has been removed from the `Sample` and `Counts` methods in the interface,
  since it unnecessarily duplicated this information. Additionally, `shots` will no longer be
  supplied by Catalyst through the `kwargs` parameter of the device constructor. The shot value must
  now be obtained through the `SetDeviceShots` method.

  Further, the documentation for the interface has been overhauled and now describes the
  expected behaviour of each method in detail. A quality of life improvement is that optional
  methods are now clearly marked as such and also come with a default implementation in the base
  class, so device plugins need only override the methods they wish to support.

  Finally, the `PrintState` and the `One`/`Zero` utility functions have been removed, since they
  did not serve a convincing purpose.

* Catalyst has removed the `experimental_capture` keyword from the `qjit` decorator in favour of
  unified behaviour with PennyLane.
  [(#1657)](https://github.com/PennyLaneAI/catalyst/pull/1657)

  Instead of enabling program capture with Catalyst via `qjit(experimental_capture=True)`, program
  capture can be enabled via the global toggle `qml.capture.enable()`:

  ```python
  import pennylane as qml
  from catalyst import qjit

  dev = qml.device("lightning.qubit", wires=2)

  qml.capture.enable()

  @qjit
  @qml.qnode(dev)
  def circuit(x):
      qml.Hadamard(0)
      qml.CNOT([0, 1])
      return qml.expval(qml.Z(0))

  circuit(0.1)
  ```

  Disabling program capture can be done with `qml.capture.disable()`.

* The `ppr_to_ppm` pass has been renamed to `merge_ppr_ppm` (same functionality). A new `ppr_to_ppm`
  will handle direct decomposition of PPRs into PPMs.
  [(#1688)](https://github.com/PennyLaneAI/catalyst/pull/1688)

<h3>Deprecations 👋</h3>

<h3>Bug fixes 🐛</h3>

* Fix AutoGraph fallback for valid iteration targets with constant data but no length, for example
  `itertools.product(range(2), repeat=2)`.
  [(#1665)](https://github.com/PennyLaneAI/catalyst/pull/1665)

* Catalyst now correctly supports `qml.StatePrep()` and `qml.BasisState()` operations in the
  experimental PennyLane program-capture pipeline.
  [(#1631)](https://github.com/PennyLaneAI/catalyst/pull/1631)

<h3>Internal changes ⚙️</h3>

* Creates a function that allows developers to register an equivalent MLIR transform for a given
  PLxPR transform.
  [(#1705)](https://github.com/PennyLaneAI/catalyst/pull/1705)

* Stop overriding the `num_wires` property when the operator can exist on `AnyWires`. This allows
  the deprecation of `WiresEnum` in pennylane.
  [(#1667)](https://github.com/PennyLaneAI/catalyst/pull/1667)
  [(#1676)](https://github.com/PennyLaneAI/catalyst/pull/1676)

* Catalyst now includes an experimental `mbqc` dialect for representing measurement-based
  quantum-computing protocols in MLIR.
  [(#1663)](https://github.com/PennyLaneAI/catalyst/pull/1663)
  [(#1679)](https://github.com/PennyLaneAI/catalyst/pull/1679)

* The Catalyst Runtime C-API now includes a stub for the experimental `mbqc.measure_in_basis`
  operation, `__catalyst__mbqc__measure_in_basis()`, allowing for mock execution of MBQC workloads
  containing parameterized arbitrary-basis measurements.
  [(#1674)](https://github.com/PennyLaneAI/catalyst/pull/1674)

  This runtime stub is currently for mock execution only and should be treated as a placeholder
  operation. Internally, it functions just as a computational-basis measurement instruction.

* PennyLane's arbitrary-basis measurement operations, such as [`qml.ftqc.measure_arbitrary_basis()`
  ](https://docs.pennylane.ai/en/stable/code/api/pennylane.ftqc.measure_arbitrary_basis.html), are
  now QJIT-compatible with program capture enabled.
  [(#1645)](https://github.com/PennyLaneAI/catalyst/pull/1645)
  [(#1710)](https://github.com/PennyLaneAI/catalyst/pull/1710)

* The utility function `EnsureFunctionDeclaration` is refactored into the `Utils` of the `Catalyst`
  dialect, instead of being duplicated in each individual dialect.
  [(#1683)](https://github.com/PennyLaneAI/catalyst/pull/1683)

* The assembly format for some MLIR operations now includes adjoint.
  [(#1695)](https://github.com/PennyLaneAI/catalyst/pull/1695)

* Improved the definition of `YieldOp` in the quantum dialect by removing `AnyTypeOf`
  [(#1696)](https://github.com/PennyLaneAI/catalyst/pull/1696)

* The bufferization of custom catalyst dialects has been migrated to the new one-shot
  bufferization interface in mlir.
  The new mlir bufferization interface is required by jax 0.4.29 or higher.
  [(#1027)](https://github.com/PennyLaneAI/catalyst/pull/1027)
  [(#1686)](https://github.com/PennyLaneAI/catalyst/pull/1686)

<h3>Documentation 📝</h3>

<h3>Contributors ✍️</h3>

This release contains contributions from (in alphabetical order):

Joey Carter,
Sengthai Heng,
David Ittah,
Tzung-Han Juang,
Christina Lee,
Erick Ochoa Lopez,
Paul Haochen Wang.
