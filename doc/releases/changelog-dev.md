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

* The :func:`~.passes.to_ppr` pass now supports conversion of Pauli gates (`X`, `Y`, `Z`),
  the phase gate adjoint (`S†`), and the π/8 gate adjoint (`T†`). This extension improves
  performance by eliminating indirect conversion.
  [(#1738)](https://github.com/PennyLaneAI/catalyst/pull/1738)

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

* (Frontend Developers Only) Some Catalyst primitives for JAX have been renamed, and the qubit
  deallocation primitive has been split into deallocation and a separate device release primitive.
  [(#1720)](https://github.com/PennyLaneAI/catalyst/pull/1720)

  - `qunitary_p` is now `unitary_p` (unchanged)
  - `qmeasure_p` is now `measure_p` (unchanged)
  - `qdevice_p` is now `device_init_p` (unchanged)
  - `qdealloc_p` no longer releases the device, thus it can be used at any point of a quantum
     execution scope
  - `device_release_p` is a new primitive that must be used to mark the end of a quantum execution
     scope, which will release the quantum device

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

* The version of JAX used by Catalyst is updated to 0.6.0.
  [(#1652)](https://github.com/PennyLaneAI/catalyst/pull/1652)
  [(#1729)](https://github.com/PennyLaneAI/catalyst/pull/1729)

  Several internal changes were made for this update.
    - LAPACK kernels are updated to adhere to the new JAX lowering rules for external functions.
    [(#1685)](https://github.com/PennyLaneAI/catalyst/pull/1685)

    - The trace stack is removed and replaced with a tracing context manager.
    [(#1662)](https://github.com/PennyLaneAI/catalyst/pull/1662)

    - A new `debug_info` argument is added to `Jaxpr`, the `make_jaxpr`
    functions, and `jax.extend.linear_util.wrap_init`.
    [(#1670)](https://github.com/PennyLaneAI/catalyst/pull/1670)
    [(#1671)](https://github.com/PennyLaneAI/catalyst/pull/1671)
    [(#1681)](https://github.com/PennyLaneAI/catalyst/pull/1681)

* The clang-format and clang-tidy versions used by Catalyst have been updated to v20.
  [(#1721)](https://github.com/PennyLaneAI/catalyst/pull/1721)

* Support for Mac x86 has been removed. This includes Macs running on Intel processors.
  [(#1716)](https://github.com/PennyLaneAI/catalyst/pull/1716)

  This is because [JAX has also dropped support for it since 0.5.0](https://github.com/jax-ml/jax/blob/main/CHANGELOG.md#jax-050-jan-17-2025),
  with the rationale being that such machines are becoming increasingly scarce.

  If support for Mac x86 platforms is still desired, please install
  Catalyst version 0.11.0, PennyLane version 0.41.0, PennyLane-Lightning
  version 0.41.0, and Jax version 0.4.28.

* Sphix version has been updated to 8.1. Some other related packages have been updated as well.
  [(#1734)](https://github.com/PennyLaneAI/catalyst/pull/1734)

<h3>Deprecations 👋</h3>

<h3>Bug fixes 🐛</h3>

* Fix Boolean arguments/results not working with the debugging functions `debug.get_cmain` and
  `debug.compile_executable`.
  [(#1687)](https://github.com/PennyLaneAI/catalyst/pull/1687)

* Fix AutoGraph fallback for valid iteration targets with constant data but no length, for example
  `itertools.product(range(2), repeat=2)`.
  [(#1665)](https://github.com/PennyLaneAI/catalyst/pull/1665)

* Catalyst now correctly supports `qml.StatePrep()` and `qml.BasisState()` operations in the
  experimental PennyLane program-capture pipeline.
  [(#1631)](https://github.com/PennyLaneAI/catalyst/pull/1631)

<h3>Internal changes ⚙️</h3>

* Add an xDSL MLIR plugin to denote whether we will be using xDSL to execute some passes.
  This changelog entry may be moved to new features once all branches are merged together.
  [(#1707)](https://github.com/PennyLaneAI/catalyst/pull/1707)

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
  [(#1708)](https://github.com/PennyLaneAI/catalyst/pull/1708)

<h3>Documentation 📝</h3>

* The header (logo+title) images in the README and in the overview on RtD have been updated,
  reflecting that Catalyst is now beyond the beta!
  [(#1718)](https://github.com/PennyLaneAI/catalyst/pull/1718)

* The API section in the documentation has been simplified. The Catalyst 'Runtime Device Interface'
  page has been updated to point directly to the documented `QuantumDevice` struct, and the 'QIR
  C-API' page has been removed due to limited utility.
  [(#1739)](https://github.com/PennyLaneAI/catalyst/pull/1739)

<h3>Contributors ✍️</h3>

This release contains contributions from (in alphabetical order):

Joey Carter,
Sengthai Heng,
David Ittah,
Tzung-Han Juang,
Christina Lee,
Erick Ochoa Lopez,
Mehrdad Malekmohammadi,
Anton Naim Ibrahim,
Paul Haochen Wang.
