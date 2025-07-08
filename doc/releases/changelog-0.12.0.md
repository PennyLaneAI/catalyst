# Release 0.12.0 (current release)

<h3>New features since last release</h3>

* Added integration with PennyLane's experimental python compiler based on xDSL.
  This allows developers and users to write xDSL transformations that can be used with Catalyst.
  [(#1715)](https://github.com/PennyLaneAI/catalyst/pull/1715)

* A new compilation pass called :func:`~.passes.ppr_to_ppm` has been added to Catalyst
  to decompose Pauli product rotations (PPRs), :math:`\exp(-iP_{\{x, y, z\}} \theta)`, into
  Pauli product measurements (PPMs). Non-Clifford PPR (:math:`\theta = \tfrac{\pi}{8}`) requires
  the consumption of a magic state, while Clifford PPR (:math:`\theta = \tfrac{\pi}{4}`) will not.
  The methods are implemented as described in [arXiv:1808.02892](https://arxiv.org/abs/1808.02892v3).
  [(#1664)](https://github.com/PennyLaneAI/catalyst/pull/1664)

  The new compilation pass can be accessed in the frontend from the :mod:`~.passes` module:
  * :func:`~.passes.ppr_to_ppm`: Decomposes all PPRs into PPMs
  (except PPR (:math:`\theta = \tfrac{\pi}{2}`)).

  Or via the Catalyst CLI as two separate passes:
  * `decompose_clifford_ppr`: Decompose Clifford PPR (:math:`\theta = \tfrac{\pi}{4}`)
  rotations into PPMs.
  * `decompose_non_clifford_ppr`: Decompose non-Cliford PPR (:math:`\theta = \tfrac{\pi}{8}`)
  into PPMs using a magic state.

* A new compilation pass called :func:`~.passes.ppm_compilation` has been added to Catalyst to 
  transform Clifford+T gates into Pauli Product Measurements (PPMs). This high-level pass simplifies
  circuit transformation and optimization by combining multiple sub-passes into a single step.
  [(#1750)](https://github.com/PennyLaneAI/catalyst/pull/1750)
  
  The sub-passes that make up the :func:`~.passes.ppm_compilation` pass are:
  * :func:`~.passes.to_ppr`: Converts gates into Pauli Product Rotations (PPRs).
  * :func:`~.passes.commute_ppr`: Commutes PPRs past non-Clifford PPRs.
  * :func:`~.passes.merge_ppr_ppm`: Merges Clifford PPRs into PPMs.
  * :func:`~.passes.ppr_to_ppm`: Decomposes non-Clifford and Clifford PPRs into PPMs.

  Use this pass via the :func:`~.passes.ppm_compilation` decorator to compile circuits 
  in a single pipeline.

  * A new function :func:`~.passes.get_ppm_specs` to get the result statistics after a PPR/PPM compilations is available. The statistics is returned in a Python dictionary.
  [(#1794)](https://github.com/PennyLaneAI/catalyst/pull/1794)
  [(#1863)](https://github.com/PennyLaneAI/catalyst/pull/1863)
  
  Example below shows an input circuit and corresponding PPM specs.

  ```python
  import pennylane as qml
  from catalyst import qjit, measure
  from catalyst.passes import get_ppm_specs, to_ppr, merge_ppr_ppm, commute_ppr

  pipe = [("pipe", ["enforce-runtime-invariants-pipeline"])]

  @qjit(pipelines=pipe, target="mlir")
  def test_convert_clifford_to_ppr_workflow():

      device = qml.device("lightning.qubit", wires=2)

      @merge_ppr_ppm
      @commute_ppr(max_pauli_size=2)
      @to_ppr
      @qml.qnode(device)
      def f():
          qml.CNOT([0, 2])
          qml.T(0)
          return measure(0), measure(1)

      @merge_ppr_ppm(max_pauli_size=1)
      @commute_ppr
      @to_ppr
      @qml.qnode(device)
      def g():
          qml.CNOT([0, 2])
          qml.T(0)
          qml.T(1)
          qml.CNOT([0, 1])
          return measure(0), measure(1)

      return f(), g()

  ppm_specs = get_ppm_specs(test_convert_clifford_to_ppr_workflow)
  print(ppm_specs)

  ```
  Output:
  ```pycon
  {
  'f_0': {'max_weight_pi8': 1, 'num_logical_qubits': 2, 'num_of_ppm': 2, 'num_pi8_gates': 1}, 
  'g_0': {'max_weight_pi4': 2, 'max_weight_pi8': 1, 'num_logical_qubits': 2, 'num_of_ppm': 2, 'num_pi4_gates': 3, 'num_pi8_gates': 2}
  }
  ```

* Support for :class:`qml.Snapshot <pennylane.Snapshot>` to capture quantum states at any 
  point in a circuit has been added to Catalyst [(#1741)](https://github.com/PennyLaneAI/catalyst/pull/1741).
  For example, the code below is capturing 
  two snapshot states:

  ``` python
  NUM_QUBITS = 2
  dev = qml.device("lightning.qubit", wires=NUM_QUBITS)

  @qjit
  @qml.qnode(dev)
  def circuit():
      wires = list(range(NUM_QUBITS))
      qml.Snapshot("Initial state")

      for wire in wires:
          qml.Hadamard(wires=wire)

      qml.Snapshot("After applying Hadamard gates")

      return qml.probs()

  results = circuit()
  print(results)
  ```

  The output would be a tuple of two elements: 
    * Array of snapshot states
    * Tuple of measurements being returned

  ```pycon
  >>> print(results)
  ([Array([1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j], dtype=complex128), 
  Array([0.5+0.j, 0.5+0.j, 0.5+0.j, 0.5+0.j], dtype=complex128)], 
  Array([0.25, 0.25, 0.25, 0.25], dtype=float64))
  ```

* Catalyst now supports automatic qubit management.
  [(#1788)](https://github.com/PennyLaneAI/catalyst/pull/1788)

  The number of wires does not need to be speficied during device initialization,
  and instead will be automatically managed by the Catalyst Runtime.

  ```python
  @qjit
  def workflow():
      dev = qml.device("lightning.qubit") # no wires here!
      @qml.qnode(dev)
      def circuit():
          qml.PauliX(wires=2)
          return qml.probs()
      return circuit()

  print(workflow())
  ```

  ```pycon
  [0. 1. 0. 0. 0. 0. 0. 0.]
  ```

  In this example, the number of wires is not specified at device initialization.
  When we encounter an X gate on `wires=2`, catalyst automatically expands the size
  of the qubit register to include the requested wire index.
  Here, the register will contain (at least) 3 qubits after the X operation.
  As a result, we can see the QNode returning the probabilities for the state |001>,
  meaning 3 wires were allocated in total.

  This feature can be turned on by omitting the `wires` argument to the device.

<h3>Improvements 🛠</h3>

* The package name of the Catalyst distribution has been updated to be inline with
  [PyPA standards](https://packaging.python.org/en/latest/specifications/binary-distribution-format/#file-name-convention),
  from `PennyLane-Catalyst` to `pennylane_catalyst`. This change is not expected to
  affect users, besides for instance the installation directory name, as tools in the
  Python ecosystem (e.g. `pip`) already handle both versions through normalization.
  [(#1817)](https://github.com/PennyLaneAI/catalyst/pull/1817)

* The behaviour of measurement processes executed on `null.qubit` with QJIT is now more in line with
  their behaviour on `null.qubit` *without* QJIT.
  [(#1598)](https://github.com/PennyLaneAI/catalyst/pull/1598)

  Previously, measurement processes like `qml.sample()`, `qml.counts()`, `qml.probs()`, etc.
  returned values from uninitialized memory when executed on `null.qubit` with QJIT. This change
  ensures that measurement processes on `null.qubit` always return the value 0 or the result
  corresponding to the '0' state, depending on the context.

* The :func:`~.passes.commute_ppr` and :func:`~.passes.merge_ppr_ppm` passes now accept an optional
  `max_pauli_size` argument, which limits the size of the Pauli strings generated by the passes
  through commutation or absorption rules.
  [(#1719)](https://github.com/PennyLaneAI/catalyst/pull/1719)

* The :func:`~.passes.to_ppr` pass now supports conversion of Pauli gates (`X`, `Y`, `Z`),
  the phase gate adjoint (`S†`), and the π/8 gate adjoint (`T†`). This extension improves
  performance by eliminating indirect conversion.
  [(#1738)](https://github.com/PennyLaneAI/catalyst/pull/1738)

* The `keep_intermediate` argument in the `qjit` decorator now accepts a new value that allows for
  saving intermediate files after each pass. The updated possible options for this argument are:
  * `False` or `0` or `"none"` or `None` : No intermediate files are kept.
  * `True` or `1` or `"pipeline"`: Intermediate files are saved after each pipeline.
  * `2` or `"pass"`: Intermediate files are saved after each pass.
  The default value is `False`.
  [(#1791)](https://github.com/PennyLaneAI/catalyst/pull/1791)

* `static_argnums` on `qjit` can now be specified with program capture through PLxPR.
  [(#1810)](https://github.com/PennyLaneAI/catalyst/pull/1810)

* Two new Python pass decorators, :func:`~.passes.disentangle_cnot` and 
  :func:`~.passes.disentangle_swap` have been added. They apply their corresponding MLIR passes 
  `--disentangle-CNOT` and `--disentangle-SWAP` respectively. 
  [(#1823)](https://github.com/PennyLaneAI/catalyst/pull/1823)

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

* (Compiler developers only) The version of LLVM, mlir-hlo and Enzyme used by Catalyst is
  updated to track those in jax 0.6.0.
  [(#1752)](https://github.com/PennyLaneAI/catalyst/pull/1752)

  The LLVM version is updated to [commit 179d30f8c3fddd3c85056fd2b8e877a4a8513158](https://github.com/llvm/llvm-project/tree/179d30f8c3fddd3c85056fd2b8e877a4a8513158).
  The mlir-hlo version is updated to [commit 617a9361d186199480c080c9e8c474a5e30c22d1](https://github.com/tensorflow/mlir-hlo/tree/617a9361d186199480c080c9e8c474a5e30c22d1).
  The Enzyme version is updated to [v0.0.180](https://github.com/EnzymeAD/Enzyme/releases/tag/v0.0.180).

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

* The device kwarg parsing in runtime has been updated to disallow nested dictionary formats.
  [(#1843)](https://github.com/PennyLaneAI/catalyst/pull/1843)
  [(#1846)](https://github.com/PennyLaneAI/catalyst/pull/1846)

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

* `make all` now correctly compiles the standalone plugin with the same compiler used to compile LLVM and MLIR.
  [(#1768)](https://github.com/PennyLaneAI/catalyst/pull/1768)

* Stacked python decorators for built-in catalyst passes are now applied in the correct order.
  [(#1798)](https://github.com/PennyLaneAI/catalyst/pull/1798)

* MLIR plugins can now be specified via lists and tuples, not just sets.
  [(#1812)](https://github.com/PennyLaneAI/catalyst/pull/1812)

* Fixes the conversion of PLxPR to JAXPR with quantum primitives when using control flow.
  [(#1809)](https://github.com/PennyLaneAI/catalyst/pull/1809)

* Fixes canonicalization of insertion and extraction into quantum registers.
  [(#1840)](https://github.com/PennyLaneAI/catalyst/pull/1840)

<h3>Internal changes ⚙️</h3>

* `qml.qjit` now integrates with the new `qml.set_shots` function.
  [(#1784)](https://github.com/PennyLaneAI/catalyst/pull/1784)

* Use `dataclass.replace` to update `ExecutionConfig` and `MCMConfig` rather than mutating properties.
  [(#1814)](https://github.com/PennyLaneAI/catalyst/pull/1814)

* `null.qubit` can now support an optional `track_resources` argument which allows it to record which gates are executed.
  [(#1619)](https://github.com/PennyLaneAI/catalyst/pull/1619)

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

* Support for quantum subroutines was added.
  This feature is expected to improve compilation times for large quantum programs.
  [(#1774)](https://github.com/PennyLaneAI/catalyst/pull/1774)
  [(#1828)](https://github.com/PennyLaneAI/catalyst/pull/1828)

* PennyLane's arbitrary-basis measurement operations, such as
  :func:`qml.ftqc.measure_arbitrary_basis() <pennylane.ftqc.measure_arbitrary_basis>`, are now
  QJIT-compatible with program capture enabled.
  [(#1645)](https://github.com/PennyLaneAI/catalyst/pull/1645)
  [(#1710)](https://github.com/PennyLaneAI/catalyst/pull/1710)

* The utility function `EnsureFunctionDeclaration` is refactored into the `Utils` of the `Catalyst`
  dialect, instead of being duplicated in each individual dialect.
  [(#1683)](https://github.com/PennyLaneAI/catalyst/pull/1683)

* The assembly format for some MLIR operations now includes adjoint.
  [(#1695)](https://github.com/PennyLaneAI/catalyst/pull/1695)

* Improved the definition of `YieldOp` in the quantum dialect by removing `AnyTypeOf`
  [(#1696)](https://github.com/PennyLaneAI/catalyst/pull/1696)

* The assembly format of `MeasureOp` in the `Quantum` dialect and `MeasureInBasisOp` in the `MBQC` dialect now contains the `postselect` attribute.
  [(#1732)](https://github.com/PennyLaneAI/catalyst/pull/1732)

* The bufferization of custom catalyst dialects has been migrated to the new one-shot
  bufferization interface in mlir.
  The new mlir bufferization interface is required by jax 0.4.29 or higher.
  [(#1027)](https://github.com/PennyLaneAI/catalyst/pull/1027)
  [(#1686)](https://github.com/PennyLaneAI/catalyst/pull/1686)
  [(#1708)](https://github.com/PennyLaneAI/catalyst/pull/1708)
  [(#1740)](https://github.com/PennyLaneAI/catalyst/pull/1740)
  [(#1751)](https://github.com/PennyLaneAI/catalyst/pull/1751)
  [(#1769)](https://github.com/PennyLaneAI/catalyst/pull/1769)

* Redundant `OptionalAttr` is removed from `adjoint` argument in `QuantumOps.td` TableGen file
  [(#1746)](https://github.com/PennyLaneAI/catalyst/pull/1746)

* `ValueRange` is replaced with `TypeRange` for creating `CustomOp` in `IonsDecompositionPatterns.cpp` to match the build constructors
  [(#1749)](https://github.com/PennyLaneAI/catalyst/pull/1749)

* The unused helper function `genArgMapFunction` in the `--lower-gradients` pass is removed.
  [(#1753)](https://github.com/PennyLaneAI/catalyst/pull/1753)

* Base components of QFuncPLxPRInterpreter have been moved into a base class called SubroutineInterpreter.
  This is to reduce code duplication once we have support for quantum subroutines.
  [(#1787)](https://github.com/PennyLaneAI/catalyst/pull/1787)

* The `qml.measure()` operation for mid-circuit measurements can now be used in QJIT-compiled
  circuits with program capture enabled.
  [(#1766)](https://github.com/PennyLaneAI/catalyst/pull/1766)

  Note that using `qml.measure()` in this way binds the operation to :func:`catalyst.measure`, which
  behaves differently than `qml.measure()` in a native PennyLane circuit, as described in the
  *Functionality differences from PennyLane* section of the
  :doc:`sharp bits and debugging tips <sharp_bits>` guide. In regular QJIT-compiled workloads
  (without program capture enabled), you must continue to use :func:`catalyst.measure`.

* An argument (`openapl_file_name`) is added to the `OQDDevice` constructor to specify the name of
  the output OpenAPL file.
  [(#1763)](https://github.com/PennyLaneAI/catalyst/pull/1763)

* The OQD device toml file is modified to only include gates that are decomposable to the OQD device
  target gate set.
  [(#1763)](https://github.com/PennyLaneAI/catalyst/pull/1763)

* The `quantum-to-ion` pass is renamed to `gates-to-pulses`.
  [(#1818)](https://github.com/PennyLaneAI/catalyst/pull/1818)

* The runtime CAPI function `__catalyst__rt__num_qubits` now has a corresponding jax primitive
  `num_qubits_p` and quantum dialect operation `NumQubitsOp`.
  [(#1793)](https://github.com/PennyLaneAI/catalyst/pull/1793)

  For measurements whose shapes depend on the number of qubits, they now properly retrieve the
  number of qubits through this new operation when it is dynamic.

* Refactored PPR/PPM pass names from snake_case to kebab-case in MLIR passes to align with MLIR conventions.
  Class names and tests were updated accordingly. Example: `--to_ppr` is now `--to-ppr`.
  [(#1802)](https://github.com/PennyLaneAI/catalyst/pull/1802)

* A new internal python module `catalyst.from_plxpr` is created to better organize the code for plxpr capture integration.
  [(#1813)](https://github.com/PennyLaneAI/catalyst/pull/1813)

* A new `from_plxpr.QregManager` is created to handle converting plxpr wire index semantics into catalyst qubit value semantics.
  [(#1813)](https://github.com/PennyLaneAI/catalyst/pull/1813)

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

Runor Agbaire,
Joey Carter,
Sengthai Heng,
David Ittah,
Tzung-Han Juang,
Christina Lee,
Mehrdad Malekmohammadi,
Anton Naim Ibrahim,
Erick Ochoa Lopez,
Ritu Thombre,
Raul Torres,
Paul Haochen Wang,
Jake Zaia.
