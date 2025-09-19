# Release 0.12.0 (current release)

<h3>New features since last release</h3>

* A new compilation pass called :func:`~.passes.ppm_compilation` has been added to Catalyst to
  transform Clifford+T gates into Pauli Product Measurements (PPMs) using just one transform, allowing for 
  exploring representations of programs in a new paradigm in logical quantum compilation.
  [(#1750)](https://github.com/PennyLaneAI/catalyst/pull/1750)
  
  Based on [arXiv:1808.02892](https://arxiv.org/abs/1808.02892v3), this new compilation pass 
  simplifies circuit transformations and optimizations by combining multiple sub-passes into a 
  single compilation pass, where Clifford+T gates are compiled down to Pauli product rotations 
  (PPRs, :math:`\exp(-iP_{\{x, y, z\}} \theta)`) and PPMs:

  - :func:`~.passes.to_ppr`: converts Clifford+T gates into PPRs.
  - :func:`~.passes.commute_ppr`: commutes PPRs past non-Clifford PPRs.
  - :func:`~.passes.merge_ppr_ppm`: merges Clifford PPRs into PPMs.
  - :func:`~.passes.ppr_to_ppm`: decomposes both non-Clifford PPRs 
  (:math:`\theta = \tfrac{\pi}{8}`), consuming a magic state in the process, and Clifford PPRs 
  (:math:`\theta = \tfrac{\pi}{4}`) into PPMs.
  [(#1664)](https://github.com/PennyLaneAI/catalyst/pull/1664)

  ```python
  import pennylane as qml
  from catalyst.passes import ppm_compilation

  pipeline = [("pipe", ["enforce-runtime-invariants-pipeline"])]

  @qml.qjit(pipelines=pipeline, target="mlir")
  @ppm_compilation(decompose_method="clifford-corrected", avoid_y_measure=True, max_pauli_size=2)
  @qml.qnode(qml.device("null.qubit", wires=2))
  def circuit():
      qml.CNOT([0, 1])
      qml.CNOT([1, 0])
      qml.adjoint(qml.T)(0)
      qml.T(1)
      return catalyst.measure(0), catalyst.measure(1)
  ```

  ```pycon
  >>> print(circuit.mlir_opt)
  ...
  %m, %out:3 = qec.ppm ["Z", "Z", "Z"] %1, %2, %4 : !quantum.bit, !quantum.bit, !quantum.bit
  %m_0, %out_1:2 = qec.ppm ["Z", "Y"] %3, %out#2 : !quantum.bit, !quantum.bit
  %m_2, %out_3 = qec.ppm ["X"] %out_1#1 : !quantum.bit
  %m_4, %out_5 = qec.select.ppm(%m, ["X"], ["Z"]) %out_1#0 : !quantum.bit
  %5 = arith.xori %m_0, %m_2 : i1
  %6:2 = qec.ppr ["Z", "Z"](2) %out#0, %out#1 cond(%5) : !quantum.bit, !quantum.bit
  quantum.dealloc_qb %out_5 : !quantum.bit
  quantum.dealloc_qb %out_3 : !quantum.bit
  %7 = quantum.alloc_qb : !quantum.bit
  %8 = qec.fabricate  magic_conj : !quantum.bit
  %m_6, %out_7:2 = qec.ppm ["Z", "Z"] %6#1, %8 : !quantum.bit, !quantum.bit
  %m_8, %out_9:2 = qec.ppm ["Z", "Y"] %7, %out_7#1 : !quantum.bit, !quantum.bit
  %m_10, %out_11 = qec.ppm ["X"] %out_9#1 : !quantum.bit
  %m_12, %out_13 = qec.select.ppm(%m_6, ["X"], ["Z"]) %out_9#0 : !quantum.bit
  %9 = arith.xori %m_8, %m_10 : i1
  %10 = qec.ppr ["Z"](2) %out_7#0 cond(%9) : !quantum.bit
  quantum.dealloc_qb %out_13 : !quantum.bit
  quantum.dealloc_qb %out_11 : !quantum.bit
  %m_14, %out_15:2 = qec.ppm ["Z", "Z"] %6#0, %10 : !quantum.bit, !quantum.bit
  %from_elements = tensor.from_elements %m_14 : tensor<i1>
  %m_16, %out_17 = qec.ppm ["Z"] %out_15#1 : !quantum.bit
  ...
  ```

* A new function called :func:`~.passes.get_ppm_specs` has been added for acquiring 
  statistics after PPM compilation.
  [(#1794)](https://github.com/PennyLaneAI/catalyst/pull/1794)
  
  After compiling a workflow with any combination of :func:`~.passes.to_ppr`, 
  :func:`~.passes.commute_ppr`, :func:`~.passes.merge_ppr_ppm`, :func:`~.passes.ppr_to_ppm`, or
  :func:`~.passes.ppm_compilation`, use :func:`~.passes.get_ppm_specs` to track useful statistics of
  the compiled workflow, including: 

  - `num_pi4_gates` : number of Clifford PPRs
  - `num_pi8_gates` : number of non-Clifford PPRs
  - `num_pi2_gates` : number of classical PPRs
  - `max_weight_pi4` : maximum weight of Clifford PPRs
  - `max_weight_pi8` : maximum weight of non-Clifford PPRs
  - `max_weight_pi2` : maximum weight of classical PPRs
  - `num_logical_qubits` : number of logical qubits
  - `num_of_ppm` : number of PPMs

  ```python
  from catalyst.passes import get_ppm_specs, to_ppr, merge_ppr_ppm, commute_ppr

  pipe = [("pipe", ["enforce-runtime-invariants-pipeline"])]

  @qjit(pipelines=pipe, target="mlir", autograph=True)
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
          for i in range(10):
            qml.Hadamard(0)
          return measure(0), measure(1)

      return f(), g()
  ```

  ```pycon
  >>> ppm_specs = get_ppm_specs(test_convert_clifford_to_ppr_workflow)
  >>> print(ppm_specs)
  {
  'f_0': {'max_weight_pi8': 1, 'num_logical_qubits': 2, 'num_of_ppm': 2, 'num_pi8_gates': 1}, 
  'g_0': {'max_weight_pi4': 2, 'max_weight_pi8': 1, 'num_logical_qubits': 2, 'num_of_ppm': 2, 'num_pi4_gates': 36, 'num_pi8_gates': 2}
  }
  ```

* Catalyst now supports :class:`qml.Snapshot <pennylane.Snapshot>`, which captures quantum states at 
  any point in a circuit.
  [(#1741)](https://github.com/PennyLaneAI/catalyst/pull/1741)

  For example, the code below is capturing two snapshot'd states, all within a qjit'd circuit:

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
  snapshots, *results = circuit()
  
  >>> print(snapshots)
  [Array([1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j], dtype=complex128), 
  Array([0.5+0.j, 0.5+0.j, 0.5+0.j, 0.5+0.j], dtype=complex128)]
  >>> print(results)
  Array([0.25, 0.25, 0.25, 0.25], dtype=float64)
  ```

  ```pycon
  >>> print(results)
  ([Array([1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j], dtype=complex128), 
  Array([0.5+0.j, 0.5+0.j, 0.5+0.j, 0.5+0.j], dtype=complex128)], 
  Array([0.25, 0.25, 0.25, 0.25], dtype=float64))
  ```

* Catalyst now supports automatic qubit management, meaning that the number of wires does not need 
  to be specified during device initialization.
  [(#1788)](https://github.com/PennyLaneAI/catalyst/pull/1788)

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

  While this feature adds a lot of convenience, it may also reduce performance on devices where
  reallocating resources can be expensive, such as statevector simulators.

* Two new peephole-optimization compilation passes called :func:`~.passes.disentangle_cnot` and 
  :func:`~.passes.disentangle_swap` have been added. Each compilation pass replaces `SWAP` or `CNOT`
  instructions with other equivalent elementary gates.
  [(#1823)](https://github.com/PennyLaneAI/catalyst/pull/1823)

  As an example, :func:`~.passes.disentangle_cnot` applied to the circuit below will replace the 
  `CNOT` gate with an `X` gate.

  ```python
  dev = qml.device("lightning.qubit", wires=2)

  @qml.qjit(keep_intermediate=True)
  @catalyst.passes.disentangle_cnot
  @qml.qnode(dev)
  def circuit():
      # first qubit in |1>
      qml.X(0)
      # second qubit in |0>
      # current state : |10>
      qml.CNOT([0,1]) # state after CNOT : |11>
      return qml.state()
  ```

  ```pycon
  >>> from catalyst.debug import get_compilation_stage
  >>> print(get_compilation_stage(circuit, stage="QuantumCompilationPass"))
  ...
  %out_qubits = quantum.custom "PauliX"() %1 : !quantum.bit
  %2 = quantum.extract %0[ 1] : !quantum.reg -> !quantum.bit
  %out_qubits_0 = quantum.custom "PauliX"() %2 : !quantum.bit
  ...
  ```

<h3>Improvements 🛠</h3>

* The :func:`qml.measure <pennylane.measure>` operation for mid-circuit measurements can now be used 
  in qjit-compiled circuits with program capture enabled.
  [(#1766)](https://github.com/PennyLaneAI/catalyst/pull/1766)

  Note that the simulation behaviour of mid-circuit measurements can differ between PennyLane and
  Catalyst, depending on the chosen `mcm_method`. Please see the 
  *Functionality differences from PennyLane* section in the
  :doc:`sharp bits and debugging tips page <sharp_bits>` for additional information.

* The behaviour of measurement processes executed on `null.qubit` with qjit is now more consistent 
  with their behaviour on `null.qubit` *without* qjit.
  [(#1598)](https://github.com/PennyLaneAI/catalyst/pull/1598)

  Previously, measurement processes like `qml.sample`, `qml.counts`, `qml.probs`, etc.,
  returned values from uninitialized memory when executed on `null.qubit` with qjit. This change
  ensures that measurement processes on `null.qubit` always return the value 0 or the result
  corresponding to the '0' state, depending on the context.

* The package name of the Catalyst distribution has been updated to be consistent with
  [PyPA standards](https://packaging.python.org/en/latest/specifications/binary-distribution-format/#file-name-convention),
  from `PennyLane-Catalyst` to `pennylane_catalyst`. This change is not expected to affect users as 
  tools in the Python ecosystem (e.g. `pip`) already handle both versions through normalization. 
  [(#1817)](https://github.com/PennyLaneAI/catalyst/pull/1817)

* The :func:`~.passes.commute_ppr` and :func:`~.passes.merge_ppr_ppm` passes now accept an optional
  `max_pauli_size` argument, which limits the size of the Pauli strings generated by the passes
  through commutation or absorption rules.
  [(#1719)](https://github.com/PennyLaneAI/catalyst/pull/1719)

* The :func:`~.passes.to_ppr` pass is now more efficient by adding support for the direct conversion 
  of Pauli gates (`qml.X`, `qml.Y`, `qml.Z`), the adjoint of `qml.S` gate, and the adjoint of the 
  `qml.T` gate.
  [(#1738)](https://github.com/PennyLaneAI/catalyst/pull/1738)

* The `keep_intermediate` argument in the `qjit` decorator now accepts a new value that allows for
  saving intermediate files after each pass. The updated possible options for this argument are:
  - `False` or `0` or `None` : No intermediate files are kept.
  - `True` or `1` or `"pipeline"`: Intermediate files are saved after each pipeline.
  - `2` or `"pass"`: Intermediate files are saved after each pass.

  The default value is `False`.
  [(#1791)](https://github.com/PennyLaneAI/catalyst/pull/1791)

* The `static_argnums` keyword argument in the `qjit` decorator is now compatible with PennyLane 
  program capture enabled (:func:`qml.capture.enable <pennylane.capture.enable>`).
  [(#1810)](https://github.com/PennyLaneAI/catalyst/pull/1810)

* Catalyst is compatible with the new :func:`qml.set_shots <pennylane.set_shots>` transform 
  introduced in PennyLane v0.42.
  [(#1784)](https://github.com/PennyLaneAI/catalyst/pull/1784)

* `null.qubit` can now support an optional `track_resources` keyword argument, which allows it to record 
  which gates are executed. 
  [(#1619)](https://github.com/PennyLaneAI/catalyst/pull/1619)

  ```python
  import json
  import glob

  dev = qml.device("null.qubit", wires=2, track_resources=True)

  @qml.qjit
  @qml.qnode(dev)
  def circuit():
      for _ in range(5):
          qml.H(0)
      qml.CNOT([0, 1])
      return qml.probs()

  circuit()

  pattern = "./__pennylane_resources_data_*"
  filepath = glob.glob(pattern)[0]
  with open(filepath) as f:
      resources = json.loads(f.read())
  ```

  ```pycon
  >>> print(resources)
  {'num_qubits': 2, 'num_gates': 6, 'gate_types': {'CNOT': 1, 'Hadamard': 5}}
  ```

<h3>Breaking changes 💔</h3>

* Support for Mac x86 has been removed. This includes Macs running on Intel processors.
  [(#1716)](https://github.com/PennyLaneAI/catalyst/pull/1716)

  This is because 
  [JAX has also dropped support for it since 0.5.0](https://github.com/jax-ml/jax/blob/main/CHANGELOG.md#jax-050-jan-17-2025),
  with the rationale being that such machines are becoming increasingly scarce.

  If support for Mac x86 platforms is still desired, please install Catalyst v0.11.0, PennyLane 
  v0.41.0, PennyLane-Lightning v0.41.0, and JAX v0.4.28.

* (Device Developers Only) The `QuantumDevice` interface in the Catalyst Runtime plugin system has been modified, which
  requires recompiling plugins for binary compatibility.
  [(#1680)](https://github.com/PennyLaneAI/catalyst/pull/1680)

  As announced in the 
  [0.10.0 release](https://docs.pennylane.ai/projects/catalyst/en/stable/dev/release_notes.html#release-0-10-0),
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

* (Frontend Developers Only) Some Catalyst primitives for JAX have been renamed, and the qubit deallocation primitive has been 
  split into deallocation and a separate device release primitive.
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
  capture can be enabled via the global toggle 
  :func:`qml.capture.enable() <pennylane.capture.enable>`:

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

  Disabling program capture can be done with 
  :func:`qml.capture.disable() <pennylane.capture.disable>`.

* The `ppr_to_ppm` pass functionality has been moved to a new pass called `merge_ppr_ppm`. The 
  `ppr_to_ppm` functionality now handles direct decomposition of PPRs into PPMs.
  [(#1688)](https://github.com/PennyLaneAI/catalyst/pull/1688)

* The version of JAX used by Catalyst has been updated to v0.6.0.
  [(#1652)](https://github.com/PennyLaneAI/catalyst/pull/1652)
  [(#1729)](https://github.com/PennyLaneAI/catalyst/pull/1729)

  Several internal changes were made for this update.
  - LAPACK kernels are updated to adhere to the new JAX lowering rules for external functions.
  [(#1685)](https://github.com/PennyLaneAI/catalyst/pull/1685)

  - The trace stack is removed and replaced with a tracing context manager.
  [(#1662)](https://github.com/PennyLaneAI/catalyst/pull/1662)

  - A new `debug_info` argument is added to `Jaxpr`, the `make_jaxpr` functions, and 
  `jax.extend.linear_util.wrap_init`.
  [(#1670)](https://github.com/PennyLaneAI/catalyst/pull/1670)
  [(#1671)](https://github.com/PennyLaneAI/catalyst/pull/1671)
  [(#1681)](https://github.com/PennyLaneAI/catalyst/pull/1681)

* The version of LLVM, mlir-hlo, and Enzyme used by Catalyst has been updated to track those in JAX 
  v0.6.0.
  [(#1752)](https://github.com/PennyLaneAI/catalyst/pull/1752)

  The LLVM version has been updated to 
  [commit a8513158](https://github.com/llvm/llvm-project/tree/179d30f8c3fddd3c85056fd2b8e877a4a8513158).
  The mlir-hlo version has been updated to 
  [commit e30c22d1](https://github.com/tensorflow/mlir-hlo/tree/617a9361d186199480c080c9e8c474a5e30c22d1).
  The Enzyme version has been updated to 
  [v0.0.180](https://github.com/EnzymeAD/Enzyme/releases/tag/v0.0.180).

* (Device developers only) Device parameters which are forwarded by the Catalyst runtime
 to plugin devices as a string may not contain nested dictionaries. Previously, these would
 be parsed incorrectly, and instead will now raise an error.
  [(#1843)](https://github.com/PennyLaneAI/catalyst/pull/1843)
  [(#1846)](https://github.com/PennyLaneAI/catalyst/pull/1846)

<h3>Deprecations 👋</h3>

* Python 3.10 is now deprecated and will not be supported in Catalyst v0.13. Please upgrade to a newer Python version.

<h3>Bug fixes 🐛</h3>

* Fixed Boolean arguments/results not working with the debugging functions `debug.get_cmain` and
  `debug.compile_executable`.
  [(#1687)](https://github.com/PennyLaneAI/catalyst/pull/1687)

* Fixed AutoGraph fallback for valid iteration targets with constant data but no length, for example
  `itertools.product(range(2), repeat=2)`.
  [(#1665)](https://github.com/PennyLaneAI/catalyst/pull/1665)

* Catalyst now correctly supports `qml.StatePrep()` and `qml.BasisState()` operations in the
  experimental PennyLane program capture pipeline.
  [(#1631)](https://github.com/PennyLaneAI/catalyst/pull/1631)

* `make all` now correctly compiles the standalone plugin with the same compiler used to compile 
  LLVM and MLIR.
  [(#1768)](https://github.com/PennyLaneAI/catalyst/pull/1768)

* Stacked Python decorators for built-in Catalyst passes are now applied in the correct order.
  [(#1798)](https://github.com/PennyLaneAI/catalyst/pull/1798)

* MLIR plugins can now be specified via lists and tuples, not just sets.
  [(#1812)](https://github.com/PennyLaneAI/catalyst/pull/1812)

* Fixed the conversion of PLxPR to JAXPR with quantum primitives when using control flow.
  [(#1809)](https://github.com/PennyLaneAI/catalyst/pull/1809)

* Fixed a bug in the internal simplification of qubit chains in the compiler, which manifested in 
  certain transformations like `cancel_inverses ` and led to incorrect results.
  [(#1840)](https://github.com/PennyLaneAI/catalyst/pull/1840)

* Fixes the conversion of PLxPR to JAXPR with quantum primitives when using dynamic wires.
  [(#1842)](https://github.com/PennyLaneAI/catalyst/pull/1842)

<h3>Internal changes ⚙️</h3>

* The clang-format and clang-tidy versions used by Catalyst have been updated to v20.
  [(#1721)](https://github.com/PennyLaneAI/catalyst/pull/1721)

* The Sphinx version has been updated to v8.1.
  [(#1734)](https://github.com/PennyLaneAI/catalyst/pull/1734)

* Integration with PennyLane's experimental Python compiler based on xDSL has been added. This 
  allows developers and users to write xDSL transformations that can be used with Catalyst.
  [(#1715)](https://github.com/PennyLaneAI/catalyst/pull/1715)

* An xDSL MLIR plugin has been added to denote whether to use xDSL to execute compilation passes.
  [(#1707)](https://github.com/PennyLaneAI/catalyst/pull/1707)

* The function `dataclass.replace` is now used to update `ExecutionConfig` and `MCMConfig` rather 
  than mutating properties.
  [(#1814)](https://github.com/PennyLaneAI/catalyst/pull/1814)

* A function has been added that allows developers to register an equivalent MLIR transform for a 
  given PLxPR transform.
  [(#1705)](https://github.com/PennyLaneAI/catalyst/pull/1705)

* Overriding the `num_wires` property of `HybridOp` is no longer happening when the operator can 
  exist on `AnyWires`. This allows the deprecation of `WiresEnum` in PennyLane.
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

* Support for quantum subroutines was added. This feature is expected to improve compilation times 
  for large quantum programs.
  [(#1774)](https://github.com/PennyLaneAI/catalyst/pull/1774)
  [(#1828)](https://github.com/PennyLaneAI/catalyst/pull/1828)

* PennyLane's arbitrary-basis measurement operations, such as 
  :func:`qml.ftqc.measure_arbitrary_basis <pennylane.ftqc.measure_arbitrary_basis>`, are now 
  qjit-compatible with PennyLane program capture enabled.
  [(#1645)](https://github.com/PennyLaneAI/catalyst/pull/1645)
  [(#1710)](https://github.com/PennyLaneAI/catalyst/pull/1710)

* The utility function `EnsureFunctionDeclaration` has been refactored into the `Utils` of the 
  Catalyst dialect instead of being duplicated in each individual dialect.
  [(#1683)](https://github.com/PennyLaneAI/catalyst/pull/1683)

* The assembly format for some MLIR operations now includes `adjoint`.
  [(#1695)](https://github.com/PennyLaneAI/catalyst/pull/1695)

* Improved the definition of `YieldOp` in the quantum dialect by removing `AnyTypeOf`.
  [(#1696)](https://github.com/PennyLaneAI/catalyst/pull/1696)

* The assembly format of `MeasureOp` in the `Quantum` dialect and `MeasureInBasisOp` in the `MBQC` 
  dialect now contains the `postselect` attribute.
  [(#1732)](https://github.com/PennyLaneAI/catalyst/pull/1732)

* The bufferization of custom Catalyst dialects has been migrated to the new one-shot bufferization 
  interface in MLIR. The new MLIR bufferization interface is required by JAX v0.4.29 or higher.
  [(#1027)](https://github.com/PennyLaneAI/catalyst/pull/1027)
  [(#1686)](https://github.com/PennyLaneAI/catalyst/pull/1686)
  [(#1708)](https://github.com/PennyLaneAI/catalyst/pull/1708)
  [(#1740)](https://github.com/PennyLaneAI/catalyst/pull/1740)
  [(#1751)](https://github.com/PennyLaneAI/catalyst/pull/1751)
  [(#1769)](https://github.com/PennyLaneAI/catalyst/pull/1769)

* The redundant `OptionalAttr` has been removed from the `adjoint` argument in the `QuantumOps.td` 
  TableGen file.
  [(#1746)](https://github.com/PennyLaneAI/catalyst/pull/1746)

* `ValueRange` has been replaced with `TypeRange` for creating `CustomOp` in 
  `IonsDecompositionPatterns.cpp` to match the build constructors.
  [(#1749)](https://github.com/PennyLaneAI/catalyst/pull/1749)

* The unused helper function `genArgMapFunction` in the `--lower-gradients` pass has been removed.
  [(#1753)](https://github.com/PennyLaneAI/catalyst/pull/1753)

* Base components of `QFuncPLxPRInterpreter` have been moved into a base class called 
  `SubroutineInterpreter`. This is intended to reduce code duplication.
  [(#1787)](https://github.com/PennyLaneAI/catalyst/pull/1787)

* An argument (`openapl_file_name`) has been added to the `OQDDevice` constructor to specify the 
  name of the output OpenAPL file.
  [(#1763)](https://github.com/PennyLaneAI/catalyst/pull/1763)

* The OQD device TOML file has been modified to only include gates that are decomposable to the OQD 
  device target gate set.
  [(#1763)](https://github.com/PennyLaneAI/catalyst/pull/1763)

* The `quantum-to-ion` pass has been renamed to `gates-to-pulses`.
  [(#1818)](https://github.com/PennyLaneAI/catalyst/pull/1818)

* The runtime CAPI function `__catalyst__rt__num_qubits` now has a corresponding JAX primitive
  `num_qubits_p` and quantum dialect operation `NumQubitsOp`.
  [(#1793)](https://github.com/PennyLaneAI/catalyst/pull/1793)

  For measurements whose shapes depend on the number of qubits, they now properly retrieve the
  number of qubits through this new operation when it is dynamic.

* The PPR/PPM pass names have been renamed from snake-case to kebab-case in MLIR to align with MLIR 
  conventions. Class names and tests were updated accordingly. Example: `--to_ppr` is now 
  `--to-ppr`.
  [(#1802)](https://github.com/PennyLaneAI/catalyst/pull/1802)

* A new internal python module called `catalyst.from_plxpr` has been created to better organize the 
  code for plxpr integration.
  [(#1813)](https://github.com/PennyLaneAI/catalyst/pull/1813)

* A new `from_plxpr.QregManager` has been created to handle converting plxpr wire index semantics 
  into catalyst qubit value semantics.
  [(#1813)](https://github.com/PennyLaneAI/catalyst/pull/1813)

<h3>Documentation 📝</h3>

* The header (logo+title) images in the README and in the overview on ReadTheDocs have been updated,
  reflecting that Catalyst is now beyond beta 🎉!
  [(#1718)](https://github.com/PennyLaneAI/catalyst/pull/1718)

* The API section in the documentation has been simplified. The Catalyst 'Runtime Device Interface'
  page has been updated to point directly to the documented `QuantumDevice` struct, and the 'QIR
  C-API' page has been removed due to limited utility.
  [(#1739)](https://github.com/PennyLaneAI/catalyst/pull/1739)

<h3>Contributors ✍️</h3>

This release contains contributions from (in alphabetical order):

Runor Agbaire,
Joey Carter,
Isaac De Vlugt,
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
