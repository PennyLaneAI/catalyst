# Release 0.15.0 (current release)

<h3>New features since last release</h3>

* An experimental lookup table (LUT) decoder is added to the `runtime`. This initial implementation
is optimized for the [[7,1,3]] Steane code using hardcoded Quantum Error Correction (QEC) data. While
the architecture supports future extension to general LUT decoding via compiler-provided information,
please note that LUT decoders scale exponentially with code size and are intended for small-scale QEC
codes only.
[(#2724)](https://github.com/PennyLaneAI/catalyst/pull/2724)
[(#2800)](https://github.com/PennyLaneAI/catalyst/pull/2800)

* Combining ``GlobalPhase`` operations into one single operation is now possible with the
  :func:`catalyst.passes.combine_global_phases` pass.
  [(#2553)](https://github.com/PennyLaneAI/catalyst/pull/2553)

  ```python
  import pennylane as qp
  import catalyst

  @qp.qjit(capture=True)
  @catalyst.passes.combine_global_phases
  @qp.qnode(qml.device("lightning.qubit", wires=5))
  def circuit():
      qp.GlobalPhase(0)
      qp.GlobalPhase(1)
      qp.GlobalPhase(2)
      qp.GlobalPhase(3)
      qp.GlobalPhase(4)
      return qp.state()

  Device: lightning.qubit
  Device wires: 5
  Shots: Shots(total=None)
  Level: combine-global-phases (MLIR-1)
  <BLANKLINE>
  Wire allocations: 5
  Total gates: 1
  Gate counts:
  - GlobalPhase: 1
  Measurements:
  - state(all wires): 1
  Depth: Not computed
  ```

* A new `~.CompilationPass` class has been added that abstracts away compiler-level details for
  seamless compilation pass creation. Used in tandem with :func:`~.compiler_transform`, compilation
  passes can be created entirely in Python and used on QNodes within a :func:`~.qjit`'d workflow.
  [(#2211)](https://github.com/PennyLaneAI/catalyst/pull/2211)

* A new MLIR transformation pass `--dynamic-one-shot` is available.
  Devices that natively support mid-circuit measurements can evaluate dynamic circuits by executing
  them one shot at a time, sampling a dynamic execution path for each shot. The `--dynamic-one-shot`
  pass first transforms the circuit so that each circuit execution only contains a singular shot,
  then performs the appropriate classical statistical postprocessing across the execution results
  from all shots.
  [(#2458)](https://github.com/PennyLaneAI/catalyst/pull/2458)
  [(#2573)](https://github.com/PennyLaneAI/catalyst/pull/2573)

  With this new MLIR pass, one shot execution mode is now available when capture is enabled.

  ```python
  dev = qp.device("lightning.qubit", wires=2)

  @qjit(capture=True)
  @qp.transform(pass_name="dynamic-one-shot")
  @qp.qnode(dev, shots=10)
  def circuit():
      qp.Hadamard(wires=0)
      m_0 = qp.measure(0)
      m_1 = qp.measure(1)
      return qp.sample([m_0, m_1]), qp.expval(m_0), qp.probs(op=[m_0,m_1]), qp.counts(wires=0)
  ```

  ```pycon
  >>> circuit()
  (Array([[1, 0],
         [0, 0],
         [1, 0],
         [1, 0],
         [0, 0],
         [0, 0],
         [1, 0],
         [1, 0],
         [1, 0],
         [0, 0]], dtype=int64), Array(0.6, dtype=float64), Array([0.4, 0. , 0.6, 0. ], dtype=float64),
         (Array([0, 1], dtype=int64), Array([4, 6], dtype=int64)))
  ```

  Note that although the one-shot transform is motivated from the context of mid-circuit measurements,
  this pass also supports terminal measurement processes that are performed on wires, instead of
  mid-circuit measurement results.

* Executing circuits that are compiled with :func:`pennylane.transforms.to_ppr`,
  :func:`pennylane.transforms.commute_ppr`, :func:`pennylane.transforms.ppr_to_ppm`,
  :func:`pennylane.transforms.merge_ppr_ppm`, :func:`pennylane.transforms.reduce_t_depth`,
  and :func:`pennylane.transforms.decompose_arbitrary_ppr` is now possible with the `lightning.qubit` device and
  with program capture enabled (:func:`pennylane.capture.enable`).
  [(#2348)](https://github.com/PennyLaneAI/catalyst/pull/2348)
  [(#2389)](https://github.com/PennyLaneAI/catalyst/pull/2389)
  [(#2390)](https://github.com/PennyLaneAI/catalyst/pull/2390)
  [(#2413)](https://github.com/PennyLaneAI/catalyst/pull/2413)
  [(#2414)](https://github.com/PennyLaneAI/catalyst/pull/2414)
  [(#2424)](https://github.com/PennyLaneAI/catalyst/pull/2424)
  [(#2443)](https://github.com/PennyLaneAI/catalyst/pull/2443)
  [(#2460)](https://github.com/PennyLaneAI/catalyst/pull/2460)
  [(#2639)](https://github.com/PennyLaneAI/catalyst/pull/2639)

  Previously, circuits compiled with these transforms were only inspectable via
  :func:`pennylane.specs` and :func:`catalyst.draw`. Now, such circuits can be executed:

  ```python
  import pennylane as qp

  @qp.qjit(capture=True)
  @qp.transforms.decompose_arbitrary_ppr
  @qp.transforms.to_ppr
  @qp.qnode(qp.device("lightning.qubit", wires=3))
  def circuit():
      qp.PauliRot(0.123, pauli_word="XXY", wires=[0, 1, 2])
      qp.pauli_measure("XYZ", wires=[0, 1, 2])
      return qp.probs([0, 1])
  ```

  ```
  >>> print(circuit())
  [0.5 0.  0.  0.5]
  ```

* Added support for ``PauliRot`` and ``PauliMeasure`` execution on the `null.qubit` device, which enables
  runtime resource tracking for those operations.
  [(#2627)](https://github.com/PennyLaneAI/catalyst/pull/2627)

* A new optimization pass has been added to reduce the number of instructions in a quantum program,
  `--merge-global-phase`, which safely combines global phase instructions for each region in the
  program. The xDSL version written in Python has been removed.
  [(#2604)](https://github.com/PennyLaneAI/catalyst/pull/2604)

* A new `~.CompilationPass` class has been added that abstracts away compiler-level details for
  seamless compilation pass creation. Used in tandem with :func:`~.compiler_transform`, compilation
  passes can be created entirely in Python and used on QNodes within a :func:`~.qjit`'d workflow.
  [(#2211)](https://github.com/PennyLaneAI/catalyst/pull/2211)

* Added a scalable MLIR resource analysis pass (`resource-analysis`) that counts quantum
  operations across the `quantum`, `qec`, and `mbqc` dialects. The analysis is implemented as a
  cacheable MLIR analysis class (`ResourceAnalysis`) that other transformation passes can query
  via `getAnalysis<ResourceAnalysis>()`, avoiding redundant recomputation.
  [(#2479)](https://github.com/PennyLaneAI/catalyst/pull/2479)
  [(#2675)](https://github.com/PennyLaneAI/catalyst/pull/2675)
  [(#2695)](https://github.com/PennyLaneAI/catalyst/pull/2695)
  [(#2755)](https://github.com/PennyLaneAI/catalyst/pull/2755)

  ```bash
  quantum-opt --resource-analysis='output-json=true' input.mlir
  quantum-opt --resource-analysis -mlir-pass-statistics input.mlir
  ```

* The `diagonalize-final-measurements` xDSL pass now accepts the optional keyword argument ``supported_base_obs``. The kwarg``to_eigvals`` is now also included in the call signature for compatibility with the tape transform, but this kwarg is unused and can only take its default value, `False`.
  [(#2517)](https://github.com/PennyLaneAI/catalyst/pull/2517)

  These pass options can be applied as follows in the example below:

  ```python
  import pennylane as qp

  def diagonalize_measurements_setup_inputs(
      to_eigvals: bool = False, supported_base_obs: tuple[str] = ("PauliX",)
  ):
      return (), {"to_eigvals": to_eigvals, "supported_base_obs": supported_base_obs}

  diagonalize_measurements = qp.transform(
      pass_name="diagonalize-final-measurements", setup_inputs=diagonalize_measurements_setup_inputs
  )

  dev = qp.device("null.qubit", wires=4)
  @qp.qjit(target="mlir", keep_intermediate=True)
  @diagonalize_measurements(supported_base_obs=('PauliX',))
  @qp.qnode(dev, shots=1000)
  def circuit():
      qp.CRX(0.1, wires=[0, 1])
      return qp.expval(qp.X(0))

  circuit()
  ```

* The `diagonalize-final-measurements` xDSL pass now includes an observable-commutativity check and
  raises an error if non-commuting terms are encountered. The check is applied to each `qnode` in
  the IR (that is, a `func.func` op with a `quantum.node` attribute). If the measurement contains
  only Pauli or Hadamard observables, the *qubit-wise commutativity* (QWC) check is applied.
  Otherwise, the more strict *non-overlapping observable* check is applied.
  [(#2538)](https://github.com/PennyLaneAI/catalyst/pull/2538)
  [(#2633)](https://github.com/PennyLaneAI/catalyst/pull/2633)

* The `diagonalize-final-measurements` xDSL pass is now available as a builtin pass accessible from the Catalyst frontend as `catalyst.passes.diagonalize_measurements`.
  [(#2630)](https://github.com/PennyLaneAI/catalyst/pull/2630)

* Added a pass to compute resource metrics of functions marked with the `target_gate` attribute,
  effectively filtering for decomposition rules in the MLIR-native decomposition framework.
  [(#2539)](https://github.com/PennyLaneAI/catalyst/pull/2539)

  ```bash
  quantum-opt input.mlir -register-decomp-rule-resource
  ```

  Input:

  ```mlir
  func.func @decomp_rule() attributes {target_gate="CustomGate"}  {
      %0 = quantum.alloc( 2) : !quantum.reg
      %1 = quantum.extract %0[ 0] : !quantum.reg -> !quantum.bit
      %2 = quantum.extract %0[ 1] : !quantum.reg -> !quantum.bit
      %3 = quantum.custom "Hadamard"() %1 : !quantum.bit
      %4 = quantum.custom "T"() %3 : !quantum.bit
      %5 = quantum.custom "S"() %2 : !quantum.bit
      %6:2 = quantum.custom "CNOT"() %4, %5 : !quantum.bit, !quantum.bit
      %7 = quantum.insert %0[ 0], %6#0 : !quantum.reg, !quantum.bit
      %8 = quantum.insert %7[ 1], %6#1 : !quantum.reg, !quantum.bit
      quantum.dealloc %8 : !quantum.reg
      return
  }
  ```

  Output:

  ```mlir
  func.func @decomp_rule() attributes {resources = {measurements = {}, num_alloc_qubits = 2 : i64, num_arg_qubits = 0 : i64, num_qubits = 2 : i64, operations = {"CNOT(2)" = 1 : i64, "Hadamard(1)" = 1 : i64, "S(1)" = 1 : i64, "T(1)" = 1 : i64}}, target_gate = "CustomGate"}  {
      %0 = quantum.alloc( 2) : !quantum.reg
      %1 = quantum.extract %0[ 0] : !quantum.reg -> !quantum.bit
      %2 = quantum.extract %0[ 1] : !quantum.reg -> !quantum.bit
      %3 = quantum.custom "Hadamard"() %1 : !quantum.bit
      %4 = quantum.custom "T"() %3 : !quantum.bit
      %5 = quantum.custom "S"() %2 : !quantum.bit
      %6:2 = quantum.custom "CNOT"() %4, %5 : !quantum.bit, !quantum.bit
      %7 = quantum.insert %0[ 0], %6#0 : !quantum.reg, !quantum.bit
      %8 = quantum.insert %7[ 1], %6#1 : !quantum.reg, !quantum.bit
      quantum.dealloc %8 : !quantum.reg
      return
  }
  ```

* Added a cache of pre-compiled PennyLane built-in decomposition rules for use with the C++ graph
  decomposition system.
  [(#2531)](https://github.com/PennyLaneAI/catalyst/pull/2531)
  [(#2619)](https://github.com/PennyLaneAI/catalyst/pull/2619)
  [(#2713)](https://github.com/PennyLaneAI/catalyst/pull/2713)
  [(#2749)](https://github.com/PennyLaneAI/catalyst/pull/2749)

* Decomposition rules are lowered as private functions (instead of public).
  [(#2658)](https://github.com/PennyLaneAI/catalyst/pull/2658)
  [(#2660)](https://github.com/PennyLaneAI/catalyst/pull/2660)

* A new optimization pass has been added to reduce the number of instructions in a quantum program,
  `--combine-global-phase`, which safely combines global phase instructions for each region in the
  program. The xDSL version written in Python has been removed.
  [(#2604)](https://github.com/PennyLaneAI/catalyst/pull/2604)
  [(#2777)](https://github.com/PennyLaneAI/catalyst/pull/2777)

* A new native-MLIR graph-based decomposition framework is now available. This system
  migrates the graph-decomposition logic from Python into the Catalyst compiler as a
  high-performance C++ library (`DecompGraphSolver`), enabling the compiler to
  automatically find optimal decomposition paths from source gates to a target gate set.
  PennyLane's built-in decompositon rules are pre-compiled to MLIR bytecode and
  is utilized in this new framework to enable fast rule loading at compile time.
  [(#2552)](https://github.com/PennyLaneAI/catalyst/pull/2552)
  [(#2568)](https://github.com/PennyLaneAI/catalyst/pull/2568)
  [(#2578)](https://github.com/PennyLaneAI/catalyst/pull/2578)
  [(#2711)](https://github.com/PennyLaneAI/catalyst/pull/2711)
  [(#2765)](https://github.com/PennyLaneAI/catalyst/pull/2765)
  [(#2722)](https://github.com/PennyLaneAI/catalyst/pull/2722)

  The framework is interfaced with a new `graph_decomposition` pass decorator
  with key capabilities:
  - Multiple graph-based decomposition transformation at MLIR
  - Weighted target gate sets for the graph solver to minimize the total decomposition cost
  - Optional `alt_decomps` to define additional rules for (user-defined) operators
  - Optional `fixed_decomps` to pin a specific decomposition rule for an operator

  ``` python
  import pennylane as qp
  import pennylane.numpy as np

  from catalyst import qjit
  from catalyst.jax_primitives import decomposition_rule
  from catalyst.passes import cancel_inverses, graph_decomposition, merge_rotations


  @decomposition_rule(op_type=qp.PauliX)
  def x_to_rx(wire: int):
      qp.RX(np.pi, wire)


  @decomposition_rule(op_type=qp.PauliY)
  def y_to_ry(wire: int):
      qp.RY(np.pi, wire)


  @decomposition_rule(op_type=qp.Hadamard)
  def h_to_rx_ry(wire: int):
      qp.RX(np.pi / 2, wire)
      qp.RY(np.pi / 2, wire)


  @qjit(capture=True)
  @graph_decomposition(gate_set={qp.Rot})
  @merge_rotations
  @graph_decomposition(
      gate_set={qp.RX: 1.0, qp.RY: 1.0, qp.Rot: 5.0},
      fixed_decomps={qp.PauliX: x_to_rx, qp.PauliY: y_to_ry},
      alt_decomps={qp.H: [h_to_rx_ry]},
  )
  @cancel_inverses
  @qp.qnode(qp.device("lightning.qubit", wires=2))
  def circuit(x: float, y: float):
      qp.H(0)
      qp.H(0)
      qp.RX(x, wires=0)
      qp.PauliX(0)
      qp.RY(y, wires=0)
      qp.PauliY(0)
      qp.RY(x + y, wires=0)

      # register custom decomposition rules, required
      # when using the decomposition_rule decorator
      x_to_rx(int)
      y_to_ry(int)
      h_to_rx_ry(int)

      return qp.state()
  ```

  ``` pycon
  >>> print(qp.specs(circuit, level="device")(1.23, 4.56).resources.gate_types)
  {'Rot': 2}
  ```

* Added a pass to compute resource metrics of functions marked with the `target_gate` attribute,
  effectively filtering for decomposition rules in the MLIR-native decomposition framework.
  [(#2539)](https://github.com/PennyLaneAI/catalyst/pull/2539)

  ```bash
  quantum-opt input.mlir -register-decomp-rule-resource
  ```

  Input:

  ```mlir
  func.func @decomp_rule() attributes {target_gate="CustomGate"}  {
      %0 = quantum.alloc( 2) : !quantum.reg
      %1 = quantum.extract %0[ 0] : !quantum.reg -> !quantum.bit
      %2 = quantum.extract %0[ 1] : !quantum.reg -> !quantum.bit
      %3 = quantum.custom "Hadamard"() %1 : !quantum.bit
      %4 = quantum.custom "T"() %3 : !quantum.bit
      %5 = quantum.custom "S"() %2 : !quantum.bit
      %6:2 = quantum.custom "CNOT"() %4, %5 : !quantum.bit, !quantum.bit
      %7 = quantum.insert %0[ 0], %6#0 : !quantum.reg, !quantum.bit
      %8 = quantum.insert %7[ 1], %6#1 : !quantum.reg, !quantum.bit
      quantum.dealloc %8 : !quantum.reg
      return
  }
  ```

  Output:

  ```mlir
  func.func @decomp_rule() attributes {resources = {measurements = {}, num_alloc_qubits = 2 : i64, num_arg_qubits = 0 : i64, num_qubits = 2 : i64, operations = {"CNOT(2)" = 1 : i64, "Hadamard(1)" = 1 : i64, "S(1)" = 1 : i64, "T(1)" = 1 : i64}}, target_gate = "CustomGate"}  {
      %0 = quantum.alloc( 2) : !quantum.reg
      %1 = quantum.extract %0[ 0] : !quantum.reg -> !quantum.bit
      %2 = quantum.extract %0[ 1] : !quantum.reg -> !quantum.bit
      %3 = quantum.custom "Hadamard"() %1 : !quantum.bit
      %4 = quantum.custom "T"() %3 : !quantum.bit
      %5 = quantum.custom "S"() %2 : !quantum.bit
      %6:2 = quantum.custom "CNOT"() %4, %5 : !quantum.bit, !quantum.bit
      %7 = quantum.insert %0[ 0], %6#0 : !quantum.reg, !quantum.bit
      %8 = quantum.insert %7[ 1], %6#1 : !quantum.reg, !quantum.bit
      quantum.dealloc %8 : !quantum.reg
      return
  }
  ```

* Added a cache of pre-compiled PennyLane built-in decomposition rules for use with the C++ graph
  decomposition system.
  [(#2531)](https://github.com/PennyLaneAI/catalyst/pull/2531)
  [(#2619)](https://github.com/PennyLaneAI/catalyst/pull/2619)
  [(#2713)](https://github.com/PennyLaneAI/catalyst/pull/2713)
  [(#2749)](https://github.com/PennyLaneAI/catalyst/pull/2749)

* Decomposition rules are lowered as private functions (instead of public).
  [(#2658)](https://github.com/PennyLaneAI/catalyst/pull/2658)
  [(#2660)](https://github.com/PennyLaneAI/catalyst/pull/2660)

* OQD (Open Quantum Design) end-to-end pipeline is added to Catalyst.
  The pipeline supports compilation to LLVM IR using the `QJIT` constructor with `link=False`, enabling integration with ARTIQ's cross-compilation toolchain. The generated LLVM IR can be used with the internal `compile_to_artiq()` function from the third-party OQD repository to produce ARTIQ binaries.
  [(#2299)](https://github.com/PennyLaneAI/catalyst/pull/2299)

  see `frontend/test/test_oqd/oqd/test_oqd_artiq_llvmir.py` for more details.
  Note: This PR only covers LLVM IR generation; the `compile_to_artiq` function itself is not included.

  For example:

  ```python
  import os
  import numpy as np
  import pennylane as qp

  from catalyst import qjit
  from catalyst.third_party.oqd import OQDDevice, OQDDevicePipeline

  OQD_PIPELINES = OQDDevicePipeline(
      os.path.join("calibration_data", "device.toml"),
      os.path.join("calibration_data", "qubit.toml"),
      os.path.join("calibration_data", "gate.toml"),
      os.path.join("device_db", "device_db.json"),
  )

  oqd_dev = OQDDevice(
      backend="default",
      shots=4,
      wires=1
  )
  qp.capture.enable()

  # Compile to LLVM IR only
  @qp.qnode(oqd_dev)
  def circuit():
      x = np.pi / 2
      qp.RX(x, wires=0)
      return qp.counts(wires=0)

  compiled_circuit = QJIT(circuit, CompileOptions(link=False, pipelines=OQD_PIPELINES))

  # Compile to ARTIQ ELF
  artiq_config = {
      "kernel_ld": "/path/to/kernel.ld",
      "llc_path": "/path/to/llc",
      "lld_path": "/path/to/ld.lld",
  }

  output_elf_path = compile_to_artiq(compiled_circuit, artiq_config)
  # Output:
  # LLVM IR file written to: /path/to/circuit.ll
  # [ARTIQ] Generated ELF: /path/to/circuit.elf
  ```

* Mid-circuit measurements (`qp.measure`) are now supported on the OQD backend.
  A `qp.measure` call is lowered to an OpenAPL's `MeasurePulse` for fluorescence detection,
  which is executed by the trapped-ion hardware at runtime.
  [(#2508)](https://github.com/PennyLaneAI/catalyst/pull/2508)

  To enable mid-circuit measurement, add a `[[detection_beam]]` section and a
  `measurement_duration` field to the `gate.toml` calibration file:

  For example:

  ```toml
  measurement_duration = 1e-4  # seconds

  [[detection_beam]]
  rabi       = 62831853071.79586
  transition = "downstate_estate"
  detuning   = 0.0
  polarization = [1, 0, 0]
  wavevector   = [0, 1, 0]
  ```

  The following circuit will produce an OpenAPL program with a `MeasurePulse`:

  ```python
  oqd_dev = OQDDevice(backend="default", wires=1, openapl_file_name="out.json")

  @qjit(pipelines=OQD_PIPELINES)
  @qp.set_shots(10)
  @qp.qnode(oqd_dev)
  def circuit():
      qp.measure(wires=0)
      return qp.counts(wires=0)

  circuit()
  ```

  In addition, the MS gate beam lookup for this measurement testbench was redesigned:
  sideband beam parameters are now read directly from the calibration database instead of being
  computed from per-qubit phonon offsets.

<h3>Improvements 🛠</h3>

* `null.qubit` resource tracking is now able to track measurements and observables. This output
  is also reflected in `qp.specs`.
  [(#2446)](https://github.com/PennyLaneAI/catalyst/pull/2446)

* `mlir_specs` now supports MLIR passes which create multiple qnode entry points, such as `split-non-commuting` pass.
  When such passes are present, `mlir_specs` will return a list of resources with 1 per entrypoint.
  [(#2534)](https://github.com/PennyLaneAI/catalyst/pull/2534)

* ``ResourceAnalysis`` and ``RegisterDecompRuleResource`` passes now record the number of classical
  parameters for each gate alongside the wire count. The operation key format changes from
  `"GateName(nWires)"` to `"GateName(nWires,nParams)"`.
  [(#2755)](https://github.com/PennyLaneAI/catalyst/pull/2755)

* The default mcm_method for the finite-shots setting (dynamic one-shot) no longer silently falls
  back to single-branch statistics in most cases. Instead, an error message is raised pointing out
  alternatives, like explicitly selecting single-branch statistics.
  [(#2398)](https://github.com/PennyLaneAI/catalyst/pull/2398)

  Importantly, single-branch statistics only explores one branch of the MCM decision tree, meaning
  program outputs are typically probabilistic and statistics produced by measurement processes are
  conditional on the selected decision tree path.

* The :func:`~.passes.parity_synth` can now be invoked from the ``passes`` module.
  [(#2553)](https://github.com/PennyLaneAI/catalyst/pull/2553)
  [(#2784)](https://github.com/PennyLaneAI/catalyst/pull/2784)

  ```python
  import pennylane as qp
  import catalyst

  dev = qp.device("lightning.qubit", wires=2)

  @qp.qjit(capture=True)
  @catalyst.passes.parity_synth
  @qp.qnode(dev)
  def circuit(x: float, y: float, z: float):
      qp.CNOT((0, 1))
      qp.RZ(x, 1)
      qp.CNOT((0, 1))
      qp.RX(y, 1)
      qp.CNOT((1, 0))
      qp.RZ(z, 1)
      qp.CNOT((1, 0))
      return qp.state()

  Device: lightning.qubit
  Device wires: 2
  Shots: Shots(total=None)
  Level: device

  Wire allocations: 2
  Total gates: 5
  Gate counts:
  - RX: 1
  - RZ: 2
  - CNOT: 2
  Measurements:
  - state(all wires): 1
  Depth: 5
  ```

  Note as well that this compilation pass used to be named ``parity_synth_pass``.

* A warning is issued when gridsynth pass is called with epsilon smaller than 1e-6 due to potential precision error.
  [(#2625)](https://github.com/PennyLaneAI/catalyst/pull/2625)

* Added `capture` keyword argument to the `@qjit` decorator for per-function control over
  PennyLane's program capture frontend. This allows selective use of the new capture-based
  compilation pathway without affecting the global `qp.capture.enabled()` state. The parameter
  accepts `"global"` (default, defer to global state), `True` (force capture on), or `False`
  (force capture off). This enables safe testing and gradual migration to the capture system.
  [(#2457)](https://github.com/PennyLaneAI/catalyst/pull/2457)

* `qp.for_loop` now supports dynamic shapes with program capture `qjit(capture=True)`.
  [(#2603)](https://github.com/PennyLaneAI/catalyst/pull/2603/)
  [(#2651)](https://github.com/PennyLaneAI/catalyst/pull/2651)

* Added support for ``StatePrep`` kwargs ``pad_with`` and ``normalize`` with program capture enabled.
  [(#2620)](https://github.com/PennyLaneAI/catalyst/pull/2620)

* `abstracted_axes` now work with `qjit` and `capture=True`.
  [(#2655)](https://github.com/PennyLaneAI/catalyst/pull/2655)

* The `quantum.adjoint` operation can now take in multiple quantum values, allowing
  both qubits and registers, as opposed to constraining the operand to be a single quantum register.
  [(#2590)](https://github.com/PennyLaneAI/catalyst/pull/2590)
  [(#2610)](https://github.com/PennyLaneAI/catalyst/pull/2610)

* `qp.value_and_grad` can now be used with program capture `qp.qjit(capture=True)`.
  [(#2587)](https://github.com/PennyLaneAI/catalyst/pull/2587)

* Catalyst with program capture now supports device preprocessing. Currently, preprocessing transforms
  that do not have native MLIR or xDSL implementations will be replaced with empty transforms.
  [(#2557)](https://github.com/PennyLaneAI/catalyst/pull/2557)

* `qp.vjp`  and `qp.jvp` can now be used with Catalyst and program capture.
  [(#2279)](https://github.com/PennyLaneAI/catalyst/pull/2279)
  [(#2316)](https://github.com/PennyLaneAI/catalyst/pull/2316)

* Catalyst with program capture can now be used with the new `qp.templates.Subroutine` class and the associated
  `qp.capture.subroutine` upstreamed from `catalyst.jax_primitives.subroutine`.
  [(#2396)](https://github.com/PennyLaneAI/catalyst/pull/2396)
  [(#2493)](https://github.com/PennyLaneAI/catalyst/pull/2493)

* Graph decomposition with qjit now accepts `num_work_wires`, and lowers and decomposes correctly
  with the `decompose-lowering` pass and with `qp.transforms.decompose`.
  [(#2470)](https://github.com/PennyLaneAI/catalyst/pull/2470)

* Added support for `stopping_condition` in user-defined `qp.decompose` when capture is enabled with both graph enabled and disabled.
  [(#2486)](https://github.com/PennyLaneAI/catalyst/pull/2486)

* The tape transform :func:`~.device.decomposition.catalyst_decompose` now accepts the optional
  keyword arguments ``target_gates``, ``num_work_wires``, ``fixed_decomps``, and ``alt_decomps``,
  which all are passed to the used PennyLane decomposition function
  ``qp.devices.preprocess.decompose`` and used if the graph-based decomposition system is enabled.
  [(#2501)](https://github.com/PennyLaneAI/catalyst/pull/2501)

* Two new verifiers were added to the `quantum.paulirot` operation. They verify that the Pauli word
  length and the number of qubit operands are the same, and that all of the Pauli words are legal.
  [(#2405)](https://github.com/PennyLaneAI/catalyst/pull/2405)

* The quantum kernel abstraction in Catalyst's IR (a nested module operation with its own transform
  schedule and entry point and subroutine functions representing a PennyLane QNode) has been
  documented and equipped with additional verification. Transformation passes scheduled from the
  frontend must ensure, and can rely on, the presence of the `quantum.node` attribute to indicate
  which functions in the module represent a separate quantum execution (with device initialization,
  shots configuration, and set of measurement processes).
  [(#2483)](https://github.com/PennyLaneAI/catalyst/pull/2483)
  [(#2497)](https://github.com/PennyLaneAI/catalyst/pull/2497)
  [(#2597)](https://github.com/PennyLaneAI/catalyst/pull/2597)

* `catalyst.python_interface.utils.get_constant_from_ssa` can now extract constant values cast using
  `arith.index_cast`.
  [(#2542)](https://github.com/PennyLaneAI/catalyst/pull/2542)

* The `measurements_from_samples` pass no longer results in `nan`s and cryptic error messages when
  `shots` aren't set. Instead, an informative error message is raised.
  [(#2456)](https://github.com/PennyLaneAI/catalyst/pull/2456)

* A performance issue in the xDSL transform `measurements-from-samples` that was caused by the
  unrolling of a `for` loop for QNodes returning `probs` has been fixed.
  [(#2611)](https://github.com/PennyLaneAI/catalyst/pull/2611)

* The `measurements-from-samples` pass now diagonalizes observables automatically before converting
  to samples in the computational basis, removing the need to apply a diagonalization pass separately.
  This behaviour matches the behaviour of the tape transform `measurements_from_samples` in PennyLane.
  [(#2617)](https://github.com/PennyLaneAI/catalyst/pull/2617)

* The `measurements-from-samples` pass is refactored to follow the conventions for a qnode transform
  as they are described in `catalyst.python_interace.transforms.qnode-transform-guide.md`.
  [(#2605)](https://github.com/PennyLaneAI/catalyst/pull/2605)

* A more informative error message is now raised when a `measurements-from-samples` xDSL pass encounters a
  program with dyanamic shots.
  [(#2616)](https://github.com/PennyLaneAI/catalyst/pull/2616)

* The `measurements-from-samples` xDSL pass is extended to support tensor product observables.
  [(#2656)](https://github.com/PennyLaneAI/catalyst/pull/2656)

<h3>Breaking changes 💔</h3>

* The ``catalyst.python_interface.transforms.parity_synth_pass`` transform has been renamed to ``catalyst.python_interface.transforms.parity_synth``.
  [(#2553)](https://github.com/PennyLaneAI/catalyst/pull/2553)

* The ``-disentangle-CNOT`` and ``-disentangle-SWAP`` Catalyst CLI commands have been renamed to
  ``-disentangle-cnot`` and ``-disentangle-swap`` (all lower-case).
  [(#2546)](https://github.com/PennyLaneAI/catalyst/pull/2546)

* `catalyst.python_interface.inspection.draw` and `catalyst.python_interface.inspection.generate_mlir_graph` no longer
  accept QNodes as the input. Now, the input must always be a :class:`~.QJIT` object.
  [(#2542)](https://github.com/PennyLaneAI/catalyst/pull/2542)

* `catalyst.from_plxpr.register_transforms` as a way to access MLIR passes from Python has been removed in favour of the new unified transforms API. MLIR passes can be accessed from Python using `qp.transform(pass_name="some-pass-name")`.
  [(#2509)](https://github.com/PennyLaneAI/catalyst/pull/2509)
  [(#2680)](https://github.com/PennyLaneAI/catalyst/pull/2680)

* `catalyst.jax_primitives.subroutine` has been moved to `qp.capture.subroutine`.
  [(#2396)](https://github.com/PennyLaneAI/catalyst/pull/2396)

* The `StableHLO` dialect has been removed from Catalyst's Python interface module.
  Downstream users should now import StableHLO dialect definitions from `xdsl_jax.dialects.stablehlo` instead.
  [(#2588)](https://github.com/PennyLaneAI/catalyst/pull/2588)

* (Compiler integrators only) The versions of StableHLO/LLVM/Enzyme used by Catalyst have been updated.
  [(#2415)](https://github.com/PennyLaneAI/catalyst/pull/2415)
  [(#2416)](https://github.com/PennyLaneAI/catalyst/pull/2416)
  [(#2444)](https://github.com/PennyLaneAI/catalyst/pull/2444)
  [(#2445)](https://github.com/PennyLaneAI/catalyst/pull/2445)
  [(#2478)](https://github.com/PennyLaneAI/catalyst/pull/2478)

  * The StableHLO version has been updated to
  [v1.13.7](https://github.com/openxla/stablehlo/tree/v1.13.7).
  * The LLVM version has been updated to
  [commit 8f26458](https://github.com/llvm/llvm-project/tree/8f264586d7521b0e305ca7bb78825aa3382ffef7).
  * The Enzyme version has been updated to
  [v0.0.238](https://github.com/EnzymeAD/Enzyme/releases/tag/v0.0.238).

* When an integer argnums is provided to `catalyst.vjp`, a singleton dimension is now squeezed
  out. This brings the behaviour in line with that of `grad` and `jacobian`.
  [(#2279)](https://github.com/PennyLaneAI/catalyst/pull/2279)

* Dropped support for NumPy 1.x following its end-of-life. NumPy 2.0 or higher is now required.
  [(#2407)](https://github.com/PennyLaneAI/catalyst/pull/2407)

* The inlining pass has been removed from the default compilation pipeline.
  [(#2473)](https://github.com/PennyLaneAI/catalyst/pull/2473)

<h3>Deprecations 👋</h3>

<h3>Bug fixes 🐛</h3>

* Refactored all passes in `catalyst.passes.builtin_passes.py` to be `pennylane.transforms.core.Transform` objects
  rather than decorators. This allows them to be used as standard transforms, enabling full compatibility with
  `pennylane.CompilePipeline`.
  [(#2722)](https://github.com/PennyLaneAI/catalyst/pull/2722)

* Fixed a bug where the `work_wire_type` argument of `qp.ctrl` was silently dropped inside `@qjit` functions.
  The parameter is now threaded through `catalyst.ctrl`, `CtrlCallable`, `HybridCtrl`, and
  `ctrl_distribute`, with the default value being `"borrowed"`.
  [(#2710)](https://github.com/PennyLaneAI/catalyst/pull/2710)

* Fixed a bug where multiple `quantum.extract` operations from the same index were being created
  when there are multiple computational basis observables, named observables or Hermitian
  observables on that same wire index, when capture is not enabled.
  [(#2641)](https://github.com/PennyLaneAI/catalyst/pull/2641)
  [(#2646)](https://github.com/PennyLaneAI/catalyst/pull/2646)
  [(#2693)](https://github.com/PennyLaneAI/catalyst/pull/2693)

* :func:`~pennylane.adjoint` can now be used on subroutines with classical arguments.
  [(#2590)](https://github.com/PennyLaneAI/catalyst/pull/2590)

* Fixed a bug where the `catalyst` CLI tool would emit text when called with `--emit-bytecode`.
  [(#2596)](https://github.com/PennyLaneAI/catalyst/pull/2596)

* Fixed a bug where input array arguments could be mutated during execution when copied inputs
  were updated in-place. Entry-point arguments are now treated as non-writable during bufferization,
  preserving the expected immutability of user inputs.
  [(#2562)](https://github.com/PennyLaneAI/catalyst/pull/2562)

* Fixed a bug in the `split-non-commuting` pass where dead `NamedObsOp`s were left behind after
  erasing composite obs (`TensorOp`, `HamiltonianOp`).
  [(#2567)](https://github.com/PennyLaneAI/catalyst/pull/2567)

* Fix a bug where `draw_graph` failed at rendering measurements containing scalar products of observables.
  [(#2545)](https://github.com/PennyLaneAI/catalyst/pull/2545)

* Fixed a bug where the unified compiler would trigger a passed callback function 1 extra time for the initial pass level.
  [(#2528)](https://github.com/PennyLaneAI/catalyst/pull/2528)

* Fix a bug in the bind call function for `PCPhase` where the signature did not match what was
  expected in `jax_primitives`. `ctrl_qubits` was missing from positional arguments in previous signature.
  [(#2467)](https://github.com/PennyLaneAI/catalyst/pull/2467)

* Fix `CATALYST_XDSL_UNIVERSE` to correctly define the available dialects and transforms, allowing
  tools like `xdsl-opt` to work with Catalyst's custom Python dialects.
  [(#2471)](https://github.com/PennyLaneAI/catalyst/pull/2471)

* Fix symbolic adjoint support for control flow operation. This means operators who are the target
  of `qp.adjoint` but require decomposition can have decompositions with control flow in them,
  which would previously raise an error. Adjoint on functions is unaffected.
  [(#2667)](https://github.com/PennyLaneAI/catalyst/pull/2667)

* The adjoint lowering pass now supports `switch` operation as well. Previously, using
  `qp.adjoint` on a circuit containing a `switch` would raise a `CompileError`. The MLIR
  `--adjoint-lowering` pass has been updated to support this usage.
  [(#2691)](https://github.com/PennyLaneAI/catalyst/pull/2691)

* Fix a bug with the xDSL `ParitySynth` pass that caused failure when the QNode being transformed
  contained operations with regions.
  [(#2408)](https://github.com/PennyLaneAI/catalyst/pull/2408)

* Fix `replace_ir` for certain stages when used with gradients.
  [(#2436)](https://github.com/PennyLaneAI/catalyst/pull/2436)

* Restore the ability to differentiate multiple (expectation value) QNode results with the
  adjoint-differentiation method.
  [(#2428)](https://github.com/PennyLaneAI/catalyst/pull/2428)

* Fixed the angle conversion when lowering `pbc.ppr` and `pbc.ppr.arbitrary` operations to
  `__catalyst__qis__PauliRot` runtime calls. The PPR rotation angle is now correctly multiplied
  by 2 to match the PauliRot convention (`PauliRot(φ) == PPR(φ/2)`).
  [(#2414)](https://github.com/PennyLaneAI/catalyst/pull/2414)

* Fixed the `catalyst` CLI tool silently listening to stdin when run without an input file, even when given flags like `--list-passes` that should override this behaviour.
  [(2447)](https://github.com/PennyLaneAI/catalyst/pull/2447)

* Fixing incorrect lowering of PPM into CAPI calls when the PPM is in the negative basis.
  [(#2422)](https://github.com/PennyLaneAI/catalyst/pull/2422)

* Fixed the GlobalPhase discrepancies when executing gridsynth in the PPR basis.
  [(#2433)](https://github.com/PennyLaneAI/catalyst/pull/2433)

* Fixed incorrect decomposition of negative PPR (Pauli Product Rotation) operations in the
  `decompose-clifford-ppr` and `decompose-non-clifford-ppr` passes. The rotation sign is now
  correctly flipped when decomposing negative rotation kinds (e.g., `-π/4` from adjoint gates
  like `T†` or `S†`) to PPM (Pauli Product Measurement) operations.
  [(#2454)](https://github.com/PennyLaneAI/catalyst/pull/2454)

* Fixed incorrect global phase when lowering CNOT gates into PPR/PPM operations.
  [(#2459)](https://github.com/PennyLaneAI/catalyst/pull/2459)

* Fixed a bug where the Catalyst measurement primitive returning a boolean type as the measurement
  result was incorrectly replacing the PennyLane measurement primitive, whose measurement
  result is integer type, during the PLxPR conversion.
  [(#2582)](https://github.com/PennyLaneAI/catalyst/pull/2582)

<h3>Internal changes ⚙️</h3>

* The compiler pipeline definitions now have a single source of truth. Previously, pipeline and
  pass sequences were duplicated between the frontend (`frontend/catalyst/pipelines.py`) and the
  compiler (`mlir/lib/Driver/Pipelines.cpp`). Now, there is a unique definition that lives in
  `mlir/include/Driver/DefaultPipelines.h` and is exposed to the frontend via a `default_pipelines`
  nanobind extension module. This module is built during the MLIR compilation phase and discovered
  at runtime.
  [(#2259)](https://github.com/PennyLaneAI/catalyst/pull/2259)
  [(#2733)](https://github.com/PennyLaneAI/catalyst/pull/2733)

* Additional integration tests have been added for the pass-by-pass version of `qp.specs`.
  [(#2690)](https://github.com/PennyLaneAI/catalyst/pull/2690/)

* Removes unnessary registrations for the various gradient primitives in `from_plxpr` when we
  are able to just inherit the base behaviour from `PlxprInterpreter`.
  [(#2706)](https://github.com/PennyLaneAI/catalyst/pull/2706)

* The legacy frontend no longer registers `qp.allocate()` and `qp.deallocate()` onto the qjit device
  capabilities, since dynamic qubit allocation is only implemented for the capture frontend.
  [(#2696)](https://github.com/PennyLaneAI/catalyst/pull/2696)

* Refactors `draw_graph` implementation to improve maintainability.
  [(#2659)](https://github.com/PennyLaneAI/catalyst/pull/2659)

* Bump `black` version to 26.3.1 to eliminate the vulnerability reported by dependabot.
  [(#2650)](https://github.com/PennyLaneAI/catalyst/pull/2650)

* Updated Catalyst's Catch2 dependency to v3.11.0.
  [(#2634)](https://github.com/PennyLaneAI/catalyst/pull/2634)

* `rtio.rpc` operation is added to the RTIO dialect for OQD. It represents a host RPC call triggered by the kernel, optionally carrying runtime arguments and supporting both synchronous and async modes. The op is lowered to rpc_send / rpc_recv LLVM calls (the ARTIQ RPC wire protocol). It is required by both AWG control (program_awg, awg_close) and measurement result collection (set_dataset, transfer_data).
  [(#2577)](https://github.com/PennyLaneAI/catalyst/pull/2577)

* Updated Catalyst's xDSL dependencies to `xdsl` 0.59.0 and `xdsl-jax` 0.5.0.
  [(#2591)](https://github.com/PennyLaneAI/catalyst/pull/2591)

* Added a optimized pathway to the xDSL `ApplyTransformSequencePass` so that it can schedule consecutive MLIR
  passes together rather than individually. This minimizes the number of round-trips between xDSL and MLIR, improving
  performance when several consecutive MLIR passes are used when there are also xDSL passes in the pipeline.
  [(#2592)](https://github.com/PennyLaneAI/catalyst/pull/2592)

* `draw_graph` now raises a more informative error when attempting to visualize an unsupported empty external function.
  [(#2559)](https://github.com/PennyLaneAI/catalyst/pull/2559)

* Catalyst internally uses the new unified transforms API rather than `PassPipelineWrapper`.
  [(#2525)](https://github.com/PennyLaneAI/catalyst/pull/2525)
  [(#2614)](https://github.com/PennyLaneAI/catalyst/pull/2614)
  [(#2647)](https://github.com/PennyLaneAI/catalyst/pull/2647)

* Added an `EmptyPass` MLIR pass that does not transform the program for debugging and standing in for
  unimplemented transforms.
  [(#2575)](https://github.com/PennyLaneAI/catalyst/pull/2575)

* The QNode lowering to MLIR now supports providing multiple named transform pipelines.
  [(#2556)](https://github.com/PennyLaneAI/catalyst/pull/2556)

* Both the MLIR and xDSL `ApplyTransformSequencePass` implementations have been updated to support interpreting multiple
  `transform.named_sequence` operations for a single transformer module.
  [(#2550)](https://github.com/PennyLaneAI/catalyst/pull/2550)

* Update nightly RC builds to be triggered by Lightning.
  [(#2491)](https://github.com/PennyLaneAI/catalyst/pull/2491)

* Updated integration tests to match changes to the PennyLane `qp.specs` frontend made in https://github.com/PennyLaneAI/pennylane/pull/9088 and https://github.com/PennyLaneAI/pennylane/pull/9091.
  [(#2513)](https://github.com/PennyLaneAI/catalyst/pull/2513)
  [(#2533)](https://github.com/PennyLaneAI/catalyst/pull/2533)

* The `prepare` operation from the PBC dialect in MLIR now implicitly allocates new qubits
  rather than requiring existing ones. This better suits our purposes for further lowering
  the PBC dialect.
  [(#2520)](https://github.com/PennyLaneAI/catalyst/pull/2520)

* Standardized the `QJITDevice.preprocess` signature to align with the base PennyLane Device API.
  * Removed the redundant `ctx` (EvaluationContext) argument from the preprocessing and decomposition pipelines. The parameter was unused and its removal simplifies the tracing data flow.
  * Decoupled `shots` from the `QJITDevice.preprocess` signature. Catalyst-specific shot configurations are now handled via `execution_config.device_options` to maintain API compatibility.
  [(#2524)](https://github.com/PennyLaneAI/catalyst/pull/2524)

* A new AI policy document is now applied across the PennyLaneAI organization for all AI contributions.
  [(#2488)](https://github.com/PennyLaneAI/catalyst/pull/2488)

* A new dialect `QRef` was created. This dialect is very similar to the existing `Quantum` dialect,
  but it is in reference semantics, whereas the existing `Quantum` dialect is in value semantics.
  [(#2320)](https://github.com/PennyLaneAI/catalyst/pull/2320)
  [(#2590)](https://github.com/PennyLaneAI/catalyst/pull/2590)
  [(#2492)](https://github.com/PennyLaneAI/catalyst/pull/2492)
  [(#2674)](https://github.com/PennyLaneAI/catalyst/pull/2674)
  [(#2642)](https://github.com/PennyLaneAI/catalyst/pull/2642)
  [(#2692)](https://github.com/PennyLaneAI/catalyst/pull/2692)
  [(#2721)](https://github.com/PennyLaneAI/catalyst/pull/2721)
  [(#2723)](https://github.com/PennyLaneAI/catalyst/pull/2723)
  [(#2758)](https://github.com/PennyLaneAI/catalyst/pull/2758)

  Unlike qubit (or qreg) SSA values in the `Quantum` dialect, a qubit (or qreg) reference SSA value
  in the `QRef` dialect is allowed to be used multiple times. The operands of gates and observables
  will be these qubit (or qreg) reference values.

  For example, in the following circuit, gates and observable ops take in the qubit reference
  they're acting on, and do not produce new qubit values.

  ```mlir
  func.func @expval_circuit() -> f64 {
      %a = qref.alloc(2) : !qref.reg<2>
      %q0 = qref.get %a[0] : !qref.reg<2> -> !qref.bit
      %q1 = qref.get %a[1] : !qref.reg<2> -> !qref.bit
      qref.custom "Hadamard"() %q0 : !qref.bit
      qref.custom "CNOT"() %q0, %q1 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %q0 : !qref.bit
      %obs = qref.namedobs %q1 [ PauliX] : !quantum.obs
      %expval = quantum.expval %obs : f64
      qref.dealloc %a : !qref.reg<2>
      return %expval : f64
  }
  ```

  Notice that qubit reference values are reusable.

  An MLIR program in the `QRef` dialect can be converted to the `Quantum` dialect with the new pass
  `--convert-to-value-semantics`, optionally followed by `--canonicalize` for removing pairs of
  neighboring inverse `quantum.extract` and `quantum.insert` operations.

  Apart from those in the `Quantum` dialect, reference semantics operations for their value
  semantics counterparts in the `MBQC` dialect were also added.

* A new pass `--verify-no-quantum-use-after-free` was added to the new `QRef` dialect, to verify
  that there are no uses of quantum values after they have been deallocated.
  [(#2674)](https://github.com/PennyLaneAI/catalyst/pull/2674)

* Removed the `condition` operand from `pbc.ppm` (Pauli Product Measurement) operations.
  Conditional PPR decompositions in the `decompose-clifford-ppr` pass now emit the
  measurement logic inside an `scf.if` region rather than propagating the condition
  to inner PPM ops.
  [(#2511)](https://github.com/PennyLaneAI/catalyst/pull/2511)

* The operands and assembly format of several PBC operations have been updated for clarity and
  improved functionality.
  [(#2637)](https://github.com/PennyLaneAI/catalyst/pull/2637)

* A :class:`~.QJIT`'s ``compile`` method can now be used to run MLIR compilation without having
  to generate LLVM IR and object code. Use with ``CompileOptions(lower_to_llvm=False, link=False)``.
  [(#2599)](https://github.com/PennyLaneAI/catalyst/pull/2599)

* Update `mlir_specs` to account for new `marker` functionality in PennyLane.
  [(#2464)](https://github.com/PennyLaneAI/catalyst/pull/2464)

* The QEC (Quantum Error Correction) dialect has been renamed to PBC (Pauli-Based Computation)
  across the entire codebase. This includes the MLIR dialect (`pbc.*` -> `pbc.*`), C++ namespaces
  (`catalyst::pbc` -> `catalyst::pbc`), Python bindings, compiler passes (e.g.,
  `lower-pbc-init-ops` -> `lower-pbc-init-ops`, `convert-pbc-to-llvm` -> `convert-pbc-to-llvm`),
  qubit type (`!quantum.bit<pbc>` -> `!quantum.bit<pbc>`), and all associated file and directory
  names. The rename better reflects the dialect's purpose as a representation for Pauli-Based
  Computation rather than general quantum error correction.
  [(#2482)](https://github.com/PennyLaneAI/catalyst/pull/2482)
  [(#2485)](https://github.com/PennyLaneAI/catalyst/pull/2485)

* Updated the integration tests for `qp.specs` to get coverage for new features
  [(#2448)](https://github.com/PennyLaneAI/catalyst/pull/2448)

* The xDSL :class:`~catalyst.python_interface.Quantum` dialect has been split into multiple files
  to structure operations and attributes more concretely.
  [(#2434)](https://github.com/PennyLaneAI/catalyst/pull/2434)

* `catalyst.python_interface.xdsl_universe.XDSL_UNIVERSE` has been renamed to `CATALYST_XDSL_UNIVERSE`.
  [(#2435)](https://github.com/PennyLaneAI/catalyst/pull/2435)

* The private helper `_extract_passes` of `qfunc.py` uses `BoundTransform.tape_transform`
  instead of the deprecated `BoundTransform.transform`.
  `jax_tracer.py` and `tracing.py` also updated accordingly.
  [(#2440)](https://github.com/PennyLaneAI/catalyst/pull/2440)

* Autograph is no longer applied to decomposition rules based on whether it's applied to the workflow itself.
  Operator developers now need to manually apply autograph to decomposition rules when needed.
  [(#2421)](https://github.com/PennyLaneAI/catalyst/pull/2421)

* The quantum dialect MLIR and TableGen source has been refactored to place type and attribute
  definitions in separate file scopes.
  [(#2329)](https://github.com/PennyLaneAI/catalyst/pull/2329)

* Improve speed and reliability of xDSL inspection functionality by only running the necessary
  compilation steps if the QJIT object does not already have an MLIR representation.
  [(#2598)](https://github.com/PennyLaneAI/catalyst/pull/2598)

* Added lowering of `pbc.ppm`, `pbc.ppr`, and `quantum.paulirot` to the runtime CAPI and QuantumDevice C++ API.
  [(#2348)](https://github.com/PennyLaneAI/catalyst/pull/2348)
  [(#2413)](https://github.com/PennyLaneAI/catalyst/pull/2413)
  [(#2683)](https://github.com/PennyLaneAI/catalyst/pull/2683)

* A new compiler pass, `unroll-conditional-ppr-ppm`, has been added to convert conditional or
  multiplexed Pauli-product rotations and measurements into their basic versions nested inside
  conditionals (from the SCF dialect). Note that this is not needed for the standard execution
  pipeline.
  [(#2390)](https://github.com/PennyLaneAI/catalyst/pull/2390)

* Increased format size for the `--mlir-timing` flag, displaying more decimals for better timing precision.
  [(#2423)](https://github.com/PennyLaneAI/catalyst/pull/2423)

* Added global phase tracking to the `to-ppr` compiler pass. When converting quantum gates to
  Pauli Product Rotations (PPR), the pass now emits `quantum.gphase` operations to preserve
  global phase correctness.
  [(#2419)](https://github.com/PennyLaneAI/catalyst/pull/2419)

* The upstream MLIR `Test` dialect is now available via the `catalyst` command line tool.
  [(#2417)](https://github.com/PennyLaneAI/catalyst/pull/2417)

* Removing some previously added guardrails that were in place due to a bug in dynamic allocation
  that is now fixed.
  [(#2427)](https://github.com/PennyLaneAI/catalyst/pull/2427)

* A new compiler pass `lower-pbc-init-ops` has been added to lower PBC initialization operations
  to Quantum dialect operations. This pass converts `pbc.prepare` to `quantum.custom` and
  `pbc.fabricate` to `quantum.alloc_qb` + `quantum.custom`, enabling runtime execution of
  PBC state preparation operations.
  [(#2424)](https://github.com/PennyLaneAI/catalyst/pull/2424)

* A new compiler pass `split-to-single-terms` has been added for QNode functions containing
  Hamiltonian expectation values. It facilitates execution on devices that don't natively support expectation values of sums of observables by splitting them into individual leaf observable expvals.
  [(#2441)](https://github.com/PennyLaneAI/catalyst/pull/2441)

  Consider the following example:

  ```python
  import pennylane as qp
  from catalyst import qjit
  from catalyst.passes import apply_pass

  @qjit
  @apply_pass("split-to-single-terms")
  @qp.qnode(qp.device("lightning.qubit", wires=3))
  def circuit():
      # Hamiltonian H = Z(0) @ X(1) + 2*Y(2)
      return qp.expval(qp.Z(0) @ qp.X(1) + 2 * qp.Y(2))
  ```

  The pass transforms the function by splitting the Hamiltonian into individual observables:

  **Before:**

  ```mlir
  func @circ1(%arg0) -> (tensor<f64>) {qnode} {
      // ... quantum ops ...
      // Z(0) @ X(1)
      %obs0 = quantum.namedobs %qubit0[ PauliZ] : !quantum.obs
      %obs1 = quantum.namedobs %qubit1[ PauliX] : !quantum.obs
      %T0 = quantum.tensor %obs0, %obs1 : !quantum.obs

      // Y(2)
      %obs2 = quantum.namedobs %qubit2[ PauliY] : !quantum.obs
      %H0 = quantum.hamiltonian(%8 : tensor<1xf64>) %obs2 : !quantum.obs

      %H = quantum.hamiltonian(%coeffs_2xf64) %T0, %H0 : !quantum.obs
      %result = quantum.expval %H : f64   // H = c_0 * (Z @ X) + c_1 * Y

      // ... to tensor ...
      %tensor_result = tensor.from_elements %result : tensor<f64>
      return %tensor_result
  }
  ```

  **After:**

  ```mlir
  func @circ1.quantum() -> (tensor<f64>, tensor<f64>) {qnode} {
      // ... quantum ops ...
      %expval0 = quantum.expval %T0 : f64
      %expval1 = quantum.expval %obs2 : f64

      // ... to tensor ...
      %tensor0 = tensor.from_elements %expval0 : tensor<f64>
      %tensor1 = tensor.from_elements %expval1 : tensor<f64>
      return %tensor0, %tensor1
  }
  func @circ1(%arg0) -> (tensor<f64>, tensor<f64>) {
      // ... setup ...
      %call:2 = call @circ1.quantum()

      // Extract coefficients and compute weighted sum
      %result = c0 * %call#0 + c1 * %call#1
      return %result
  }
  ```

* A new compiler pass `split-non-commuting` has been added for QNode functions that measure
  non-commuting observables. It facilitates execution on devices that don't natively support
  measuring multiple non-commuting observables simultaneously by splitting them into separate
  circuit executions. The pass supports a `grouping_strategy` option: the default (`None`) assigns
  each observable to its own group, while `"wires"` groups observables on non-overlapping wires into
  the same execution, reducing the total number of generated circuits. Duplicate observables are
  measured only once and their results are reused.
  [(#2437)](https://github.com/PennyLaneAI/catalyst/pull/2437)
  [(#2657)](https://github.com/PennyLaneAI/catalyst/pull/2657)

  **Relationship to `split-to-single-terms`:** The `split-non-commuting` pass internally runs
  `split-to-single-terms` first when processing Hamiltonian expectation values. The
  `split-to-single-terms` pass decomposes a Hamiltonian (sum of observables) into individual
  leaf observables and computes the weighted sum in post-processing by running the circuit
  once. By contrast, `split-non-commuting` goes further: it splits non-commuting observables
  into multiple groups and runs the circuit once per group

  Consider the following example:
  ```python
  import pennylane as qp
  from catalyst import qjit

  @qjit
  @qp.transform(pass_name="split-non-commuting")(grouping_strategy="wires")
  @qp.qnode(qp.device("lightning.qubit", wires=3))
  def circuit():
      # Hamiltonian H = Z(0) + 2 * X(0) + 3 * Identity
      return qp.expval(qp.Z(0) + 2 * qp.X(0) + 3 * qp.Identity(2))
  ```

  The pass first runs `split-to-single-terms` to decompose the Hamiltonian, then splits
  non-commuting observables into separate groups. Shots are distributed among groups using
  integer division (rounded down); e.g., 100 shots with 3 groups yields 33 shots per group.

  **Before:**
  ```mlir
  func @circ1(%arg0) -> (tensor<f64>) {qnode} {
      %shots = arith.constant 100
      quantum.device shots(%shots)
      // ... quantum ops ...
      %H = quantum.hamiltonian(%coeffs) %T0, %obs2 : !quantum.obs
      %result = quantum.expval %H : f64
      return %tensor_result
  }
  ```

  **After:**
  ```mlir
  func @circ1() -> (tensor<f64>) {
      %r0, %r1 = call @circ1.quantum.group.0()  // expval(Z), 1.0
      %r2 = call @circ1.quantum.group.1()  // expval(X)
      // Weighted sum: 1 * r0 + 3 * r1 + 2 * r2
      return %result
  }
  func @circ1.quantum.group.0() -> (tensor<f64>, tensor<f64>) {qnode} {
      // ... quantum ops ...
      %shots = arith.constant 100
      %num_group = arith.constant 3 : i64
      // Shots are divided among groups via integer division (rounded down)
      %new_shots = arith.divsi %shots, %num_group
      quantum.device shots(%new_shots)
      %obs = quantum.namedobs %out_qubits[ PauliZ] : !quantum.obs
      %r0 = quantum.expval %obs

      // expval(Identity) be simplified to one
      %one = arith.constant dense<1.000000e+00>
      return %r0, %one
  }
  func @circ1.quantum.group.1() -> tensor<f64> {qnode} {
      // ... quantum ops, single expval ...
  }
  ```

* A new MLIR op, `MCMObsOp`, is defined as a pseudo-observable of mid-circuit measurements for use in
  measurement processes. It is also registered in xDSL.
  [(#2458)](https://github.com/PennyLaneAI/catalyst/pull/2458)
  [(#2536)](https://github.com/PennyLaneAI/catalyst/pull/2536)

* An experimental *QEC Logical* MLIR dialect has been added. An equivalent xDSL dialect has also
  been added for compatibility with the Python interface to Catalyst.
  [(#2512)](https://github.com/PennyLaneAI/catalyst/pull/2512)
  [(#2535)](https://github.com/PennyLaneAI/catalyst/pull/2535)
  [(#2543)](https://github.com/PennyLaneAI/catalyst/pull/2543)
  [(#2544)](https://github.com/PennyLaneAI/catalyst/pull/2544)
  [(#2547)](https://github.com/PennyLaneAI/catalyst/pull/2547)
  [(#2549)](https://github.com/PennyLaneAI/catalyst/pull/2549)
  [(#2665)](https://github.com/PennyLaneAI/catalyst/pull/2665)

* An experimental *QEC Physical* MLIR dialect has been added. An equivalent xDSL dialect has also
  been added for compatibility with the Python interface to Catalyst.
  [(#2519)](https://github.com/PennyLaneAI/catalyst/pull/2519)
  [(#2537)](https://github.com/PennyLaneAI/catalyst/pull/2537)
  [(#2563)](https://github.com/PennyLaneAI/catalyst/pull/2563)
  [(#2571)](https://github.com/PennyLaneAI/catalyst/pull/2571)
  [(#2572)](https://github.com/PennyLaneAI/catalyst/pull/2572)
  [(#2574)](https://github.com/PennyLaneAI/catalyst/pull/2574)
  [(#2576)](https://github.com/PennyLaneAI/catalyst/pull/2576)
  [(#2673)](https://github.com/PennyLaneAI/catalyst/pull/2673)
  [(#2768)](https://github.com/PennyLaneAI/catalyst/pull/2768)

* An experimental pass to convert `qecl.noise` operations in the *QEC Logical* layer to subroutine calls in the *QEC Phyiscal* layer.
  [(#2678)](https://github.com/PennyLaneAI/catalyst/pull/2678)

* A new, experimental compiler pass `convert-quantum-to-qecl` has been added to lower operations
  from the `quantum` dialect into the QEC Logical (`qecl`) dialect.
  [(#2589)](https://github.com/PennyLaneAI/catalyst/pull/2589)

* An experimental compiler pass `inject-noise-to-qecl` has been added to inject noise operations
  into the QEC Logical (`qecl`) layer to validate QEC protocols under development.
  [(#2705)](https://github.com/PennyLaneAI/catalyst/pull/2705)

* A new, experimental compiler pass `convert-qecl-to-qecp` has been added to lower operations
  from the QEC Logical (`qecl`) dialect into the QEC Physical (`qecp`) dialect.
  [(#2697)](https://github.com/PennyLaneAI/catalyst/pull/2697)
  [(#2714)](https://github.com/PennyLaneAI/catalyst/pull/2714)
  [(#2716)](https://github.com/PennyLaneAI/catalyst/pull/2716)
  [(#2737)](https://github.com/PennyLaneAI/catalyst/pull/2737)
  [(#2731)](https://github.com/PennyLaneAI/catalyst/pull/2731)
  [(#2735)](https://github.com/PennyLaneAI/catalyst/pull/2735)
  [(#2754)](https://github.com/PennyLaneAI/catalyst/pull/2754)

* A number of deprecation warnings have been fixed in the compiler python interface.
  [(#2621)](https://github.com/PennyLaneAI/catalyst/pull/2621)

* Python `dataclass` objects can now be converted to MLIR dictionary attributes, allowing them to be
  used as xDSL pass options, for example.
  [(#2719)](https://github.com/PennyLaneAI/catalyst/pull/2719)

<h3>Documentation 📝</h3>

* The `qp` alias as in `import pennylane as qp` has been updated to `qp` in our source code and documentation.
  [(#2764)](https://github.com/PennyLaneAI/catalyst/pull/2764)
  [(#2763)](https://github.com/PennyLaneAI/catalyst/pull/2763)
  [(#2748)](https://github.com/PennyLaneAI/catalyst/pull/2748)
  [(#2746)](https://github.com/PennyLaneAI/catalyst/pull/2746)
  [(#2745)](https://github.com/PennyLaneAI/catalyst/pull/2745)
  [(#2744)](https://github.com/PennyLaneAI/catalyst/pull/2744)
  [(#2743)](https://github.com/PennyLaneAI/catalyst/pull/2743)
  [(#2742)](https://github.com/PennyLaneAI/catalyst/pull/2742)
  [(#2741)](https://github.com/PennyLaneAI/catalyst/pull/2741)
  [(#2739)](https://github.com/PennyLaneAI/catalyst/pull/2739)
  [(#2738)](https://github.com/PennyLaneAI/catalyst/pull/2738)
  [(#2736)](https://github.com/PennyLaneAI/catalyst/pull/2736)
  [(#2715)](https://github.com/PennyLaneAI/catalyst/pull/2715)

* The "Compatibility with PennyLane transforms" section of the
  :doc:`Sharp bits and debugging tips <../dev/sharp_bits>` document has been updated to describe
  potential oddities that can be encountered when composing PennyLane transforms together.
  Additionally, some sharp bits listed were removed, as they are no longer sharp bits.
  [(#2662)](https://github.com/PennyLaneAI/catalyst/pull/2662)

* Docstrings for :func:`~.passes.disentangle_cnot` and :func:`~.passes.disentangle_swap` have been improved
  by using updated features for inspection and by calling them from the PennyLane frontend.
  [(#2546)](https://github.com/PennyLaneAI/catalyst/pull/2546)

* Typos and rendering issues in various docstrings in the :mod:`catalyst.passes` module were fixed.
  [(#2649)](https://github.com/PennyLaneAI/catalyst/pull/2649)

* Updated the Unified Compiler Cookbook to be compatible with the latest versions of PennyLane and Catalyst.
  [(#2406)](https://github.com/PennyLaneAI/catalyst/pull/2406)

* Updated the changelog and builtin_passes.py to link to <https://pennylane.ai/compilation/pauli-based-computation> instead.
  [(#2409)](https://github.com/PennyLaneAI/catalyst/pull/2409)

* Infrastructure has been put in place for features that are accessible from both PennyLane and
  Catalyst to have a single source of truth for documentation, which will provide a better overall
  experience when consulting our documentation.
  [(#2481)](https://github.com/PennyLaneAI/catalyst/pull/2481)
  [(#2629)](https://github.com/PennyLaneAI/catalyst/pull/2629)

  Several entry-points were added to ``setup.py`` for the Pauli-based computation compilation passes
  and the :func:`~.draw_graph` function. This allows for the ability to use Catalyst features from
  PennyLane directly (related: [(#9020)](https://github.com/PennyLaneAI/pennylane/pull/9020)) and
  for the documentation of those features to be accessible to both Catalyst and PennyLane, creating
  a single source of truth for such features.

  In addition, the documentation for all Pauli-based computation transforms has been updated to be
  more user-focused by showing examples with :func:`~.specs` and by calling the transforms from the
  PennyLane frontend.

<h3>Contributors ✍️</h3>

This release contains contributions from (in alphabetical order):
Ali Asadi,
Joey Carter,
Yushao Chen,
Isaac De Vlugt,
Marcus Edwards,
Lillian Frederiksen,
Sengthai Heng,
David Ittah,
Jeffrey Kam,
Joseph Lee,
Mehrdad Malekmohammadi,
River McCubbin,
Mudit Pandey,
Andrija Paurevic,
David D.W. Ren,
Shuli Shu,
Paul Haochen Wang,
David Wierichs,
Jake Zaia,
Hongsheng Zheng.
