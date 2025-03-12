# Release 0.11.0 (development release)

<h3>New features since last release</h3>

* Conversion Clifford+T gates to Pauli Product Rotation (PPR) and measurement to Pauli Product Measurement (PPM) are now available through the `to_ppr` pass transform.
  [(#1499)](https://github.com/PennyLaneAI/catalyst/pull/1499)
  [(#1551)](https://github.com/PennyLaneAI/catalyst/pull/1551)
  [(#1564)](https://github.com/PennyLaneAI/catalyst/pull/1564)

  Supported gate conversions:
    - H gate → PPR with (Z · X · Z)π/4
    - S gate → PPR with (Z)π/4
    - T gate → PPR with (Z)π/8
    - CNOT → PPR with (Z ⊗ X)π/4 · (Z ⊗ 1)−π/4 · (1 ⊗ X)−π/4

    Example: 
    ```python
        @qjit(keep_intermediate=True)
        @to_ppr
        @qml.qnode(dev)
        def circuit():
            qml.H(0)
            qml.S(1)
            qml.T(0)
            qml.CNOT([0, 1])
            m1 = catalyst.measure(wires=0)
            m2 = catalyst.measure(wires=1)
            return m1, m2
        circuit()
    ```

    The PPRs and PPMs are currently only represented symbolically. However, these operations are not yet executable on any backend since they exist purely as intermediate representations for analysis and potential future execution when a suitable backend is available.
    
    Example MLIR Representation:
    ```mlir
      . . .
        %0 = quantum.alloc( 2) : !quantum.reg
        %1 = quantum.extract %0[ 1] : !quantum.reg -> !quantum.bit
        %2 = qec.ppr ["Z"](4) %1 : !quantum.bit
        %3 = quantum.extract %0[ 0] : !quantum.reg -> !quantum.bit
        %4 = qec.ppr ["Z"](4) %3 : !quantum.bit
        %5 = qec.ppr ["X"](4) %4 : !quantum.bit
        %6 = qec.ppr ["Z"](4) %5 : !quantum.bit
        %7 = qec.ppr ["Z"](8) %6 : !quantum.bit
        %8:2 = qec.ppr ["Z", "X"](4) %7, %2 : !quantum.bit, !quantum.bit
        %9 = qec.ppr ["Z"](-4) %8#0 : !quantum.bit
        %10 = qec.ppr ["X"](-4) %8#1 : !quantum.bit
        %mres, %out_qubits = qec.ppm ["Z"] %9 : !quantum.bit
        %mres_0, %out_qubits_1 = qec.ppm ["Z"] %10 : !quantum.bit
      . . . 
    ```

<h3>Improvements 🛠</h3>

* Changed pattern rewritting in `quantum-to-ion` lowering pass to use MLIR's dialect conversion
  infrastracture.
  [(#1442)](https://github.com/PennyLaneAI/catalyst/pull/1442)

* Extend `merge-rotations` peephole optimization pass to also merge compatible rotation gates (either both controlled, or both uncontrolled) where rotation angles are any combination of static constants or dynamic values.
  [(#1489)](https://github.com/PennyLaneAI/catalyst/pull/1489)

* Catalyst now supports experimental capture of `cond`, `for_loop` and `while_loop` control flow.
  [(#1468)](https://github.com/PennyLaneAI/catalyst/pull/1468)
  [(#1509)](https://github.com/PennyLaneAI/catalyst/pull/1509)
  [(#1521)](https://github.com/PennyLaneAI/catalyst/pull/1521)
  
  To trigger the PennyLane pipeline for capturing the program as a Jaxpr, simply set
  `experimental_capture=True` in the qjit decorator.

  ```python
  import pennylane as qml
  from catalyst import qjit

  dev = qml.device("lightning.qubit", wires=1)

  @qjit(experimental_capture=True)
  @qml.qnode(dev)
  def circuit(x: float):

      def ansatz_true():
          qml.RX(x, wires=0)
          qml.Hadamard(wires=0)

      def ansatz_false():
          qml.RY(x, wires=0)

      qml.cond(x > 1.4, ansatz_true, ansatz_false)()

      return qml.expval(qml.Z(0))
  ```

* Catalyst now supports experimental capture of `qml.transforms.cancel_inverses` and `qml.transforms.merge_rotations` transforms.
  [(#1544)](https://github.com/PennyLaneAI/catalyst/pull/1544)
  [(#1561)](https://github.com/PennyLaneAI/catalyst/pull/1561)

  To trigger the PennyLane pipeline for capturing the mentioned transforms,
  simply set `experimental_capture=True` in the qjit decorator. Catalyst will
  then apply its own pass in replacement of the original transform
  provided by PennyLane.

  ```python
  import pennylane as qml
  from catalyst import qjit

  dev = qml.device("lightning.qubit", wires=1)

  @qjit(experimental_capture=True)
  def func(x: float):
      @qml.transforms.cancel_inverses
      @qml.qnode(dev)
      def circuit(x: float):
          qml.RX(x, wires=0)
          qml.Hadamard(wires=0)
          qml.Hadamard(wires=0)
          return qml.expval(qml.PauliZ(0))

      return circuit(x)
  ```

* Changes to reduce compile time:

  - Turn off MLIR's verifier.
    [(#1513)](https://github.com/PennyLaneAI/catalyst/pull/1513)
  - Remove unnecessary I/O.
    [(#1514)](https://github.com/PennyLaneAI/catalyst/pull/1514)
  - Sort improvements to reduce complexity and memory.
    [(#1524)](https://github.com/PennyLaneAI/catalyst/pull/1524)
  - Lazy IR canonicalization and LLVMIR textual generation.
    [(#1530)](https://github.com/PennyLaneAI/catalyst/pull/1530)

* Changes to support a dynamic number of qubits:

  - The `qalloc_p` custom JAX primitive can now take in a dynamic number of qubits as a tracer
    and lower it to mlir.
    [(#1549)](https://github.com/PennyLaneAI/catalyst/pull/1549)

  - `ComputationalBasisOp` can now take in a quantum register in mlir, instead of an explicit, fixed-size list of qubits.
    [(#1553)](https://github.com/PennyLaneAI/catalyst/pull/1553)

<h3>Breaking changes 💔</h3>

<h3>Deprecations 👋</h3>

<h3>Bug fixes 🐛</h3>

* Fixed `argnums` parameter of `grad` and `value_and_grad` being ignored.
  [(#1478)](https://github.com/PennyLaneAI/catalyst/pull/1478)

* Fixed an issue ([(#1488)](https://github.com/PennyLaneAI/catalyst/pull/1488)) where Catalyst could
  give incorrect results for circuits containing `qml.StatePrep`.
  [(#1491)](https://github.com/PennyLaneAI/catalyst/pull/1491)

* Fixes an issue ([(#1501)](https://github.com/PennyLaneAI/catalyst/issues/1501)) where using 
  autograph in conjunction with catalyst passes causes a crash.
  [(#1541)](https://github.com/PennyLaneAI/catalyst/pull/1541)

<h3>Internal changes ⚙️</h3>

* Updated the call signature for the PLXPR `qnode_prim` primitive.
  [(#1538)](https://github.com/PennyLaneAI/catalyst/pull/1538)

* Update deprecated access to `QNode.execute_kwargs["mcm_config"]`.
  Instead `postselect_mode` and `mcm_method` should be accessed instead.
  [(#1452)](https://github.com/PennyLaneAI/catalyst/pull/1452)

* `from_plxpr` now uses the `qml.capture.PlxprInterpreter` class for reduced code duplication.
  [(#1398)](https://github.com/PennyLaneAI/catalyst/pull/1398)

* Improve the error message for invalid measurement in `adjoin()` or `ctrl()` region.
  [(#1425)](https://github.com/PennyLaneAI/catalyst/pull/1425)

* Replace `ValueRange` with `ResultRange` and `Value` with `OpResult` to better align with the semantics of `**QubitResult()` functions like `getNonCtrlQubitResults()`. This change ensures clearer intent and usage. Improve the `matchAndRewrite` function by using `replaceAllUsesWith` instead of for loop.
  [(#1426)](https://github.com/PennyLaneAI/catalyst/pull/1426)

* Several changes for experimental support of trapped-ion OQD devices have been made, including:

  - The `get_c_interface` method has been added to the OQD device, which enables retrieval of the C++
    implementation of the device from Python. This allows `qjit` to accept an instance of the device
    and connect to its runtime.
    [(#1420)](https://github.com/PennyLaneAI/catalyst/pull/1420)

  - Improved ion dialect to reduce redundant code generated. Added a string attribute `label` to Level.
    Also changed the levels of a transition from `LevelAttr` to `string`
    [(#1471)](https://github.com/PennyLaneAI/catalyst/pull/1471)

  - The region of a `ParallelProtocolOp` is now always terminated with a `ion::YieldOp` with explicitly yielded SSA values. This ensures the op is well-formed, and improves readability.
    [(#1475)](https://github.com/PennyLaneAI/catalyst/pull/1475)

  - Add a new pass `convert-ion-to-llvm` which lowers the Ion dialect to llvm dialect. This pass 
    introduces oqd device specific stubs that will be implemented in oqd runtime including: 
    `@ __catalyst__oqd__pulse`, `@ __catalyst__oqd__ParallelProtocol`.
    [(#1466)](https://github.com/PennyLaneAI/catalyst/pull/1466)

  - The OQD device can now generate OpenAPL JSON specs during runtime. The oqd stubs
  `@ __catalyst__oqd__pulse`, and `@ __catalyst__oqd__ParallelProtocol`, which
  are called in the llvm dialect after the aforementioned lowering ([(#1466)](https://github.com/PennyLaneAI/catalyst/pull/1466)), are defined to produce JSON specs that OpenAPL expects.
    [(#1516)](https://github.com/PennyLaneAI/catalyst/pull/1516)

  - The OQD device is moved from `frontend/catalyst/third_party/oqd` to `runtime/lib/backend/oqd`. An overall switch, `ENABLE_OQD`, is added to control the OQD build system from a single entry point. The switch is `OFF` by default, and OQD can be built from source via `make all ENABLE_OQD=ON`, or `make runtime ENABLE_OQD=ON`.
    [(#1508)](https://github.com/PennyLaneAI/catalyst/pull/1508)

  - Ion dialect now supports phonon modes using `ion.modes` operation.
    [(#1517)](https://github.com/PennyLaneAI/catalyst/pull/1517)

  - Rotation angles are normalized to avoid negative duration for pulses during ion dialect lowering.
    [(#1517)](https://github.com/PennyLaneAI/catalyst/pull/1517)

  - Catalyst now generates OpenAPL programs for Pennylane circuits of up to two qubits using the OQD device.
    [(#1517)](https://github.com/PennyLaneAI/catalyst/pull/1517)

  - The end-to-end compilation pipeline for OQD devices is available as an API function.
    [(#1545)](https://github.com/PennyLaneAI/catalyst/pull/1545)

* Update source code to comply with changes requested by black v25.1.0
  [(#1490)](https://github.com/PennyLaneAI/catalyst/pull/1490)

* Revert `StaticCustomOp` in favour of adding helper functions (`isStatic()`, `getStaticParams()`
  to the `CustomOp` which preserves the same functionality. More specifically, this reverts
  [#1387] and [#1396], modifies [#1484].
  [(#1558)](https://github.com/PennyLaneAI/catalyst/pull/1558)
  [(#1555)](https://github.com/PennyLaneAI/catalyst/pull/1555)

* Updated the c++ standard in mlir layer from 17 to 20.
  [(#1229)](https://github.com/PennyLaneAI/catalyst/pull/1229)

<h3>Documentation 📝</h3>

<h3>Contributors ✍️</h3>

This release contains contributions from (in alphabetical order):

Joey Carter,
Yushao Chen,
Zach Goldthorpe,
Sengthai Heng,
David Ittah,
Rohan Nolan Lasrado,
Christina Lee,
Mehrdad Malekmohammadi,
Erick Ochoa Lopez,
Andrija Paurevic,
Raul Torres,
Paul Haochen Wang.
