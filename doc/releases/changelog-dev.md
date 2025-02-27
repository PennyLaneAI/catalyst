# Release 0.11.0 (development release)

<h3>New features since last release</h3>

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

* Changes to reduce compile time:

  - Turn off MLIR's verifier.
    [(#1513)](https://github.com/PennyLaneAI/catalyst/pull/1513)
  - Remove unnecessary I/O.
    [(#1514)](https://github.com/PennyLaneAI/catalyst/pull/1514)
  - Sort improvements to reduce complexity and memory.
    [(#1524)](https://github.com/PennyLaneAI/catalyst/pull/1524)
  - Lazy IR canonicalization and LLVMIR textual generation.
    [(#1530)](https://github.com/PennyLaneAI/catalyst/pull/1530)

<h3>Breaking changes 💔</h3>

<h3>Deprecations 👋</h3>

<h3>Bug fixes 🐛</h3>

* Fixed `argnums` parameter of `grad` and `value_and_grad` being ignored.
  [(#1478)](https://github.com/PennyLaneAI/catalyst/pull/1478)

* Fixed an issue ([(#1488)](https://github.com/PennyLaneAI/catalyst/pull/1488)) where Catalyst could
  give incorrect results for circuits containing `qml.StatePrep`.
  [(#1491)](https://github.com/PennyLaneAI/catalyst/pull/1491)

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

* Update source code to comply with changes requested by black v25.1.0
  [(#1490)](https://github.com/PennyLaneAI/catalyst/pull/1490)

<h3>Documentation 📝</h3>

<h3>Contributors ✍️</h3>

This release contains contributions from (in alphabetical order):

Joey Carter,
Yushao Chen,
Zach Goldthorpe,
Sengthai Heng,
Rohan Nolan Lasrado,
Christina Lee,
Mehrdad Malekmohammadi,
Erick Ochoa Lopez,
Andrija Paurevic,
Raul Torres,
Paul Haochen Wang.
