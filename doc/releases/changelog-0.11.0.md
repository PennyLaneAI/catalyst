# Release 0.11.0

<h3>New features since last release</h3>

* A novel optimization technique is implemented in Catalyst that performs quantum peephole
  optimizations across loop boundaries. The technique has been added to the existing optimizations
  `cancel_inverses` and `merge_rotations` to increase their effectiveness in structured programs.
  [(#1476)](https://github.com/PennyLaneAI/catalyst/pull/1476)

  A frequently occurring pattern is operations at the beginning and end of a loop that cancel each
  other out. With loop boundary analysis, the `cancel_inverses` optimization can eliminate
  these redundant operations and thus reduce quantum circuit depth. 

  For example,

  ```python
  dev = qml.device("lightning.qubit", wires=2)

  @qml.qjit
  @catalyst.passes.cancel_inverses
  @qml.qnode(dev)
  def circuit():
      for i in range(3):
          qml.Hadamard(0)
          qml.CNOT([0, 1])
          qml.Hadamard(0)
      return qml.expval(qml.Z(0))
  ```

  Here, the Hadamard gate pairs which are consecutive across two iterations are eliminated,
  leaving behind only two unpaired Hadamard gates, from the first and last iteration, without unrolling the for loop.
  For more details on loop-boundary optimization, see the
  [PennyLane Compilation entry](https://pennylane.ai/compilation/loop-boundary-optimization).

* A new intermediate representation and compilation framework has been added to Catalyst to describe 
  and manipulate programs in the Pauli product measurement (PPM) representation. As part of this framework, 
  three new passes are now available to convert Clifford + T gates to Pauli product measurements as 
  described in [arXiv:1808.02892](https://arxiv.org/abs/1808.02892v3).
  [(#1499)](https://github.com/PennyLaneAI/catalyst/pull/1499)
  [(#1551)](https://github.com/PennyLaneAI/catalyst/pull/1551)
  [(#1563)](https://github.com/PennyLaneAI/catalyst/pull/1563)
  [(#1564)](https://github.com/PennyLaneAI/catalyst/pull/1564)
  [(#1577)](https://github.com/PennyLaneAI/catalyst/pull/1577)
  
  Note that programs in the PPM representation cannot yet be executed on available backends.
  The passes currently exist for analysis, but PPM programs may become executable in the future
  when a suitable backend is available.

  The following new compilation passes can be accessed from the :mod:`~.passes` module or in :func:`~.pipeline`:

  * :func:`catalyst.passes.to_ppr <~.passes.to_ppr>`: Clifford + T gates are converted into Pauli product 
    rotations (PPRs) (:math:`\exp{iP \theta}`, where :math:`P` is a tensor product of Pauli operators):
    * `H` gate → 3 rotations with :math:`P_1 = Z, P_2 = X, P_3 = Z` and :math:`\theta = \tfrac{\pi}{4}` 
    * `S` gate → 1 rotation with :math:`P = Z` and :math:`\theta = \tfrac{\pi}{4}` 
    * `T` gate → 1 rotation with :math:`P = Z` and :math:`\theta = \tfrac{\pi}{8}` 
    * `CNOT` gate → 3 rotations with :math:`P_1 = (Z \otimes X), P_2 = (-Z \otimes \mathbb{1}), P_3 = (-\mathbb{1} \otimes X)` and :math:`\theta = \tfrac{\pi}{4}` 

  * :func:`catalyst.passes.commute_ppr <~.passes.commute_ppr>`: Commute Clifford PPR operations 
    (PPRs with :math:`\theta = \tfrac{\pi}{4}`) to the end of the circuit, past non-Clifford PPRs (PPRs 
    with :math:`\theta = \tfrac{\pi}{8}`)

  * :func:`catalyst.passes.ppr_to_ppm <~.passes.ppr_to_ppm>`: Absorb Clifford PPRs into terminal Pauli 
    product measurements (PPMs).

  For more information on PPMs, please refer to our [PPM documentation page](https://pennylane.ai/compilation/pauli-product-measurement).

* Catalyst now supports qubit number-invariant compilation. That is, programs can be compiled without
  specifying the number of qubits to allocate ahead of time. Instead, the device can be supplied with
  a dynamic program variable as the number of wires.
  [(#1549)](https://github.com/PennyLaneAI/catalyst/pull/1549)
  [(#1553)](https://github.com/PennyLaneAI/catalyst/pull/1553)
  [(#1565)](https://github.com/PennyLaneAI/catalyst/pull/1565)
  [(#1574)](https://github.com/PennyLaneAI/catalyst/pull/1574)

  For example, the following toy workflow is now supported, where the number of qubits, `n`, is provided
  as an argument to a qjit'd function:

  ```python
  import catalyst
  import pennylane as qml

  @catalyst.qjit(autograph=True)
  def f(n):  
      device = qml.device("lightning.qubit", wires=n, shots=10)

      @qml.qnode(device)
      def circuit():

          for i in range(n):
              qml.RX(1.5, wires=i)

          return qml.counts()

      return circuit()
  ```

  ```pycon
  >>> f(3)
  (Array([0, 1, 2, 3, 4, 5, 6, 7], dtype=int64),
  Array([0, 0, 3, 2, 3, 1, 1, 0], dtype=int64))
  >>> f(4)
  (Array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15],      dtype=int64),
  Array([0, 0, 1, 1, 2, 0, 0, 0, 0, 0, 1, 1, 2, 1, 0, 1], dtype=int64))
  ```

* Catalyst better integrates with PennyLane program capture, supporting PennyLane-native control flow 
  operations and providing more efficient transform handling when both Catalyst and PennyLane support 
  a transform.
  [(#1468)](https://github.com/PennyLaneAI/catalyst/pull/1468)
  [(#1509)](https://github.com/PennyLaneAI/catalyst/pull/1509)
  [(#1521)](https://github.com/PennyLaneAI/catalyst/pull/1521)
  [(#1544)](https://github.com/PennyLaneAI/catalyst/pull/1544)
  [(#1561)](https://github.com/PennyLaneAI/catalyst/pull/1561)
  [(#1567)](https://github.com/PennyLaneAI/catalyst/pull/1567)
  [(#1578)](https://github.com/PennyLaneAI/catalyst/pull/1578)

  Using PennyLane's program capture mechanism involves setting `experimental_capture=True` in the qjit 
  decorator. With this present, the following control flow functions in PennyLane are now usable with
  qjit:

  * Support for `qml.cond`:

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

    ```pycon
    >>> circuit(0.1)
    Array(0.99500417, dtype=float64)
    ```

  * Support for `qml.for_loop`:

    ```python
    dev = qml.device("lightning.qubit", wires=2)

    @qjit(experimental_capture=True)
    @qml.qnode(dev)
    def circuit(x: float):

        @qml.for_loop(10)
        def loop(i):
            qml.H(wires=1)
            qml.RX(x, wires=0)
            qml.CNOT(wires=[0, 1])

        loop()
        return qml.expval(qml.Z(0))
    ```

    ```pycon
    >>> circuit(0.1)
    Array(0.97986841, dtype=float64)
    ```

  * Support for `qml.while_loop`:

    ```python
    @qjit(experimental_capture=True)
    @qml.qnode(dev)
    def circuit(x: float):

        f = lambda c: c < 5

        @qml.while_loop(f)
        def loop(c):
            qml.H(wires=1)
            qml.RX(x, wires=0)
            qml.CNOT(wires=[0, 1])

            return c + 1
        
        loop(0)
        return qml.expval(qml.Z(0))
    ```

    ```pycon
    >>> circuit(0.1)
    Array(0.97526892, dtype=float64)
    ```

  Additionally, Catalyst can now apply its own compilation passes when equivalent transforms are provided 
  by PennyLane (e.g., `cancel_inverses` and `merge_rotations`). In cases where Catalyst does not have 
  its own analogous implementation of a transform available in PennyLane, the transform will be expanded 
  according to rules provided by PennyLane.

  For example, consider this workflow that contains two PennyLane transforms: `cancel_inverses` and 
  `single_qubit_fusion`. Catalyst has its own implementation of `cancel_inverses` in the `passes` module, 
  and will smartly invoke its implementation intead. Conversely, Catalyst does not have its own implementation
  of `single_qubit_fusion`, and will therefore resort to PennyLane's implementation of the transform.

  ```python
  dev = qml.device("lightning.qubit", wires=1)

  @qjit(experimental_capture=True)
  def func(r1, r2):

      @qml.transforms.cancel_inverses
      @qml.transforms.single_qubit_fusion
      @qml.qnode(dev)
      def circuit(r1, r2):
          qml.Rot(*r1, wires=0)
          qml.Rot(*r2, wires=0)
          qml.RZ(r1[0], wires=0)
          qml.RZ(r2[0], wires=0) 

          qml.Hadamard(wires=0)
          qml.Hadamard(wires=0)
          
          return qml.expval(qml.PauliZ(0))  

      return circuit(r1, r2)
  ```

  ```pycon
  >>> r1 = jnp.array([0.1, 0.2, 0.3])
  >>> r2 = jnp.array([0.4, 0.5, 0.6])
  >>> func(r1, r2)
  Array(0.7872403, dtype=float64)
  ```

<h3>Improvements 🛠</h3>


* Several changes have been made to reduce compile time:
  * MLIR's verifier has been turned off.
    [(#1513)](https://github.com/PennyLaneAI/catalyst/pull/1513)
  * Unnecessary I/O has been removed.
    [(#1514)](https://github.com/PennyLaneAI/catalyst/pull/1514)
    [(#1602)](https://github.com/PennyLaneAI/catalyst/pull/1602)
  * Improvements have been made to reduce complexity and memory.
    [(#1524)](https://github.com/PennyLaneAI/catalyst/pull/1524)
  * IR canonicalization and LLVMIR textual generation is now performed lazily.
    [(#1530)](https://github.com/PennyLaneAI/catalyst/pull/1530)
  * Speed up how tracers are overwritten for hybrid ops.
    [(#1622)](https://github.com/PennyLaneAI/catalyst/pull/1622)

* Catalyst now decomposes non-differentiable gates when differentiating through workflows. Additionally, with `diff_method=parameter-shift`,
  circuits are now verified to be fully compatible with Catalyst's parameter-shift implementation before compilation.
  [(#1562)](https://github.com/PennyLaneAI/catalyst/pull/1562)
  [(#1568)](https://github.com/PennyLaneAI/catalyst/pull/1568)
  [(#1569)](https://github.com/PennyLaneAI/catalyst/pull/1569)
  [(#1604)](https://github.com/PennyLaneAI/catalyst/pull/1604)

  Gates that are constant, such as when all parameters are Python or NumPy data types, are not
  decomposed when this is allowable. For the adjoint differentiation method, this is allowable
  for the `StatePrep`, `BasisState`, and `QubitUnitary` operations. For the parameter-shift method,
  this is allowable for all operations.

* An `mlir_opt` property has been added to `qjit` to access the optimized MLIR representation of a compiled 
  function. This is the representation of the program after running everything in the MLIR stage of the entire pipeline.
  [(#1579)](https://github.com/PennyLaneAI/catalyst/pull/1579)
  [(#1637)](https://github.com/PennyLaneAI/catalyst/pull/1637)

  ```python
  from catalyst import qjit

  @qjit
  def f(x):
      return x**2
  ```

  ```pycon
  >>> f(2)
  Array(4, dtype=int64)
  >>> print(f.mlir_opt)
  module @f {
    llvm.func @__catalyst__rt__finalize()
    llvm.func @__catalyst__rt__initialize(!llvm.ptr)
    llvm.func @_mlir_memref_to_llvm_alloc(i64) -> !llvm.ptr
    llvm.func @jit_f(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: i64) -> !llvm.struct<(ptr, ptr, i64)> attributes {llvm.copy_memref, llvm.emit_c_interface} 
    ...
    llvm.func @teardown() {
      llvm.call @__catalyst__rt__finalize() : () -> ()
      llvm.return
    }
  }
  ```

* The error messages that indicate invalid `scale_factors` in `catalyst.mitigate_with_zne` have been
  improved to be formatted properly.
  [(#1603)](https://github.com/PennyLaneAI/catalyst/pull/1603)

<h3>Bug fixes 🐛</h3>

* Fixed the `argnums` parameter of `grad` and `value_and_grad` being ignored.
  [(#1478)](https://github.com/PennyLaneAI/catalyst/pull/1478)

* All dialects are loaded preemptively. This allows third-party plugins to load their dialects.
  [(#1584)](https://github.com/PennyLaneAI/catalyst/pull/1584)

* Fixed an issue where Catalyst could give incorrect results for circuits containing `qml.StatePrep`.
  [(#1491)](https://github.com/PennyLaneAI/catalyst/pull/1491)

* Fixed an issue where using autograph in conjunction with catalyst passes caused a crash.
  [(#1541)](https://github.com/PennyLaneAI/catalyst/pull/1541)

* Fixed an issue where using
  autograph in conjunction with catalyst pipeline caused a crash.
  [(#1576)](https://github.com/PennyLaneAI/catalyst/pull/1576)

* Fixed an issue where using
  chained catalyst passes decorators caused a crash.
  [(#1576)](https://github.com/PennyLaneAI/catalyst/pull/1576)

* Specialized handling for `pipeline`s was added.
  [(#1599)](https://github.com/PennyLaneAI/catalyst/pull/1599)

* Fixed an issue where using
  autograph with control/adjoint functions used on operator objects caused a crash.
  [(#1605)](https://github.com/PennyLaneAI/catalyst/pull/1605)

* Fixed an issue where using
  pytrees inside a loop with autograph caused falling back to Python.
  [(#1601)](https://github.com/PennyLaneAI/catalyst/pull/1601)

For example, the following example will now be captured and executed properly with Autograph enabled:

```python
from catalyst import qjit

def updateList(x):
    return [x[0]+1, x[1]+2]

@qjit(autograph=True)
def fn(x):
    for i in range(4):
        x = updateList(x)
    return x
```

```pycon
>>> fn([1, 2])
[Array(5, dtype=int64), Array(10, dtype=int64)]
```

* Closure variables are now supported with `grad` and `value_and_grad`.
  [(#1613)](https://github.com/PennyLaneAI/catalyst/pull/1613)

<h3>Internal changes ⚙️</h3>

* Pattern rewriting in the `quantum-to-ion` lowering pass has been changed to use MLIR's dialect conversion
  infrastructure.
  [(#1442)](https://github.com/PennyLaneAI/catalyst/pull/1442)

* Updated the call signature for the plxpr `qnode_prim` primitive.
  [(#1538)](https://github.com/PennyLaneAI/catalyst/pull/1538)

* Update deprecated access to `QNode.execute_kwargs["mcm_config"]`.
  Instead `postselect_mode` and `mcm_method` should be accessed instead.
  [(#1452)](https://github.com/PennyLaneAI/catalyst/pull/1452)

* `from_plxpr` now uses the `qml.capture.PlxprInterpreter` class for reduced code duplication.
  [(#1398)](https://github.com/PennyLaneAI/catalyst/pull/1398)

* Improved the error message for invalid measurement in `adjoin()` or `ctrl()` region.
  [(#1425)](https://github.com/PennyLaneAI/catalyst/pull/1425)

* Replaced `ValueRange` with `ResultRange` and `Value` with `OpResult` to better align with the semantics 
  of `**QubitResult()` functions like `getNonCtrlQubitResults()`. This change ensures clearer intent 
  and usage. Also, the `matchAndRewrite` function has improved by using `replaceAllUsesWith` instead 
  of a `for` loop.
  [(#1426)](https://github.com/PennyLaneAI/catalyst/pull/1426)

* Several changes for experimental support of trapped-ion OQD devices have been made, including:

  * The `get_c_interface` method has been added to the OQD device, which enables retrieval of the C++
    implementation of the device from Python. This allows `qjit` to accept an instance of the device
    and connect to its runtime.
    [(#1420)](https://github.com/PennyLaneAI/catalyst/pull/1420)

  * The ion dialect has been improved to reduce redundant code generated, a string attribute `label` 
    has been added to Level, and the levels of a transition have changed from `LevelAttr` to `string`.
    [(#1471)](https://github.com/PennyLaneAI/catalyst/pull/1471)

  * The region of a `ParallelProtocolOp` is now always terminated with a `ion::YieldOp` with explicitly 
    yielded SSA values. This ensures the op is well-formed, and improves readability.
    [(#1475)](https://github.com/PennyLaneAI/catalyst/pull/1475)

  * Added a new pass called `convert-ion-to-llvm` which lowers the Ion dialect to llvm dialect. This 
    pass introduces oqd device specific stubs that will be implemented in oqd runtime including:
    `@ __catalyst__oqd__pulse`, `@ __catalyst__oqd__ParallelProtocol`.
    [(#1466)](https://github.com/PennyLaneAI/catalyst/pull/1466)

  * The OQD device can now generate OpenAPL JSON specs during runtime. The oqd stubs `@ __catalyst__oqd__pulse`, 
    and `@ __catalyst__oqd__ParallelProtocol`, which are called in the llvm dialect after the aforementioned 
    lowering ([(#1466)](https://github.com/PennyLaneAI/catalyst/pull/1466)), are defined to produce 
    JSON specs that OpenAPL expects.
    [(#1516)](https://github.com/PennyLaneAI/catalyst/pull/1516)

  * The OQD device has been moved from `frontend/catalyst/third_party/oqd` to `runtime/lib/backend/oqd`. 
    An overall switch, `ENABLE_OQD`, is added to control the OQD build system from a single entry point. 
    The switch is `OFF` by default, and OQD can be built from source via `make all ENABLE_OQD=ON`, or 
    `make runtime ENABLE_OQD=ON`.
    [(#1508)](https://github.com/PennyLaneAI/catalyst/pull/1508)

  * Ion dialect now supports phonon modes using `ion.modes` operation.
    [(#1517)](https://github.com/PennyLaneAI/catalyst/pull/1517)

  * Rotation angles are normalized to avoid negative duration for pulses during ion dialect lowering.
    [(#1517)](https://github.com/PennyLaneAI/catalyst/pull/1517)

  * Catalyst now generates OpenAPL programs for Pennylane circuits of up to two qubits using the OQD device.
    [(#1517)](https://github.com/PennyLaneAI/catalyst/pull/1517)

  * The end-to-end compilation pipeline for OQD devices is available as an API function.
    [(#1545)](https://github.com/PennyLaneAI/catalyst/pull/1545)

* The source code has been updated to comply with changes requested by black v25.1.0
  [(#1490)](https://github.com/PennyLaneAI/catalyst/pull/1490)

* Reverted `StaticCustomOp` in favour of adding helper functions `isStatic()`, `getStaticParams()`
  to the `CustomOp` which preserves the same functionality. More specifically, this reverts
  [#1387] and [#1396], modifies [#1489].
  [(#1558)](https://github.com/PennyLaneAI/catalyst/pull/1558)
  [(#1555)](https://github.com/PennyLaneAI/catalyst/pull/1555)

* Updated the C++ standard in mlir layer from 17 to 20.
  [(#1229)](https://github.com/PennyLaneAI/catalyst/pull/1229)

<h3>Documentation 📝</h3>

* Added more details to JAX integration documentation regarding the use of `.at` with multiple indices.
  [(#1595)](https://github.com/PennyLaneAI/catalyst/pull/1595)

<h3>Contributors ✍️</h3>

This release contains contributions from (in alphabetical order):

Joey Carter,
Yushao Chen,
Isaac De Vlugt,
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
