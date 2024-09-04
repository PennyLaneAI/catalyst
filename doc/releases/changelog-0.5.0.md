# Release 0.5.0

<h3>New features</h3>

* Catalyst now provides a QJIT compatible `catalyst.vmap`
  function, which makes it even easier to modify functions to map over inputs
  with additional batch dimensions.
  [(#497)](https://github.com/PennyLaneAI/catalyst/pull/497)
  [(#569)](https://github.com/PennyLaneAI/catalyst/pull/569)

  When working with tensor/array frameworks in Python, it can be important to ensure that code is
  written to minimize usage of Python for loops (which can be slow and inefficient), and instead
  push as much of the computation through to the array manipulation library, by taking advantage of
  extra batch dimensions.

  For example, consider the following QNode:

  ```python
  dev = qml.device("lightning.qubit", wires=1)

  @qml.qnode(dev)
  def circuit(x, y):
      qml.RX(jnp.pi * x[0] + y, wires=0)
      qml.RY(x[1] ** 2, wires=0)
      qml.RX(x[1] * x[2], wires=0)
      return qml.expval(qml.PauliZ(0))
  ```

  ```pycon
  >>> circuit(jnp.array([0.1, 0.2, 0.3]), jnp.pi)
  Array(-0.93005586, dtype=float64)
  ```

  We can use `catalyst.vmap` to introduce additional batch dimensions to our input arguments,
  without needing to use a Python for loop:

  ```pycon
  >>> x = jnp.array([[0.1, 0.2, 0.3],
  ...                [0.4, 0.5, 0.6],
  ...                [0.7, 0.8, 0.9]])
  >>> y = jnp.array([jnp.pi, jnp.pi / 2, jnp.pi / 4])
  >>> qjit(vmap(cost))(x, y)
  array([-0.93005586, -0.97165424, -0.6987465 ])
  ```

  `catalyst.vmap()` has been implemented to match the same behaviour of `jax.vmap`, so should be a drop-in
  replacement in most cases. Under-the-hood, it is automatically inserting Catalyst-compatible for loops,
  which will be compiled and executed outside of Python for increased performance.

* Catalyst now supports compiling and executing QJIT-compiled QNodes using the
  CUDA Quantum compiler toolchain.
  [(#477)](https://github.com/PennyLaneAI/catalyst/pull/477)
  [(#536)](https://github.com/PennyLaneAI/catalyst/pull/536)
  [(#547)](https://github.com/PennyLaneAI/catalyst/pull/547)

  Simply import the CUDA Quantum `@cudaqjit` decorator to use this functionality:

  ```python
  from catalyst.cuda import cudaqjit
  ```

  Or, if using Catalyst from PennyLane, simply specify `@qml.qjit(compiler="cuda_quantum")`.

  The following devices are available when compiling with CUDA Quantum:

  * `softwareq.qpp`: a modern C++ state-vector simulator
  * `nvidia.custatevec`: The NVIDIA CuStateVec GPU simulator (with support for multi-gpu)
  * `nvidia.cutensornet`: The NVIDIA CuTensorNet GPU simulator (with support for matrix product state)

  For example:

  ```python
  dev = qml.device("softwareq.qpp", wires=2)

  @cudaqjit
  @qml.qnode(dev)
  def circuit(x):
      qml.RX(x[0], wires=0)
      qml.RY(x[1], wires=1)
      qml.CNOT(wires=[0, 1])
      return qml.expval(qml.PauliY(0))
  ```

  ```pycon
  >>> circuit(jnp.array([0.5, 1.4]))
  -0.47244976756708373
  ```

  Note that CUDA Quantum compilation currently does not have feature parity with Catalyst
  compilation; in particular, AutoGraph, control flow, differentiation, and various measurement
  statistics (such as probabilities and variance) are not yet supported.
  Classical code support is also limited.

* Catalyst now supports just-in-time compilation of static (compile-time constant) arguments.
  [(#476)](https://github.com/PennyLaneAI/catalyst/pull/476)
  [(#550)](https://github.com/PennyLaneAI/catalyst/pull/550)

  The `@qjit` decorator takes a new argument `static_argnums`, which specifies positional
  arguments of the decorated function should be treated as compile-time static arguments.

  This allows any hashable Python object to be passed to the function during compilation;
  the function will only be re-compiled if the hash value of the static arguments change.
  Otherwise, re-using previous static argument values will result in no re-compilation.

  ```python
  @qjit(static_argnums=(1,))
  def f(x, y):
      print(f"Compiling with y={y}")
      return x + y
  ```

  ```pycon
  >>> f(0.5, 0.3)
  Compiling with y=0.3
  array(0.8)
  >>> f(0.1, 0.3)  # no re-compilation occurs
  array(0.4)
  >>> f(0.1, 0.4)  # y changes, re-compilation
  Compiling with y=0.4
  array(0.5)
  ```

  This functionality can be used to support passing arbitrary Python objects to QJIT-compiled
  functions, as long as they are hashable:

  ```py
  from dataclasses import dataclass

  @dataclass
  class MyClass:
      val: int

      def __hash__(self):
          return hash(str(self))

  @qjit(static_argnums=(1,))
  def f(x: int, y: MyClass):
      return x + y.val
  ```

  ```pycon
  >>> f(1, MyClass(5))
  array(6)
  >>> f(1, MyClass(6))  # re-compilation
  array(7)
  >>> f(2, MyClass(5))  # no re-compilation
  array(7)
  ```

* Mid-circuit measurements now support post-selection and qubit reset when used with
  the Lightning simulators.
  [(#491)](https://github.com/PennyLaneAI/catalyst/pull/491)
  [(#507)](https://github.com/PennyLaneAI/catalyst/pull/507)

  To specify post-selection, simply pass the `postselect` argument to the `catalyst.measure`
  function:

  ```python
  dev = qml.device("lightning.qubit", wires=1)

  @qjit
  @qml.qnode(dev)
  def f():
      qml.Hadamard(0)
      m = measure(0, postselect=1)
      return qml.expval(qml.PauliZ(0))
  ```

  Likewise, to reset a wire after mid-circuit measurement, simply specify `reset=True`:

  ```python
  dev = qml.device("lightning.qubit", wires=1)

  @qjit
  @qml.qnode(dev)
  def f():
      qml.Hadamard(0)
      m = measure(0, reset=True)
      return qml.expval(qml.PauliZ(0))
  ```

<h3>Improvements</h3>

* Catalyst now supports Python 3.12
  [(#532)](https://github.com/PennyLaneAI/catalyst/pull/532)

* The JAX version used by Catalyst has been updated to `v0.4.23`.
  [(#428)](https://github.com/PennyLaneAI/catalyst/pull/428)

* Catalyst now supports the `qml.GlobalPhase` operation.
  [(#563)](https://github.com/PennyLaneAI/catalyst/pull/563)

* Native support for `qml.PSWAP` and `qml.ISWAP` gates on Amazon Braket devices has been added.
  [(#458)](https://github.com/PennyLaneAI/catalyst/pull/458)

  Specifically, a circuit like

  ```py
  dev = qml.device("braket.local.qubit", wires=2, shots=100)

  @qjit
  @qml.qnode(dev)
  def f(x: float):
      qml.Hadamard(0)
      qml.PSWAP(x, wires=[0, 1])
      qml.ISWAP(wires=[1, 0])
      return qml.probs()
  ```

* Add support for `GlobalPhase` gate in the runtime.
  [(#563)](https://github.com/PennyLaneAI/catalyst/pull/563)

  would no longer decompose the `PSWAP` and `ISWAP` gates.

* The `qml.BlockEncode` operator is now supported with Catalyst.
  [(#483)](https://github.com/PennyLaneAI/catalyst/pull/483)

* Catalyst no longer relies on a TensorFlow installation for its AutoGraph functionality. Instead,
  the standalone `diastatic-malt` package is used and automatically installed as a dependency.
  [(#401)](https://github.com/PennyLaneAI/catalyst/pull/401)

* The `@qjit` decorator will remember previously compiled functions when the PyTree metadata
  of arguments changes, in addition to also remembering compiled functions when static
  arguments change.
  [(#522)](https://github.com/PennyLaneAI/catalyst/pull/531)

  The following example will no longer trigger a third compilation:
  ```py
  @qjit
  def func(x):
      print("compiling")
      return x
  ```
  ```pycon
  >>> func([1,]);             # list
  compiling
  >>> func((2,));             # tuple
  compiling
  >>> func([3,]);             # list
  ```

  Note however that in order to keep overheads low, changing the argument *type* or *shape* (in a
  promotion incompatible way) may override a previously stored function (with identical PyTree
  metadata and static argument values):

  ```py
  @qjit
  def func(x):
      print("compiling")
      return x
  ```
  ```pycon
  >>> func(jnp.array(1));     # scalar
  compiling
  >>> func(jnp.array([2.]));  # 1-D array
  compiling
  >>> func(jnp.array(3));     # scalar
  compiling
  ```

* Catalyst gradient functions (`grad`, `jacobian`, `vjp`, and `jvp`) now support
  being applied to functions that use (nested) container
  types as inputs and outputs. This includes lists and dictionaries, as well
  as any data structure implementing the [PyTree protocol](https://jax.readthedocs.io/en/latest/pytrees.html).
  [(#500)](https://github.com/PennyLaneAI/catalyst/pull/500)
  [(#501)](https://github.com/PennyLaneAI/catalyst/pull/501)
  [(#508)](https://github.com/PennyLaneAI/catalyst/pull/508)
  [(#549)](https://github.com/PennyLaneAI/catalyst/pull/549)

  ```py
  dev = qml.device("lightning.qubit", wires=1)

  @qml.qnode(dev)
  def circuit(phi, psi):
      qml.RY(phi, wires=0)
      qml.RX(psi, wires=0)
      return [{"expval0": qml.expval(qml.PauliZ(0))}, qml.expval(qml.PauliZ(0))]

  psi = 0.1
  phi = 0.2
  ```
  ```pycon
  >>> qjit(jacobian(circuit, argnum=[0, 1]))(psi, phi)
  [{'expval0': (array(-0.0978434), array(-0.19767681))}, (array(-0.0978434), array(-0.19767681))]
  ```

* Support has been added for linear algebra functions which depend on computing the eigenvalues
  of symmetric matrices, such as `np.sqrt_matrix()`.
  [(#488)](https://github.com/PennyLaneAI/catalyst/pull/488)

  For example, you can compile `qml.math.sqrt_matrix`:

  ```python
  @qml.qjit
  def workflow(A):
      B = qml.math.sqrt_matrix(A)
      return B @ A
  ```

  Internally, this involves support for lowering the eigenvectors/values computation lapack method
  `lapack_dsyevd` via `stablehlo.custom_call`.

* Additional debugging functions are now available in the `catalyst.debug` directory.
  [(#529)](https://github.com/PennyLaneAI/catalyst/pull/529)
  [(#522)](https://github.com/PennyLaneAI/catalyst/pull/531)

  This includes:

  - `filter_static_args(args, static_argnums)` to remove static values from arguments using the
    provided index list.

  - `get_cmain(fn, *args)` to return a C program that calls a jitted function with the provided
    arguments.

  - `print_compilation_stage(fn, stage)` to print one of the recorded compilation stages for a
    JIT-compiled function.

  For more details, please see the `catalyst.debug` documentation.

* Remove redundant copies of TOML files for `lightning.kokkos` and `lightning.qubit`.
  [(#472)](https://github.com/PennyLaneAI/catalyst/pull/472)

  `lightning.kokkos` and `lightning.qubit` now ship with their own TOML file. As such, we use the TOML file provided by them.

* Capturing quantum circuits with many gates prior to compilation is now quadratically faster (up to
  a factor), by removing `qextract_p` and `qinst_p` from forced-order primitives.
  [(#469)](https://github.com/PennyLaneAI/catalyst/pull/469)

* Update `AllocateQubit` and `AllocateQubits` in `LightningKokkosSimulator` to preserve
  the current state-vector before qubit re-allocations in the runtime dynamic qubits management.
  [(#479)](https://github.com/PennyLaneAI/catalyst/pull/479)

* The [PennyLane custom compiler entry point name convention has changed](https://github.com/PennyLaneAI/pennylane/pull/5140), necessitating
  a change to the Catalyst entry points.
  [(#493)](https://github.com/PennyLaneAI/catalyst/pull/493)

<h3>Breaking changes</h3>

* Catalyst gradient functions now match the Jax convention for the returned axes of
  gradients, Jacobians, VJPs, and JVPs. As a result, the returned tensor shape from various
  Catalyst gradient functions may differ compared to previous versions of Catalyst.
  [(#500)](https://github.com/PennyLaneAI/catalyst/pull/500)
  [(#501)](https://github.com/PennyLaneAI/catalyst/pull/501)
  [(#508)](https://github.com/PennyLaneAI/catalyst/pull/508)

* The Catalyst Python frontend has been partially refactored. The impact on user-facing
  functionality is minimal, but the location of certain classes and methods used by the package
  may have changed.
  [(#529)](https://github.com/PennyLaneAI/catalyst/pull/529)
  [(#522)](https://github.com/PennyLaneAI/catalyst/pull/531)

  The following changes have been made:

  * Some debug methods and features on the QJIT class have been turned into free functions and moved
    to the `catalyst.debug` module, which will now appear in the public documention. This includes
    compiling a program from IR, obtaining a C program to invoke a compiled function from, and
    printing fine-grained MLIR compilation stages.

  * The `compilation_pipelines.py` module has been renamed to `jit.py`, and certain functionality
    has been moved out (see following items).

  * A new module `compiled_functions.py` now manages low-level access to compiled functions.

  * A new module `tracing/type_signatures.py` handles functionality related managing arguments
    and type signatures during the tracing process.

  * The `contexts.py` module has been moved from `utils` to the new `tracing` sub-module.

<h3>Internal changes</h3>

* Changes to the runtime QIR API and dependencies, to avoid symbol conflicts
  with other libraries that utilize QIR.
  [(#464)](https://github.com/PennyLaneAI/catalyst/pull/464)
  [(#470)](https://github.com/PennyLaneAI/catalyst/pull/470)

  The existing Catalyst runtime implements QIR as a library that can be linked against a QIR module.
  This works great when Catalyst is the only implementor of QIR, however it may generate
  symbol conflicts when used alongside other QIR implementations.

  To avoid this, two changes were necessary:

  * The Catalyst runtime now has a different API from QIR instructions.

    The runtime has been modified such that QIR instructions are lowered to functions where
    the `__quantum__` part of the function name is replaced with `__catalyst__`. This prevents
    the possibility of symbol conflicts with other libraries that implement QIR as a library.

  * The Catalyst runtime no longer depends on QIR runner's stdlib.

    We no longer depend nor link against QIR runner's stdlib. By linking against QIR runner's stdlib,
    some definitions persisted that may be different than ones used by third party implementors. To
    prevent symbol conflicts QIR runner's stdlib was removed and is no longer linked against. As a
    result, the following functions are now defined and implemented in Catalyst's runtime:

    * `int64_t __catalyst__rt__array_get_size_1d(QirArray *)`
    * `int8_t *__catalyst__rt__array_get_element_ptr_1d(QirArray *, int64_t)`

    and the following functions were removed since the frontend does not generate them

    * `QirString *__catalyst__rt__qubit_to_string(QUBIT *)`
    * `QirString *__catalyst__rt__result_to_string(RESULT *)`

* Fix an issue when no qubit number was specified for the `qinst` primitive. The primitive now
  correctly deduces the number of qubits when no gate parameters are present. This change is not
  user facing.
  [(#496)](https://github.com/PennyLaneAI/catalyst/pull/496)

<h3>Bug fixes</h3>

* Fixed a bug where differentiation of sliced arrays would result in an error.
  [(#552)](https://github.com/PennyLaneAI/catalyst/pull/552)

  ```py
  def f(x):
    return jax.numpy.sum(x[::2])

  x = jax.numpy.array([0.1, 0.2, 0.3, 0.4])
  ```
  ```pycon
  >>> catalyst.qjit(catalyst.grad(f))(x)
  [1. 0. 1. 0.]
  ```

* Fixed a bug where quantum control applied to a subcircuit was not correctly mapping wires,
  and the wires in the nested region remained unchanged.
  [(#555)](https://github.com/PennyLaneAI/catalyst/pull/555)

* Catalyst will no longer print a warning that recompilation is triggered when a `@qjit` decorated
  function with no arguments is invoke without having been compiled first, for example via the use
  of `target="mlir"`.
  [(#522)](https://github.com/PennyLaneAI/catalyst/pull/531)

* Fixes a bug in the configuration of dynamic shaped arrays that would cause certain program to
  error with `TypeError: cannot unpack non-iterable ShapedArray object`.
  [(#526)](https://github.com/PennyLaneAI/catalyst/pull/526)

  This is fixed by replacing the code which updates the `JAX_DYNAMIC_SHAPES` option with a
  `transient_jax_config()` context manager which temporarily sets the value of
  `JAX_DYNAMIC_SHAPES` to True and then restores the original configuration value following the
  yield. The context manager is used by `trace_to_jaxpr()` and `lower_jaxpr_to_mlir()`.

* Exceptions encountered in the runtime when using the `@qjit` option `async_qnodes=Tue`
  will now be properly propagated to the frontend.
  [(#447)](https://github.com/PennyLaneAI/catalyst/pull/447)
  [(#510)](https://github.com/PennyLaneAI/catalyst/pull/510)

  This is done by:
  * changeing `llvm.call` to `llvm.invoke`
  * setting async runtime tokens and values to be errors
  * deallocating live tokens and values

* Fixes a bug when computing gradients with the indexing/slicing,
  by fixing the scatter operation lowering when `updatedWindowsDim` is empty.
  [(#475)](https://github.com/PennyLaneAI/catalyst/pull/475)

* Fix the issue in `LightningKokkos::AllocateQubits` with allocating too many qubit IDs on
  qubit re-allocation.
  [(#473)](https://github.com/PennyLaneAI/catalyst/pull/473)

* Fixed an issue where wires was incorrectly set as `<Wires = [<WiresEnum.AnyWires: -1>]>`
  when using `catalyst.adjoint` and `catalyst.ctrl`, by adding a `wires` property to
  these operations.
  [(#480)](https://github.com/PennyLaneAI/catalyst/pull/480)

* Fix the issue with multiple lapack symbol definitions in the compiled program by updating
  the `stablehlo.custom_call` conversion pass.
  [(#488)](https://github.com/PennyLaneAI/catalyst/pull/488)

<h3>Contributors</h3>

This release contains contributions from (in alphabetical order):

Mikhail Andrenkov,
Ali Asadi,
David Ittah,
Tzung-Han Juang,
Erick Ochoa Lopez,
Romain Moyard,
Raul Torres,
Haochen Paul Wang.
