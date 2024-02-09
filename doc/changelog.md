# Release 0.5.0-dev

<h3>New features</h3>

* Catalyst now supports just-in-time compilation of static arguments.
  [(#476)](https://github.com/PennyLaneAI/catalyst/pull/476)

  The ``@qjit`` decorator can now be used to compile functions with static arguments with
  the ``static_argnums`` keyword argument. ``static_argnums`` can be an integer or an iterable
  of integers that specify which positional arguments of a compiled function should be treated as
  static. This feature allows users to pass hashable Python objects to a compiled function (as the
  static arguments).

  A ``qjit`` object stores the hash value of a compiled function's static arguments. If any static
  arguments are changed, the ``qjit`` object will check its stored hash values. If no hash value is
  found, the function will be re-compiled with new static arguments. Otherwise, no re-compilation
  will be triggered and the previously compiled function will be used.

  ```py
  from dataclasses import dataclass

  from catalyst import qjit

  @dataclass
  class MyClass:
      val: int

      def __hash__(self):
          return hash(str(self))

  @qjit(static_argnums=(1,))
  def f(
      x: int,
      y: MyClass,
  ):
      return x + y.val

  f(1, MyClass(5))
  f(1, MyClass(6)) # re-compilation
  f(2, MyClass(5)) # no re-compilation
  ```

<h3>Improvements</h3>

* Keep the structure of the function return when taking the derivatives, JVP and VJP (pytrees support).
  [(#500)](https://github.com/PennyLaneAI/catalyst/pull/501)

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

* Add native support for `qml.PSWAP` and `qml.ISWAP` gates on Amazon Braket devices. Specifically, a circuit like

  ```py
  import pennylane as qml
  from catalyst import qjit

  dev = qml.device("braket.local.qubit", wires=2, shots=100)

  @qjit
  @qml.qnode(dev)
  def f(x: float):
      qml.Hadamard(0)
      qml.PSWAP(x, wires=[0, 1])
      qml.ISWAP(wires=[1, 0])
      return qml.probs()
  ```

  would no longer decompose the `PSWAP` and `ISWAP` gates to `SWAP`s, `CNOT`s and `Hadamard`s. Instead it would just call Braket's native `PSWAP` and `ISWAP` gates at runtime.
  [(#458)](https://github.com/PennyLaneAI/catalyst/pull/458)

* Add support for the `BlockEncode` operator in Catalyst.
  [(#483)](https://github.com/PennyLaneAI/catalyst/pull/483)

* Remove copies of TOML device configuration files for Lightning device in Catalyst.
  [(#472)](https://github.com/PennyLaneAI/catalyst/pull/472)

* Remove `qextract_p` and `qinst_p` from forced-order primitives.
  [(#469)](https://github.com/PennyLaneAI/catalyst/pull/469)

* Update `AllocateQubit` and `AllocateQubits` in `LightningKokkosSimulator` to preserve
  the current state-vector before qubit re-allocations.
  [(#479)](https://github.com/PennyLaneAI/catalyst/pull/479)

* Add support for post-selection in mid-circuit measurements. This is currently supported only on Lightning simulators.
  [(#491)](https://github.com/PennyLaneAI/catalyst/pull/491)

  This is an example of post-selection usage:

  ```py
  import pennylane as qml
  from catalyst import qjit

  dev = qml.device("lightning.qubit", wires=1)

  @qjit
  @qml.qnode(dev)
  def f():
      qml.Hadamard(0)
      m = measure(0, postselect=1)
      return qml.expval(qml.PauliZ(0))
  ```

<h3>Breaking changes</h3>

* The entry point name convention has changed.
  [(#493)](https://github.com/PennyLaneAI/catalyst/pull/493)

  From the [documentation](https://packaging.python.org/en/latest/specifications/entry-points/#data-model):

     Within a distribution, entry points should be unique.

  This meant that within a single distribution there cannot be two different `qjit` entry points like so:

  ```py
  entry_points={"pennylane.compilers": [
      "qjit = foo:Foo",
      "qjit = bar:Bar",
  ]}
  ```

  Now, entry point's names are preppended with the name of the compiler.

  ```py
  entry_points={"pennylane.compilers": [
      "compilerFoo.qjit = foo:Foo",
      "compilerBar.qjit = bar:Bar",
  ]}
  ```

  The behaviour will be reflected when using the `@qml.qjit` decorator.
  For example,

  ```py
  @qml.qjit(compiler="compilerFoo")
  def circuit(): ...
  ```

  will point call `foo:Foo` in the entry points' distribution.
  Please note that `compilerFoo` can also be `foo`. The compiler name
  does not have to be distinct from the object reference path. To keep
  the previous user facing behaviour, simply define entry points as follows:

  ```py
  entry_points={"pennylane.compilers": [
      "foo.qjit = foo:Foo",
      "bar.qjit = bar:Bar",
  ]}
  ```

  and the following example will still work:

  ```py
  @qml.qjit(compiler="foo")
  def circuit(): ...
  ```

* The Catalyst runtime now has a different API from QIR instructions.
  [(#464)](https://github.com/PennyLaneAI/catalyst/pull/464)

  QIR encodes quantum instructions as LLVM function calls. This allows frontends to generate
  QIR, middle-ends to optimizie QIR, and code generators to lower these function calls into
  appropriate instructions for the target. One of the possible implementations for QIR is to
  lower QIR instructions to function calls with the same signature and implement QIR as a library.
  Other implementations might choose to lower QIR to platform specific APIs, or replace function
  calls with semantically equivalent code.

  Catalyst implemented QIR as a library that can be linked against a QIR module.
  This works great when Catalyst is the only implementor of QIR.
  However, when other QIR implementors, who also lower implement a quantum runtime as functions to be
  linked against, this may generate symbol conflicts.

  This PR changes the runtime such that QIR instructions are now lowered to functions where
  the `__quantum__` part of the function name is replaced with `__catalyst__`. This prevents
  the possibility of symbol conflicts with other libraries that implement QIR as a library.
  However, it doesn't solve it completely. Since the `__catalyst__` functions are still exported.
  If another library implemented the same symbols exported by the runtime, the same problem would
  presist.

* The Catalyst runtime no longer depends on QIR runner's stdlib.
  [(#470)](https://github.com/PennyLaneAI/catalyst/pull/470)

  Similar to changing the runtime API for QIR instructions, we no longer depend nor link against
  QIR runner's stdlib. With PR #464, most of the symbol conflicts were resolved, but by linking
  against QIR runner's stdlib, some definitions persisted that may be different than ones
  used by third party implementors. To prevent symbol conflicts QIR runner's stdlib was removed
  and is no longer linked against. As a result, the following functions are now defined and
  implemented in Catalyst's runtime:
  * `int64_t __catalyst__rt__array_get_size_1d(QirArray *)`
  * `int8_t *__catalyst__rt__array_get_element_ptr_1d(QirArray *, int64_t)`

  and the following functions were removed since the frontend does not generate them
  * `QirString *__catalyst__rt__qubit_to_string(QUBIT *)`
  * `QirString *__catalyst__rt__result_to_string(RESULT *)`

<h3>Bug fixes</h3>

* Fix an issue when no qubit number was specified for the `qinst` primitive. The primitive now
  correctly deduces the number of qubits when no gate parameters are present. This change is not
  user facing.
  [(#496)](https://github.com/PennyLaneAI/catalyst/pull/496)

* Fix the scatter operation lowering when `updatedWindowsDim` is empty.
  [(#475)](https://github.com/PennyLaneAI/catalyst/pull/475)

* Fix the issue in `LightningKokkos::AllocateQubits` with allocating too many qubit IDs on
  qubit re-allocation.
  [(#473)](https://github.com/PennyLaneAI/catalyst/pull/473)

* Add the `wires` property to `catalyst.adjoint` and `catalyst.ctrl`.
  [(#480)](https://github.com/PennyLaneAI/catalyst/pull/480)

  Without implementing the `wires` property, users would get `<Wires = [<WiresEnum.AnyWires: -1>]>`
  for the list of wires in these operations.
  Currently, `catalyst.adjoint.wires` supports static wires and workflows with nested branches,
  and `catalyst.ctrl.wires` provides support for workflows with static and variable wires as well as
  nested branches. The `wires` property in `adjoint` and `ctrl` cannot be used in workflows with
  control flow operations.

* Add support for lowering the eigen vectors/values computation lapack method: `lapack_dsyevd`
  via `stablehlo.custom_call`. For Catalyst, it means that you now can QJIT compile eigen
  vector/values operations and any other `qml.math` methods that uses these operations.
  [(#488)](https://github.com/PennyLaneAI/catalyst/pull/488)

  For example, you can compile `qml.math.sqrt_matrix`:

  ```python
  @qml.qjit
  def workflow(A):
      B = qml.math.sqrt_matrix(A)
      return B @ A
  ```

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

# Release 0.4.1

<h3>Improvements</h3>

* Catalyst wheels are now packaged with OpenMP and ZStd, which avoids installing additional
  requirements separately in order to use pre-packaged Catalyst binaries.
  [(#457)](https://github.com/PennyLaneAI/catalyst/pull/457)
  [(#478)](https://github.com/PennyLaneAI/catalyst/pull/478)

  Note that OpenMP support for the `lightning.kokkos` backend has been disabled on macOS x86_64, due
  to memory issues in the computation of Lightning's adjoint-jacobian in the presence of multiple
  OMP threads.

<h3>Bug fixes</h3>

* Resolve an infinite recursion in the decomposition of the `Controlled`
  operator whenever computing a Unitary matrix for the operator fails.
  [(#468)](https://github.com/PennyLaneAI/catalyst/pull/468)

* Resolve a failure to generate gradient code for specific input circuits.
  [(#439)](https://github.com/PennyLaneAI/catalyst/pull/439)

  In this case, `jnp.mod`
  was used to compute wire values in a for loop, which prevented the gradient
  architecture from fully separating quantum and classical code. The following
  program is now supported:
  ```py
  @qjit
  @grad
  @qml.qnode(dev)
  def f(x):
      def cnot_loop(j):
          qml.CNOT(wires=[j, jnp.mod((j + 1), 4)])

      for_loop(0, 4, 1)(cnot_loop)()

      return qml.expval(qml.PauliZ(0))
  ```

* Resolve unpredictable behaviour when importing libraries that share Catalyst's LLVM dependency
  (e.g. TensorFlow). In some cases, both packages exporting the same symbols from their shared
  libraries can lead to process crashes and other unpredictable behaviour, since the wrong functions
  can be called if both libraries are loaded in the current process.
  The fix involves building shared libraries with hidden (macOS) or protected (linux) symbol
  visibility by default, exporting only what is necessary.
  [(#465)](https://github.com/PennyLaneAI/catalyst/pull/465)

* Resolve a failure to find the SciPy OpenBLAS library when running Catalyst,
  due to a different SciPy version being used to build Catalyst than to run it.
  [(#471)](https://github.com/PennyLaneAI/catalyst/pull/471)

* Resolve a memory leak in the runtime stemming from  missing calls to device destructors
  at the end of programs.
  [(#446)](https://github.com/PennyLaneAI/catalyst/pull/446)

<h3>Contributors</h3>

This release contains contributions from (in alphabetical order):

Ali Asadi,
David Ittah.

# Release 0.4.0

<h3>New features</h3>

* Catalyst is now accessible directly within the PennyLane user interface,
  once Catalyst is installed, allowing easy access to Catalyst just-in-time
  functionality.

  Through the use of the `qml.qjit` decorator, entire workflows can be JIT
  compiled down to a machine binary on first-function execution, including both quantum
  and classical processing. Subsequent calls to the compiled function will execute
  the previously-compiled binary, resulting in significant performance improvements.

  ```python
  import pennylane as qml

  dev = qml.device("lightning.qubit", wires=2)

  @qml.qjit
  @qml.qnode(dev)
  def circuit(theta):
      qml.Hadamard(wires=0)
      qml.RX(theta, wires=1)
      qml.CNOT(wires=[0, 1])
      return qml.expval(qml.PauliZ(wires=1))
  ```

  ```pycon
  >>> circuit(0.5)  # the first call, compilation occurs here
  array(0.)
  >>> circuit(0.5)  # the precompiled quantum function is called
  array(0.)
  ```

  Currently, PennyLane supports the [Catalyst hybrid compiler](https://github.com/pennylaneai/catalyst)
  with the `qml.qjit` decorator, which directly aliases Catalyst's `catalyst.qjit`.

  In addition to the above `qml.qjit` integration, the following native PennyLane functions can now
  be used with the `qjit` decorator: `qml.adjoint`, `qml.ctrl`, `qml.grad`, `qml.jacobian`,
  `qml.vjp`, `qml.jvp`, and `qml.adjoint`, `qml.while_loop`, `qml.for_loop`, `qml.cond`. These will
  alias to the corresponding Catalyst functions when used within a `qjit` context.

  For more details on these functions, please refer to the
 [PennyLane compiler documentation](https://docs.pennylane.ai/en/stable/introduction/compiling_workflows.html) and
 [compiler module documentation](https://docs.pennylane.ai/en/stable/code/qml_compiler.html).

* Just-in-time compiled functions now support asynchronuous execution of QNodes.
  [(#374)](https://github.com/PennyLaneAI/catalyst/pull/374)
  [(#381)](https://github.com/PennyLaneAI/catalyst/pull/381)
  [(#420)](https://github.com/PennyLaneAI/catalyst/pull/420)
  [(#424)](https://github.com/PennyLaneAI/catalyst/pull/424)
  [(#433)](https://github.com/PennyLaneAI/catalyst/pull/433)

  Simply specify `async_qnodes=True` when using the `@qjit` decorator to enable the async
  execution of QNodes. Currently, asynchronous execution is only supported by
  `lightning.qubit` and `lightning.kokkos`.

  Asynchronous execution will be most beneficial for just-in-time compiled functions that
  contain --- or generate --- multiple QNodes.

  For example,

  ```python
  dev = qml.device("lightning.qubit", wires=2)

  @qml.qnode(device=dev)
  def circuit(params):
      qml.RX(params[0], wires=0)
      qml.RY(params[1], wires=1)
      qml.CNOT(wires=[0, 1])
      return qml.expval(qml.PauliZ(wires=0))

  @qjit(async_qnodes=True)
  def multiple_qnodes(params):
      x = jnp.sin(params)
      y = jnp.cos(params)
      z = jnp.array([circuit(x), circuit(y)]) # will be executed in parallel
      return circuit(z)
  ```
  ``` pycon
  >>> func(jnp.array([1.0, 2.0]))
  1.0
  ```

  Here, the first two circuit executions will occur in parallel across multiple threads,
  as their execution can occur indepdently.

* Preliminary support for PennyLane transforms has been added.
  [(#280)](https://github.com/PennyLaneAI/catalyst/pull/280)

  ```python
  @qjit
  @qml.transforms.split_non_commuting
  @qml.qnode(dev)
  def circuit(x):
      qml.RX(x,wires=0)
      return [qml.expval(qml.PauliY(0)), qml.expval(qml.PauliZ(0))]
  ```

  ```pycon
  >>> circuit(0.4)
  [array(-0.51413599), array(0.85770868)]
  ```

  Currently, most PennyLane transforms will work with Catalyst
  as long as:

  - The circuit does not include any Catalyst-specific features, such
    as Catalyst control flow or measurement,

  - The QNode returns only lists of measurement processes,

  - AutoGraph is disabled, and

  - The transformation does not require or depend on the numeric value of
    dynamic variables.

* Catalyst now supports just-in-time compilation of dynamically-shaped arrays.
  [(#366)](https://github.com/PennyLaneAI/catalyst/pull/366)
  [(#386)](https://github.com/PennyLaneAI/catalyst/pull/385)
  [(#390)](https://github.com/PennyLaneAI/catalyst/pull/390)
  [(#411)](https://github.com/PennyLaneAI/catalyst/pull/411)

  The `@qjit` decorator can now be used to compile functions that accepts or contain tensors
  whose dimensions are not known at compile time; runtime execution with different shapes
  is supported without recompilation.

  In addition, standard tensor initialization functions `jax.numpy.ones`, `jnp.zeros`, and
  `jnp.empty` now accept dynamic variables (where the value is only known at
  runtime).

  ``` python
  @qjit
  def func(size: int):
      return jax.numpy.ones([size, size], dtype=float)
  ```

  ``` pycon
  >>> func(3)
  [[1. 1. 1.]
   [1. 1. 1.]
   [1. 1. 1.]]
  ```

  When passing tensors as arguments to compiled functions, the
  `abstracted_axes` keyword argument to the `@qjit` decorator can be used to specify
  which axes of the input arguments should be treated as abstract (and thus
  avoid recompilation).

  For example, without specifying `abstracted_axes`, the following `sum` function
  would recompile each time an array of different size is passed
  as an argument:

  ```pycon
  >>> @qjit
  >>> def sum_fn(x):
  >>>     return jnp.sum(x)
  >>> sum_fn(jnp.array([1]))     # Compilation happens here.
  >>> sum_fn(jnp.array([1, 1]))  # And here!
  ```

  By passing `abstracted_axes`, we can specify that the first axes
  of the first argument is to be treated as dynamic during initial compilation:

  ```pycon
  >>> @qjit(abstracted_axes={0: "n"})
  >>> def sum_fn(x):
  >>>     return jnp.sum(x)
  >>> sum_fn(jnp.array([1]))     # Compilation happens here.
  >>> sum_fn(jnp.array([1, 1]))  # No need to recompile.
  ```

  Note that support for dynamic arrays in control-flow primitives (such as loops),
  is not yet supported.

* Error mitigation using the zero-noise extrapolation method is now available through the
  `catalyst.mitigate_with_zne` transform.
  [(#324)](https://github.com/PennyLaneAI/catalyst/pull/324)
  [(#414)](https://github.com/PennyLaneAI/catalyst/pull/414)

  For example, given a noisy device (such as noisy hardware available through Amazon Braket):

  ```python
  dev = qml.device("noisy.device", wires=2)

  @qml.qnode(device=dev)
  def circuit(x, n):

      @for_loop(0, n, 1)
      def loop_rx(i):
          qml.RX(x, wires=0)

      loop_rx()

      qml.Hadamard(wires=0)
      qml.RZ(x, wires=0)
      loop_rx()
      qml.RZ(x, wires=0)
      qml.CNOT(wires=[1, 0])
      qml.Hadamard(wires=1)
      return qml.expval(qml.PauliY(wires=0))

  @qjit
  def mitigated_circuit(args, n):
      s = jax.numpy.array([1, 2, 3])
      return mitigate_with_zne(circuit, scale_factors=s)(args, n)
  ```

  ```pycon
  >>> mitigated_circuit(0.2, 5)
  0.5655341100116512
  ```

  In addition, a mitigation dialect has been added to the MLIR layer of Catalyst.
  It contains a Zero Noise Extrapolation (ZNE) operation,
  with a lowering to a global folded circuit.

<h3>Improvements</h3>

* The three backend devices provided with Catalyst, `lightning.qubit`, `lightning.kokkos`, and
  `braket.aws`, are now dynamically loaded at runtime.
  [(#343)](https://github.com/PennyLaneAI/catalyst/pull/343)
  [(#400)](https://github.com/PennyLaneAI/catalyst/pull/400)

  This takes advantage of the new backend plugin system provided in Catalyst v0.3.2,
  and allows the devices to be packaged separately from the runtime CAPI. Provided backend
  devices are now loaded at runtime, instead of being linked at compile time.

  For more details on the backend plugin system, see the
  [custom devices documentation](https://docs.pennylane.ai/projects/catalyst/en/stable/dev/custom_devices.html).

* Finite-shot measurement statistics (`expval`, `var`, and `probs`) are now supported
  for the `lightning.qubit` and `lightning.kokkos` devices. Previously, exact statistics
  were returned even when finite shots were specified.
  [(#392)](https://github.com/PennyLaneAI/catalyst/pull/392)
  [(#410)](https://github.com/PennyLaneAI/catalyst/pull/410)

  ```pycon
  >>> dev = qml.device("lightning.qubit", wires=2, shots=100)
  >>> @qjit
  >>> @qml.qnode(dev)
  >>> def circuit(x):
  >>>     qml.RX(x, wires=0)
  >>>     return qml.probs(wires=0)
  >>> circuit(0.54)
  array([0.94, 0.06])
  >>> circuit(0.54)
  array([0.93, 0.07])
  ```

* Catalyst gradient functions `grad`, `jacobian`, `jvp`, and `vjp` can now be invoked from
  outside a `@qjit` context.
  [(#375)](https://github.com/PennyLaneAI/catalyst/pull/375)

  This simplifies the process of writing functions where compilation
  can be turned on and off easily by adding or removing the decorator. The functions dispatch to
  their JAX equivalents when the compilation is turned off.

  ```python
  dev = qml.device("lightning.qubit", wires=2)

  @qml.qnode(dev)
  def circuit(x):
      qml.RX(x, wires=0)
      return qml.expval(qml.PauliZ(0))
  ```

  ```pycon
  >>> grad(circuit)(0.54)  # dispatches to jax.grad
  Array(-0.51413599, dtype=float64, weak_type=True)
  >>> qjit(grad(circuit))(0.54). # differentiates using Catalyst
  array(-0.51413599)
  ```

* New `lightning.qubit` configuration options are now supported via the `qml.device` loader,
  including Markov Chain Monte Carlo sampling support.
  [(#369)](https://github.com/PennyLaneAI/catalyst/pull/369)

  ```python
  dev = qml.device("lightning.qubit", wires=2, shots=1000, mcmc=True)

  @qml.qnode(dev)
  def circuit(x):
      qml.RX(x, wires=0)
      return qml.expval(qml.PauliZ(0))
  ```

  ```pycon
  >>> circuit(0.54)
  array(0.856)
  ```

* Improvements have been made to the runtime and quantum MLIR dialect in order
  to support asynchronous execution.

  - The runtime now supports multiple active devices managed via a device pool. The new `RTDevice`
    data-class and `RTDeviceStatus` along with the `thread_local` device instance pointer enable
    the runtime to better scope the lifetime of device instances concurrently. With these changes,
    one can create multiple active devices and execute multiple programs in a multithreaded
    environment.
    [(#381)](https://github.com/PennyLaneAI/catalyst/pull/381)

  - The ability to dynamically release devices has been added via `DeviceReleaseOp` in the Quantum
    MLIR dialect. This is lowered to the `__quantum__rt__device_release()` runtime instruction,
    which updates the status of the device instance from `Active` to `Inactive`. The runtime will reuse
    this deactivated instance instead of creating a new one automatically at runtime in a
    multi-QNode workflow when another device with identical specifications is requested.
    [(#381)](https://github.com/PennyLaneAI/catalyst/pull/381)

  - The `DeviceOp` definition in the Quantum MLIR dialect has been updated to lower a tuple
    of device information `('lib', 'name', 'kwargs')` to a single device initialization call
    `__quantum__rt__device_init(int8_t *, int8_t *, int8_t *)`. This allows the runtime to initialize
    device instances without keeping partial information of the device
    [(#396)](https://github.com/PennyLaneAI/catalyst/pull/396)

* The quantum adjoint compiler routine has been extended to support function calls that affect the
  quantum state within an adjoint region. Note that the function may only provide a single result
  consisting of the quantum register. By itself this provides no user-facing changes, but compiler
  pass developers may now generate quantum adjoint operations around a block of code containing
  function calls as well as quantum operations and control flow operations.
  [(#353)](https://github.com/PennyLaneAI/catalyst/pull/353)

* The allocation and deallocation operations in MLIR (`AllocOp`, `DeallocOp`) now follow simple
  value semantics for qubit register values, instead of modelling memory in the MLIR trait system.
  Similarly, the frontend generates proper value semantics by deallocating the final register value.

  The change enables functions at the MLIR level to accept and return quantum register values,
  which would otherwise not be correctly identified as aliases of existing register values by the
  bufferization system.
  [(#360)](https://github.com/PennyLaneAI/catalyst/pull/360)

<h3>Breaking changes</h3>

* Third party devices must now provide a configuration TOML file, in order to specify their
  supported operations, measurements, and features for Catalyst compatibility. For more information
  please visit the [Custom Devices](https://docs.pennylane.ai/projects/catalyst/en/latest/dev/custom_devices.html) section in our documentation.
  [(#369)](https://github.com/PennyLaneAI/catalyst/pull/369)

<h3>Bug fixes</h3>

* Resolves a bug in the compiler's differentiation engine that results in a segmentation fault
  when [attempting to differentiate non-differentiable quantum operations](https://github.com/PennyLaneAI/catalyst/issues/384).
  The fix ensures that all existing quantum operation types are removed during gradient passes that
  extract classical code from a QNode function. It also adds a verification step that will raise an error
  if a gradient pass cannot successfully eliminate all quantum operations for such functions.
  [(#397)](https://github.com/PennyLaneAI/catalyst/issues/397)

* Resolves a bug that caused unpredictable behaviour when printing string values with
  the `debug.print` function. The issue was caused by non-null-terminated strings.
  [(#418)](https://github.com/PennyLaneAI/catalyst/pull/418)

<h3>Contributors</h3>

This release contains contributions from (in alphabetical order):

Ali Asadi,
David Ittah,
Romain Moyard,
Sergei Mironov,
Erick Ochoa Lopez,
Shuli Shu.

# Release 0.3.2

<h3>New features</h3>

* The experimental AutoGraph feature now supports Python `while` loops, allowing native Python loops
  to be captured and compiled with Catalyst.
  [(#318)](https://github.com/PennyLaneAI/catalyst/pull/318)

  ```python
  dev = qml.device("lightning.qubit", wires=4)

  @qjit(autograph=True)
  @qml.qnode(dev)
  def circuit(n: int, x: float):
      i = 0

      while i < n:
          qml.RX(x, wires=i)
          i += 1

      return qml.expval(qml.PauliZ(0))
  ```

  ```pycon
  >>> circuit(4, 0.32)
  array(0.94923542)
  ```

  This feature extends the existing AutoGraph support for Python `for` loops and `if` statements
  introduced in v0.3. Note that TensorFlow must be installed for AutoGraph support.

  For more details, please see the
  [AutoGraph guide](https://docs.pennylane.ai/projects/catalyst/en/stable/dev/autograph.html).

* In addition to loops and conditional branches, AutoGraph now supports native Python `and`, `or`
  and `not` operators in Boolean expressions.
  [(#325)](https://github.com/PennyLaneAI/catalyst/pull/325)

  ```python
  dev = qml.device("lightning.qubit", wires=1)

  @qjit(autograph=True)
  @qml.qnode(dev)
  def circuit(x: float):

      if x >= 0 and x < jnp.pi:
          qml.RX(x, wires=0)

      return qml.probs()
  ```

  ```pycon
  >>> circuit(0.43)
  array([0.95448287, 0.04551713])
  >>> circuit(4.54)
  array([1., 0.])
  ```

  Note that logical Boolean operators will only be captured by AutoGraph if all
  operands are dynamic variables (that is, a value known only at runtime, such
  as a measurement result or function argument). For other use
  cases, it is recommended to use the `jax.numpy.logical_*` set of functions where
  appropriate.

* Debug compiled programs and print dynamic values at runtime with ``debug.print``
  [(#279)](https://github.com/PennyLaneAI/catalyst/pull/279)
  [(#356)](https://github.com/PennyLaneAI/catalyst/pull/356)

  You can now print arbitrary values from your running program, whether they are arrays, constants,
  strings, or abitrary Python objects. Note that while non-array Python objects
  *will* be printed at runtime, their string representation is captured at
  compile time, and thus will always be the same regardless of program inputs.
  The output for arrays optionally includes a descriptor for how the data is stored in memory
  ("memref").

  ```python
  @qjit
  def func(x: float):
      debug.print(x, memref=True)
      debug.print("exit")
  ```

  ```pycon
  >>> func(jnp.array(0.43))
  MemRef: base@ = 0x5629ff2b6680 rank = 0 offset = 0 sizes = [] strides = [] data =
  0.43
  exit
  ```

* Catalyst now officially supports macOS X86_64 devices, with macOS binary wheels
  available for both AARCH64 and X86_64.
  [(#347)](https://github.com/PennyLaneAI/catalyst/pull/347)
  [(#313)](https://github.com/PennyLaneAI/catalyst/pull/313)

* It is now possible to dynamically load third-party Catalyst compatible devices directly
  into a pre-installed Catalyst runtime on Linux.
  [(#327)](https://github.com/PennyLaneAI/catalyst/pull/327)

  To take advantage of this, third-party devices must implement the `Catalyst::Runtime::QuantumDevice`
  interface, in addition to defining the following method:

  ```cpp
  extern "C" Catalyst::Runtime::QuantumDevice*
  getCustomDevice() { return new CustomDevice(); }
  ```

  This support can also be integrated into existing PennyLane Python devices that inherit from
  the `QuantumDevice` class, by defining the `get_c_interface` static method.

  For more details, see the
  [custom devices documentation](https://docs.pennylane.ai/projects/catalyst/en/stable/dev/custom_devices.html).

<h3>Improvements</h3>

* Return values of conditional functions no longer need to be of exactly the same type.
  Type promotion is automatically applied to branch return values if their types don't match.
  [(#333)](https://github.com/PennyLaneAI/catalyst/pull/333)

  ```python
  @qjit
  def func(i: int, f: float):

      @cond(i < 3)
      def cond_fn():
          return i

      @cond_fn.otherwise
      def otherwise():
          return f

      return cond_fn()
  ```

  ```pycon
  >>> func(1, 4.0)
  array(1.0)
  ```

  Automatic type promotion across conditional branches also works with AutoGraph:

  ```python
  @qjit(autograph=True)
  def func(i: int, f: float):

      if i < 3:
          i = i
      else:
          i = f

      return i
  ```

  ```pycon
  >>> func(1, 4.0)
  array(1.0)
  ```

* AutoGraph now supports converting functions even when they are invoked through functional wrappers such
  as `adjoint`, `ctrl`, `grad`, `jacobian`, etc.
  [(#336)](https://github.com/PennyLaneAI/catalyst/pull/336)

  For example, the following should now succeed:

  ```python
  def inner(n):
    for i in range(n):
      qml.T(i)

  @qjit(autograph=True)
  @qml.qnode(dev)
  def f(n: int):
      adjoint(inner)(n)
      return qml.state()
  ```

* To prepare for Catalyst's frontend being integrated with PennyLane, the appropriate plugin entry point
  interface has been added to Catalyst.
  [(#331)](https://github.com/PennyLaneAI/catalyst/pull/331)

  For any compiler packages seeking to be registered in PennyLane, the `entry_points`
  metadata under the the group name `pennylane.compilers` must be added, with the following entry points:

  - `context`: Path to the compilation evaluation context manager. This context manager should have
    the method `context.is_tracing()`, which returns True if called within a program that is being
    traced or captured.

  - `ops`: Path to the compiler operations module. This operations module may contain compiler
    specific versions of PennyLane operations. Within a JIT context, PennyLane operations may
    dispatch to these.

  - `qjit`: Path to the JIT compiler decorator provided by the compiler. This decorator should have
    the signature `qjit(fn, *args, **kwargs)`, where `fn` is the function to be compiled.

* The compiler driver diagnostic output has been improved, and now includes failing IR as well as
  the names of failing passes.
  [(#349)](https://github.com/PennyLaneAI/catalyst/pull/349)

* The scatter operation in the Catalyst dialect now uses an SCF for loop to avoid ballooning
  the compiled code.
  [(#307)](https://github.com/PennyLaneAI/catalyst/pull/307)

* The `CopyGlobalMemRefPass` pass of our MLIR processing pipeline now supports
  dynamically shaped arrays.
  [(#348)](https://github.com/PennyLaneAI/catalyst/pull/348)

* The Catalyst utility dialect is now included in the Catalyst MLIR C-API.
  [(#345)](https://github.com/PennyLaneAI/catalyst/pull/345)

* Fix an issue with the AutoGraph conversion system that would prevent the fallback to Python from
  working correctly in certain instances.
  [(#352)](https://github.com/PennyLaneAI/catalyst/pull/352)

  The following type of code is now supported:

  ```python
  @qjit(autograph=True)
  def f():
    l = jnp.array([1, 2])
    for _ in range(2):
        l = jnp.kron(l, l)
    return l
  ```

* Catalyst now supports `jax.numpy.polyfit` inside a qjitted function.
  [(#367)](https://github.com/PennyLaneAI/catalyst/pull/367/)

* Catalyst now supports custom calls (including the one from HLO). We added support in MLIR (operation, bufferization
  and lowering). In the `lib_custom_calls`, developers then implement their custom calls and use external functions
  directly (e.g. Lapack). The OpenBlas library is taken from Scipy and linked in Catalyst, therefore any function from
  it can be used.
  [(#367)](https://github.com/PennyLaneAI/catalyst/pull/367/)

<h3>Breaking changes</h3>

* The axis ordering for `catalyst.jacobian` is updated to match `jax.jacobian`. Assuming we have
  parameters of shape `[a,b]` and results of shape `[c,d]`, the returned Jacobian will now have
  shape `[c, d, a, b]` instead of `[a, b, c, d]`.
  [(#283)](https://github.com/PennyLaneAI/catalyst/pull/283)

<h3>Bug fixes</h3>

* An upstream change in the PennyLane-Lightning project was addressed to prevent compilation issues
  in the `StateVectorLQubitDynamic` class in the runtime.
  The issue was introduced in [#499](https://github.com/PennyLaneAI/pennylane-lightning/pull/499).
  [(#322)](https://github.com/PennyLaneAI/catalyst/pull/322)

* The `requirements.txt` file to build Catalyst from source has been updated with a minimum pip
  version, `>=22.3`. Previous versions of pip are unable to perform editable installs when the
  system-wide site-packages are read-only, even when the `--user` flag is provided.
  [(#311)](https://github.com/PennyLaneAI/catalyst/pull/311)

* The frontend has been updated to make it compatible with PennyLane `MeasurementProcess` objects
  now being PyTrees in PennyLane version 0.33.
  [(#315)](https://github.com/PennyLaneAI/catalyst/pull/315)

<h3>Contributors</h3>

This release contains contributions from (in alphabetical order):

Ali Asadi,
David Ittah,
Sergei Mironov,
Romain Moyard,
Erick Ochoa Lopez.

# Release 0.3.1

<h3>New features</h3>

* The experimental AutoGraph feature, now supports Python `for` loops, allowing native Python loops
  to be captured and compiled with Catalyst.
  [(#258)](https://github.com/PennyLaneAI/catalyst/pull/258)

  ```python
  dev = qml.device("lightning.qubit", wires=n)

  @qjit(autograph=True)
  @qml.qnode(dev)
  def f(n):
      for i in range(n):
          qml.Hadamard(wires=i)

      return qml.expval(qml.PauliZ(0))
  ```

  This feature extends the existing AutoGraph support for Python `if` statements introduced in v0.3.
  Note that TensorFlow must be installed for AutoGraph support.

* The quantum control operation can now be used in conjunction with Catalyst control flow, such as
  loops and conditionals, via the new `catalyst.ctrl` function.
  [(#282)](https://github.com/PennyLaneAI/catalyst/pull/282)

  Similar in behaviour to the `qml.ctrl` control modifier from PennyLane, `catalyst.ctrl` can
  additionally wrap around quantum functions which contain control flow, such as the Catalyst
  `cond`, `for_loop`, and `while_loop` primitives.

  ```python
  @qjit
  @qml.qnode(qml.device("lightning.qubit", wires=4))
  def circuit(x):

      @for_loop(0, 3, 1)
      def repeat_rx(i):
          qml.RX(x / 2, wires=i)

      catalyst.ctrl(repeat_rx, control=3)()

      return qml.expval(qml.PauliZ(0))
  ```

  ```pycon
  >>> circuit(0.2)
  array(1.)
  ```

* Catalyst now supports JAX's `array.at[index]` notation for array element assignment and updating.
  [(#273)](https://github.com/PennyLaneAI/catalyst/pull/273)

  ```python
  @qjit
  def add_multiply(l: jax.core.ShapedArray((3,), dtype=float), idx: int):
      res = l.at[idx].multiply(3)
      res2 = l.at[idx].add(2)
      return res + res2

  res = add_multiply(jnp.array([0, 1, 2]), 2)
  ```

  ```pycon
  >>> res
  [0, 2, 10]
  ```

  For more details on available methods, see the
  [JAX documentation](https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.ndarray.at.html).

<h3>Improvements</h3>

* The Lightning backend device has been updated to work with the new PL-Lightning monorepo.
  [(#259)](https://github.com/PennyLaneAI/catalyst/pull/259)
  [(#277)](https://github.com/PennyLaneAI/catalyst/pull/277)

* A new compiler driver has been implemented in C++. This improves compile-time performance by
  avoiding *round-tripping*, which is when the entire program being compiled is dumped to
  a textual form and re-parsed by another tool.

  This is also a requirement for providing custom metadata at the LLVM level, which is
  necessary for better integration with tools like Enzyme. Finally, this makes it more natural
  to improve error messages originating from C++ when compared to the prior subprocess-based
  approach.
  [(#216)](https://github.com/PennyLaneAI/catalyst/pull/216)

* Support the `braket.devices.Devices` enum class and `s3_destination_folder`
  device options for AWS Braket remote devices.
  [(#278)](https://github.com/PennyLaneAI/catalyst/pull/278)

* Improvements have been made to the build process, including avoiding unnecessary processes such
  as removing `opt` and downloading the wheel.
  [(#298)](https://github.com/PennyLaneAI/catalyst/pull/298)

* Remove a linker warning about duplicate `rpath`s when Catalyst wheels are installed on macOS.
  [(#314)](https://github.com/PennyLaneAI/catalyst/pull/314)

<h3>Bug fixes</h3>

* Fix incompatibilities with GCC on Linux introduced in v0.3.0 when compiling user programs.
  Due to these, Catalyst v0.3.0 only works when clang is installed in the user environment.

  - Resolve an issue with an empty linker flag, causing `ld` to error.
    [(#276)](https://github.com/PennyLaneAI/catalyst/pull/276)

  - Resolve an issue with undefined symbols provided the Catalyst runtime.
    [(#316)](https://github.com/PennyLaneAI/catalyst/pull/316)

* Remove undocumented package dependency on the zlib/zstd compression library.
  [(#308)](https://github.com/PennyLaneAI/catalyst/pull/308)

* Fix filesystem issue when compiling multiple functions with the same name and
  `keep_intermediate=True`.
  [(#306)](https://github.com/PennyLaneAI/catalyst/pull/306)

* Add support for applying the `adjoint` operation to `QubitUnitary` gates.
  `QubitUnitary` was not able to be `adjoint`ed when the variable holding the unitary matrix might
  change. This can happen, for instance, inside of a for loop.
  To solve this issue, the unitary matrix gets stored in the array list via push and pops.
  The unitary matrix is later reconstructed from the array list and `QubitUnitary` can be executed
  in the `adjoint`ed context.
  [(#304)](https://github.com/PennyLaneAI/catalyst/pull/304)
  [(#310)](https://github.com/PennyLaneAI/catalyst/pull/310)

<h3>Contributors</h3>

This release contains contributions from (in alphabetical order):

Ali Asadi,
David Ittah,
Erick Ochoa Lopez,
Jacob Mai Peng,
Sergei Mironov,
Romain Moyard.

# Release 0.3.0

<h3>New features</h3>

* Catalyst now officially supports macOS ARM devices, such as Apple M1/M2 machines,
  with macOS binary wheels available on PyPI. For more details on the changes involved to support
  macOS, please see the improvements section.
  [(#229)](https://github.com/PennyLaneAI/catalyst/pull/230)
  [(#232)](https://github.com/PennyLaneAI/catalyst/pull/232)
  [(#233)](https://github.com/PennyLaneAI/catalyst/pull/233)
  [(#234)](https://github.com/PennyLaneAI/catalyst/pull/234)

* Write Catalyst-compatible programs with native Python conditional statements.
  [(#235)](https://github.com/PennyLaneAI/catalyst/pull/235)

  AutoGraph is a new, experimental, feature that automatically converts Python conditional
  statements like `if`, `else`, and `elif`, into their equivalent functional forms provided by
  Catalyst (such as `catalyst.cond`).

  This feature is currently opt-in, and requires setting the `autograph=True` flag in the `qjit`
  decorator:

  ```python
  dev = qml.device("lightning.qubit", wires=1)

  @qjit(autograph=True)
  @qml.qnode(dev)
  def f(x):
      if x < 0.5:
          qml.RY(jnp.sin(x), wires=0)
      else:
          qml.RX(jnp.cos(x), wires=0)

      return qml.expval(qml.PauliZ(0))
  ```

  The implementation is based on the AutoGraph module from TensorFlow, and requires a working
  TensorFlow installation be available. In addition, Python loops (`for` and `while`) are not
  yet supported, and do not work in AutoGraph mode.

  Note that there are some caveats when using this feature especially around the use of global
  variables or object mutation inside of methods. A functional style is always recommended when
  using `qjit` or AutoGraph.

* The quantum adjoint operation can now be used in conjunction with Catalyst control flow, such as
  loops and conditionals. For this purpose a new instruction, `catalyst.adjoint`, has been added.
  [(#220)](https://github.com/PennyLaneAI/catalyst/pull/220)

  `catalyst.adjoint` can wrap around quantum functions which contain the Catalyst `cond`,
  `for_loop`, and `while_loop` primitives. Previously, the usage of `qml.adjoint` on functions with
  these primitives would result in decomposition errors. Note that a future release of Catalyst will
  merge the behaviour of `catalyst.adjoint` into `qml.adjoint` for convenience.

  ```python
  dev = qml.device("lightning.qubit", wires=3)

  @qjit
  @qml.qnode(dev)
  def circuit(x):

      @for_loop(0, 3, 1)
      def repeat_rx(i):
          qml.RX(x / 2, wires=i)

      adjoint(repeat_rx)()

      return qml.expval(qml.PauliZ(0))
  ```

  ```pycon
  >>> circuit(0.2)
  array(0.99500417)
  ```

  Additionally, the ability to natively represent the adjoint construct in Catalyst's program
  representation (IR) was added.

* QJIT-compiled programs now support (nested) container types as inputs and outputs of compiled
  functions. This includes lists and dictionaries, as well as any data structure implementing the
  [PyTree protocol](https://jax.readthedocs.io/en/latest/pytrees.html).
  [(#215)](https://github.com/PennyLaneAI/catalyst/pull/215)
  [(#221)](https://github.com/PennyLaneAI/catalyst/pull/221)

  For example, a program that accepts and returns a mix of dictionaries, lists, and tuples:

  ```python
  @qjit
  def workflow(params1, params2):
      res1 = params1["a"][0][0] + params2[1]
      return {"y1": jnp.sin(res1), "y2": jnp.cos(res1)}
  ```

  ```pycon
  >>> params1 = {"a": [[0.1], 0.2]}
  >>> params2 = (0.6, 0.8)
  >>> workflow(params1, params2)
  array(0.78332691)
  ```

* Compile-time backpropagation of arbitrary hybrid programs is now supported, via integration with
  [Enzyme AD](https://enzyme.mit.edu/).
  [(#158)](https://github.com/PennyLaneAI/catalyst/pull/158)
  [(#193)](https://github.com/PennyLaneAI/catalyst/pull/193)
  [(#224)](https://github.com/PennyLaneAI/catalyst/pull/224)
  [(#225)](https://github.com/PennyLaneAI/catalyst/pull/225)
  [(#239)](https://github.com/PennyLaneAI/catalyst/pull/239)
  [(#244)](https://github.com/PennyLaneAI/catalyst/pull/244)

  This allows `catalyst.grad` to differentiate hybrid functions that contain both classical
  pre-processing (inside & outside of QNodes), QNodes, as well as classical post-processing
  (outside of QNodes) via a combination of backpropagation and quantum gradient methods.

  The new default for the differentiation `method` attribute in `catalyst.grad` has been changed to
  `"auto"`, which performs Enzyme-based reverse mode AD on classical code, in conjunction with the
  quantum `diff_method` specified on each QNode:

  ```python
  dev = qml.device("lightning.qubit", wires=1)

  @qml.qnode(dev, diff_method="parameter-shift")
  def circuit(theta):
      qml.RX(jnp.exp(theta ** 2) / jnp.cos(theta / 4), wires=0)
      return qml.expval(qml.PauliZ(wires=0))
  ```

  ```pycon
  >>> grad = qjit(catalyst.grad(circuit, method="auto"))
  >>> grad(jnp.pi)
  array(0.05938718)
  ```

  The reworked differentiation pipeline means you can now compute exact derivatives of programs with
  both classical pre- and post-processing, as shown below:

  ```python
  @qml.qnode(qml.device("lightning.qubit", wires=1), diff_method="adjoint")
  def circuit(theta):
      qml.RX(jnp.exp(theta ** 2) / jnp.cos(theta / 4), wires=0)
      return qml.expval(qml.PauliZ(wires=0))

  def loss(theta):
      return jnp.pi / jnp.tanh(circuit(theta))

  @qjit
  def grad_loss(theta):
      return catalyst.grad(loss)(theta)
  ```

  ```pycon
  >>> grad_loss(1.0)
  array(-1.90958669)
  ```

  You can also use multiple QNodes with different differentiation methods:

  ```python
  @qml.qnode(qml.device("lightning.qubit", wires=1), diff_method="parameter-shift")
  def circuit_A(params):
      qml.RX(jnp.exp(params[0] ** 2) / jnp.cos(params[1] / 4), wires=0)
      return qml.probs()

  @qml.qnode(qml.device("lightning.qubit", wires=1), diff_method="adjoint")
  def circuit_B(params):
      qml.RX(jnp.exp(params[1] ** 2) / jnp.cos(params[0] / 4), wires=0)
      return qml.expval(qml.PauliZ(wires=0))

  def loss(params):
      return jnp.prod(circuit_A(params)) + circuit_B(params)

  @qjit
  def grad_loss(theta):
      return catalyst.grad(loss)(theta)
  ```

  ```pycon
  >>> grad_loss(jnp.array([1.0, 2.0]))
  array([ 0.57367285, 44.4911605 ])
  ```

  And you can differentiate purely classical functions as well:

  ```python
  def square(x: float):
      return x ** 2

  @qjit
  def dsquare(x: float):
      return catalyst.grad(square)(x)
  ```

  ```pycon
  >>> dsquare(2.3)
  array(4.6)
  ```

  Note that the current implementation of reverse mode AD is restricted to 1st order derivatives,
  but you can still use `catalyst.grad(method="fd")` is still available to perform a finite
  differences approximation of _any_ differentiable function.

* Add support for the new PennyLane arithmetic operators.
  [(#250)](https://github.com/PennyLaneAI/catalyst/pull/250)

  PennyLane is in the process of replacing `Hamiltonian` and `Tensor` observables with a set of
  general arithmetic operators. These consist of
  [Prod](https://docs.pennylane.ai/en/stable/code/api/pennylane.ops.op_math.Prod.html),
  [Sum](https://docs.pennylane.ai/en/stable/code/api/pennylane.ops.op_math.Sum.html) and
  [SProd](https://docs.pennylane.ai/en/stable/code/api/pennylane.ops.op_math.SProd.html).

  By default, using dunder methods (eg. `+`, `-`, `@`, `*`) to combine
  operators with scalars or other operators will create `Hamiltonian` and
  `Tensor` objects. However, these two methods will be deprecated in coming
  releases of PennyLane.

  To enable the new arithmetic operators, one can use `Prod`, `Sum`, and
  `Sprod` directly or activate them by calling [enable_new_opmath](https://docs.pennylane.ai/en/stable/code/api/pennylane.operation.enable_new_opmath.html)
  at the beginning of your PennyLane program.

  ``` python
  dev = qml.device("lightning.qubit", wires=2)

  @qjit
  @qml.qnode(dev)
  def circuit(x: float, y: float):
      qml.RX(x, wires=0)
      qml.RX(y, wires=1)
      qml.CNOT(wires=[0, 1])
      return qml.expval(0.2 * qml.PauliX(wires=0) - 0.4 * qml.PauliY(wires=1))
  ```

  ```pycon
  >>> qml.operation.enable_new_opmath()
  >>> qml.operation.active_new_opmath()
  True
  >>> circuit(np.pi / 4, np.pi / 2)
  array(0.28284271)
  ```

<h3>Improvements</h3>

* Better support for Hamiltonian observables:

  - Allow Hamiltonian observables with integer coefficients.
    [(#248)](https://github.com/PennyLaneAI/catalyst/pull/248)

    For example, compiling the following circuit wasn't previously allowed, but is
    now supported in Catalyst:

    ```python
    dev = qml.device("lightning.qubit", wires=2)

    @qjit
    @qml.qnode(dev)
    def circuit(x: float, y: float):
        qml.RX(x, wires=0)
        qml.RY(y, wires=1)

        coeffs = [1, 2]
        obs = [qml.PauliZ(0), qml.PauliZ(1)]
        return qml.expval(qml.Hamiltonian(coeffs, obs))
    ```

  - Allow nested Hamiltonian observables.
    [(#255)](https://github.com/PennyLaneAI/catalyst/pull/255)

    ```python
    @qjit
    @qml.qnode(qml.device("lightning.qubit", wires=3))
    def circuit(x, y, coeffs1, coeffs2):
        qml.RX(x, wires=0)
        qml.RX(y, wires=1)
        qml.RY(x + y, wires=2)

        obs = [
            qml.PauliX(0) @ qml.PauliZ(1),
            qml.Hamiltonian(coeffs1, [qml.PauliZ(0) @ qml.Hadamard(2)]),
        ]

        return qml.var(qml.Hamiltonian(coeffs2, obs))
    ```

* Various performance improvements:

  - The execution and compile time of programs has been reduced, by generating more efficient code
    and avoiding unnecessary optimizations. Specifically, a scalarization procedure was added to the
    MLIR pass pipeline, and LLVM IR compilation is now invoked with optimization level 0.
    [(#217)](https://github.com/PennyLaneAI/catalyst/pull/217)

  - The execution time of compiled functions has been improved in the frontend.
    [(#213)](https://github.com/PennyLaneAI/catalyst/pull/213)

    Specifically, the following changes have been made, which leads to a small but measurable
    improvement when using larger matrices as inputs, or functions with many inputs:

    + only loading the user program library once per compilation,
    + generating return value types only once per compilation,
    + avoiding unnecessary type promotion, and
    + avoiding unnecessary array copies.

  - Peak memory utilization of a JIT compiled program has been reduced, by allowing tensors to be
    scheduled for deallocation. Previously, the tensors were not deallocated until the end of the
    call to the JIT compiled function.
    [(#201)](https://github.com/PennyLaneAI/catalyst/pull/201)

* Various improvements have been made to enable Catalyst to compile on macOS:

  - Remove unnecessary `reinterpret_cast` from `ObsManager`. Removal of
    these `reinterpret_cast` allows compilation of the runtime to succeed
    in macOS. macOS uses an ILP32 mode for Aarch64 where they use the full 64
    bit mode but with 32 bit Integer, Long, and Pointers. This patch also
    changes a test file to prevent a mismatch in machines which compile using
    ILP32 mode.
    [(#229)](https://github.com/PennyLaneAI/catalyst/pull/230)

  - Allow runtime to be compiled on macOS. Substitute `nproc` with a call to
    `os.cpu_count()` and use correct flags for `ld.64`.
    [(#232)](https://github.com/PennyLaneAI/catalyst/pull/232)

  - Improve portability on the frontend to be available on macOS. Use
    `.dylib`, remove unnecessary flags, and address behaviour difference in
    flags.
    [(#233)](https://github.com/PennyLaneAI/catalyst/pull/233)

  - Small compatibility changes in order for all integration tests to succeed
    on macOS.
    [(#234)](https://github.com/PennyLaneAI/catalyst/pull/234)

* Dialects can compile with older versions of clang by avoiding type mismatches.
  [(#228)](https://github.com/PennyLaneAI/catalyst/pull/228)

* The runtime is now built against `qir-stdlib` pre-build artifacts.
  [(#236)](https://github.com/PennyLaneAI/catalyst/pull/236)

* Small improvements have been made to the CI/CD, including fixing the Enzyme
  cache, generalize caches to other operating systems, fix build wheel
  recipe, and remove references to QIR in runtime's Makefile.
  [(#243)](https://github.com/PennyLaneAI/catalyst/pull/243)
  [(#247)](https://github.com/PennyLaneAI/catalyst/pull/247)


<h3>Breaking changes</h3>

* Support for Python 3.8 has been removed.
  [(#231)](https://github.com/PennyLaneAI/catalyst/pull/231)

* The default differentiation method on ``grad`` and ``jacobian`` is reverse-mode
  automatic differentiation instead of finite differences. When a QNode does not have a
  ``diff_method`` specified, it will default to using the parameter shift method instead of
  finite-differences.
  [(#244)](https://github.com/PennyLaneAI/catalyst/pull/244)
  [(#271)](https://github.com/PennyLaneAI/catalyst/pull/271)

* The JAX version used by Catalyst has been updated to `v0.4.14`, the minimum PennyLane version
  required is now `v0.32`.
  [(#264)](https://github.com/PennyLaneAI/catalyst/pull/264)

* Due to the change allowing Python container objects as inputs to QJIT-compiled functions, Python
  lists are no longer automatically converted to JAX arrays.
  [(#231)](https://github.com/PennyLaneAI/catalyst/pull/231)

  This means that indexing on lists when the index is not static will cause a
  `TracerIntegerConversionError`, consistent with JAX's behaviour.

  That is, the following example is no longer support:

  ```python
  @qjit
  def f(x: list, index: int):
      return x[index]
  ```

  However, if the parameter `x` above is a JAX or NumPy array, the compilation will continue to
  succeed.

* The `catalyst.grad` function has been renamed to `catalyst.jacobian` and supports differentiation
  of functions that return multiple or non-scalar outputs. A new `catalyst.grad` function has been
  added that enforces that it is differentiating a function with a single scalar return value.
  [(#254)](https://github.com/PennyLaneAI/catalyst/pull/254)

<h3>Bug fixes</h3>

* Fixed an issue preventing the differentiation of `qml.probs` with the parameter-shift method.
  [(#211)](https://github.com/PennyLaneAI/catalyst/pull/211)

* Fixed the incorrect return value data-type with functions returning `qml.counts`.
  [(#221)](https://github.com/PennyLaneAI/catalyst/pull/221)

* Fix segmentation fault when differentiating a function where a quantum measurement is used
  multiple times by the same operation.
  [(#242)](https://github.com/PennyLaneAI/catalyst/pull/242)

<h3>Contributors</h3>

This release contains contributions from (in alphabetical order):

Ali Asadi,
David Ittah,
Erick Ochoa Lopez,
Jacob Mai Peng,
Romain Moyard,
Sergei Mironov.


# Release 0.2.1

<h3>Bug fixes</h3>

* Add missing OpenQASM backend in binary distribution, which relies on the latest version of the
  AWS Braket plugin for PennyLane to resolve dependency issues between the plugin, Catalyst, and
  PennyLane. The Lightning-Kokkos backend with Serial and OpenMP modes is also added to the binary
  distribution.
  [#198](https://github.com/PennyLaneAI/catalyst/pull/198)

* Return a list of decompositions when calling the decomposition method for control operations.
  This allows Catalyst to be compatible with upstream PennyLane.
  [#241](https://github.com/PennyLaneAI/catalyst/pull/241)

<h3>Improvements</h3>

* When using OpenQASM-based devices the string representation of the circuit is printed on
  exception.
  [#199](https://github.com/PennyLaneAI/catalyst/pull/199)

* Use ``pybind11::module`` interface library instead of ``pybind11::embed`` in the runtime for
  OpenQasm backend to avoid linking to the python library at compile time.
  [#200](https://github.com/PennyLaneAI/catalyst/pull/200)

<h3>Contributors</h3>

This release contains contributions from (in alphabetical order):

Ali Asadi,
David Ittah.

# Release 0.2.0

<h3>New features</h3>

* Catalyst programs can now be used inside of a larger JAX workflow which uses
  JIT compilation, automatic differentiation, and other JAX transforms.
  [#96](https://github.com/PennyLaneAI/catalyst/pull/96)
  [#123](https://github.com/PennyLaneAI/catalyst/pull/123)
  [#167](https://github.com/PennyLaneAI/catalyst/pull/167)
  [#192](https://github.com/PennyLaneAI/catalyst/pull/192)

  For example, call a Catalyst qjit-compiled function from within a JAX jit-compiled
  function:

  ```python
  dev = qml.device("lightning.qubit", wires=1)

  @qjit
  @qml.qnode(dev)
  def circuit(x):
      qml.RX(jnp.pi * x[0], wires=0)
      qml.RY(x[1] ** 2, wires=0)
      qml.RX(x[1] * x[2], wires=0)
      return qml.probs(wires=0)

  @jax.jit
  def cost_fn(weights):
      x = jnp.sin(weights)
      return jnp.sum(jnp.cos(circuit(x)) ** 2)
  ```

  ```pycon
  >>> cost_fn(jnp.array([0.1, 0.2, 0.3]))
  Array(1.32269195, dtype=float64)
  ```

  Catalyst-compiled functions can now also be automatically differentiated
  via JAX, both in forward and reverse mode to first-order,

  ```pycon
  >>> jax.grad(cost_fn)(jnp.array([0.1, 0.2, 0.3]))
  Array([0.49249037, 0.05197949, 0.02991883], dtype=float64)
  ```

  as well as vectorized using `jax.vmap`:

  ```pycon
  >>> jax.vmap(cost_fn)(jnp.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]))
  Array([1.32269195, 1.53905377], dtype=float64)
  ```

  In particular, this allows for a reduction in boilerplate when using
  JAX-compatible optimizers such as `jaxopt`:

  ```pycon
  >>> opt = jaxopt.GradientDescent(cost_fn)
  >>> params = jnp.array([0.1, 0.2, 0.3])
  >>> (final_params, _) = jax.jit(opt.run)(params)
  >>> final_params
  Array([-0.00320799,  0.03475223,  0.29362844], dtype=float64)
  ```

  Note that, in general, best performance will be seen when the Catalyst
  `@qjit` decorator is used to JIT the entire hybrid workflow. However, there
  may be cases where you may want to delegate only the quantum part of your
  workflow to Catalyst, and let JAX handle classical components (for example,
  due to missing a feature or compatibility issue in Catalyst).

* Support for Amazon Braket devices provided via the PennyLane-Braket plugin.
  [#118](https://github.com/PennyLaneAI/catalyst/pull/118)
  [#139](https://github.com/PennyLaneAI/catalyst/pull/139)
  [#179](https://github.com/PennyLaneAI/catalyst/pull/179)
  [#180](https://github.com/PennyLaneAI/catalyst/pull/180)

  This enables quantum subprograms within a JIT-compiled Catalyst workflow to
  execute on Braket simulator and hardware devices, including remote
  cloud-based simulators such as SV1.

  ```python
  def circuit(x, y):
      qml.RX(y * x, wires=0)
      qml.RX(x * 2, wires=1)
      return qml.expval(qml.PauliY(0) @ qml.PauliZ(1))

  @qjit
  def workflow(x: float, y: float):
      device = qml.device("braket.local.qubit", backend="braket_sv", wires=2)
      g = qml.qnode(device)(circuit)
      h = catalyst.grad(g)
      return h(x, y)

  workflow(1.0, 2.0)
  ```

  For a list of available devices, please see the [PennyLane-Braket](https://amazon-braket-pennylane-plugin-python.readthedocs.io/en/latest/)
  documentation.

  Internally, the quantum instructions are generating OpenQASM3 kernels at
  runtime; these are then executed on both local (`braket.local.qubit`) and
  remote (`braket.aws.qubit`) devices backed by Amazon Braket Python SDK,

  with measurement results then propagated back to the frontend.

  Note that at initial release, not all Catalyst features are supported with Braket.
  In particular, dynamic circuit features, such as mid-circuit measurements, will
  not work with Braket devices.

* Catalyst conditional functions defined via `@catalyst.cond` now support an arbitrary
  number of 'else if' chains.
  [#104](https://github.com/PennyLaneAI/catalyst/pull/104)

  ```python
  dev = qml.device("lightning.qubit", wires=1)

  @qjit
  @qml.qnode(dev)
  def circuit(x):

      @catalyst.cond(x > 2.7)
      def cond_fn():
          qml.RX(x, wires=0)

      @cond_fn.else_if(x > 1.4)
      def cond_elif():
          qml.RY(x, wires=0)

      @cond_fn.otherwise
      def cond_else():
          qml.RX(x ** 2, wires=0)

      cond_fn()

      return qml.probs(wires=0)
  ```

* Iterating in reverse is now supported with constant negative step sizes via `catalyst.for_loop`.
  [#129](https://github.com/PennyLaneAI/catalyst/pull/129)

  ```python
  dev = qml.device("lightning.qubit", wires=1)

  @qjit
  @qml.qnode(dev)
  def circuit(n):

      @catalyst.for_loop(n, 0, -1)
      def loop_fn(_):
          qml.PauliX(0)

      loop_fn()
      return measure(0)
  ```

* Additional gradient transforms for computing the vector-Jacobian product (VJP)
  and Jacobian-vector product (JVP) are now available in Catalyst.
  [#98](https://github.com/PennyLaneAI/catalyst/pull/98)

  Use `catalyst.vjp` to compute the forward-pass value and VJP:

  ```python
  @qjit
  def vjp(params, cotangent):
      def f(x):
          y = [jnp.sin(x[0]), x[1] ** 2, x[0] * x[1]]
          return jnp.stack(y)

      return catalyst.vjp(f, [params], [cotangent])
  ```

  ```pycon
  >>> x = jnp.array([0.1, 0.2])
  >>> dy = jnp.array([-0.5, 0.1, 0.3])
  >>> vjp(x, dy)
  [array([0.09983342, 0.04      , 0.02      ]),
   array([-0.43750208,  0.07000001])]
  ```

  Use `catalyst.jvp` to compute the forward-pass value and JVP:

  ```python
  @qjit
  def jvp(params, tangent):
      def f(x):
          y = [jnp.sin(x[0]), x[1] ** 2, x[0] * x[1]]
          return jnp.stack(y)

      return catalyst.jvp(f, [params], [tangent])
  ```

  ```pycon
  >>> x = jnp.array([0.1, 0.2])
  >>> tangent = jnp.array([0.3, 0.6])
  >>> jvp(x, tangent)
  [array([0.09983342, 0.04      , 0.02      ]),
   array([0.29850125, 0.24000006, 0.12      ])]
  ```

* Support for multiple backend devices within a single qjit-compiled function
  is now available.
  [#86](https://github.com/PennyLaneAI/catalyst/pull/86)
  [#89](https://github.com/PennyLaneAI/catalyst/pull/89)

  For example, if you compile the Catalyst runtime
  with `lightning.kokkos` support (via the compilation flag
  `ENABLE_LIGHTNING_KOKKOS=ON`), you can use `lightning.qubit` and
  `lightning.kokkos` within a singular workflow:

  ```python
  dev1 = qml.device("lightning.qubit", wires=1)
  dev2 = qml.device("lightning.kokkos", wires=1)

  @qml.qnode(dev1)
  def circuit1(x):
      qml.RX(jnp.pi * x[0], wires=0)
      qml.RY(x[1] ** 2, wires=0)
      qml.RX(x[1] * x[2], wires=0)
      return qml.var(qml.PauliZ(0))

  @qml.qnode(dev2)
  def circuit2(x):

      @catalyst.cond(x > 2.7)
      def cond_fn():
          qml.RX(x, wires=0)

      @cond_fn.otherwise
      def cond_else():
          qml.RX(x ** 2, wires=0)

      cond_fn()

      return qml.probs(wires=0)

  @qjit
  def cost(x):
      return circuit2(circuit1(x))
  ```

  ```pycon
  >>> x = jnp.array([0.54, 0.31])
  >>> cost(x)
  array([0.80842369, 0.19157631])
  ```

* Support for returning the variance of Hamiltonians,
  Hermitian matrices, and Tensors via `qml.var` has been added.
  [#124](https://github.com/PennyLaneAI/catalyst/pull/124)

  ```python
  dev = qml.device("lightning.qubit", wires=2)

  @qjit
  @qml.qnode(dev)
  def circuit(x):
      qml.RX(jnp.pi * x[0], wires=0)
      qml.RY(x[1] ** 2, wires=1)
      qml.CNOT(wires=[0, 1])
      qml.RX(x[1] * x[2], wires=0)
      return qml.var(qml.PauliZ(0) @ qml.PauliX(1))
  ```

  ```pycon
  >>> x = jnp.array([0.54, 0.31])
  >>> circuit(x)
  array(0.98851544)
  ```

<h3>Breaking changes</h3>

* The `catalyst.grad` function now supports using the differentiation
  method defined on the QNode (via the `diff_method` argument) rather than
  applying a global differentiation method.
  [#163](https://github.com/PennyLaneAI/catalyst/pull/163)

  As part of this change, the `method` argument now accepts
  the following options:

  - `method="auto"`:  Quantum components of the hybrid function are
    differentiated according to the corresponding QNode `diff_method`, while
    the classical computation is differentiated using traditional auto-diff.

    With this strategy, Catalyst only currently supports QNodes with
    `diff_method="param-shift" and `diff_method="adjoint"`.

  - `method="fd"`: First-order finite-differences for the entire hybrid function.
    The `diff_method` argument for each QNode is ignored.

  This is an intermediate step towards differentiating functions that
  internally call multiple QNodes, and towards supporting differentiation of
  classical postprocessing.

<h3>Improvements</h3>

* Catalyst has been upgraded to work with JAX v0.4.13.
  [#143](https://github.com/PennyLaneAI/catalyst/pull/143)
  [#185](https://github.com/PennyLaneAI/catalyst/pull/185)

* Add a Backprop operation for using autodifferentiation (AD) at the LLVM
  level with Enzyme AD. The Backprop operations has a bufferization pattern
  and a lowering to LLVM.
  [#107](https://github.com/PennyLaneAI/catalyst/pull/107)
  [#116](https://github.com/PennyLaneAI/catalyst/pull/116)

* Error handling has been improved. The runtime now throws more descriptive
  and unified expressions for runtime errors and assertions.
  [#92](https://github.com/PennyLaneAI/catalyst/pull/92)

* In preparation for easier debugging, the compiler has been refactored to
  allow easy prototyping of new compilation pipelines.
  [#38](https://github.com/PennyLaneAI/catalyst/pull/38)

  In the future, this will allow the ability to generate MLIR or LLVM-IR by
  loading input from a string or file, rather than generating it from Python.

  As part of this refactor, the following changes were made:

  - Passes are now classes. This allows developers/users looking to change
    flags to inherit from these passes and change the flags.

  - Passes are now passed as arguments to the compiler. Custom passes can just
    be passed to the compiler as an argument, as long as they implement a run
    method which takes an input and the output of this method can be fed to
    the next pass.

* Improved Python compatibility by providing a stable signature for user
  generated functions.
  [#106](https://github.com/PennyLaneAI/catalyst/pull/106)

* Handle C++ exceptions without unwinding the whole stack.
  [#99](https://github.com/PennyLaneAI/catalyst/pull/99)

* Reduce the number of classical invocations by counting the number of gate parameters in
  the `argmap` function.
  [#136](https://github.com/PennyLaneAI/catalyst/pull/136)

  Prior to this, the computation of hybrid gradients executed all of the classical code
  being differentiated in a `pcount` function that solely counted the number of gate
  parameters in the quantum circuit. This was so `argmap` and other downstream
  functions could allocate memrefs large enough to store all gate parameters.

  Now, instead of counting the number of parameters separately, a dynamically-resizable
  array is used in the `argmap` function directly to store the gate parameters. This
  removes one invocation of all of the classical code being differentiated.

* Use Tablegen to define MLIR passes instead of C++ to reduce overhead of adding new passes.
  [#157](https://github.com/PennyLaneAI/catalyst/pull/157)

* Perform constant folding on wire indices for `quantum.insert` and `quantum.extract` ops,
  used when writing (resp. reading) qubits to (resp. from) quantum registers.
  [#161](https://github.com/PennyLaneAI/catalyst/pull/161)

* Represent known named observables as members of an MLIR Enum rather than a raw integer.
  This improves IR readability.
  [#165](https://github.com/PennyLaneAI/catalyst/pull/165)

<h3>Bug fixes</h3>

* Fix a bug in the mapping from logical to concrete qubits for mid-circuit measurements.
  [#80](https://github.com/PennyLaneAI/catalyst/pull/80)

* Fix a bug in the way gradient result type is inferred.
  [#84](https://github.com/PennyLaneAI/catalyst/pull/84)

* Fix a memory regression and reduce memory footprint by removing unnecessary
  temporary buffers.
  [#100](https://github.com/PennyLaneAI/catalyst/pull/100)

* Provide a new abstraction to the `QuantumDevice` interface in the runtime
  called `DataView`. C++ implementations of the interface can iterate
  through and directly store results into the `DataView` independent of the
  underlying memory layout. This can eliminate redundant buffer copies at the
  interface boundaries, which has been applied to existing devices.
  [#109](https://github.com/PennyLaneAI/catalyst/pull/109)

* Reduce memory utilization by transferring ownership of buffers from the
  runtime to Python instead of copying them. This includes adding a compiler
  pass that copies global buffers into the heap as global buffers cannot be
  transferred to Python.
  [#112](https://github.com/PennyLaneAI/catalyst/pull/112)

* Temporary fix of use-after-free and dependency of uninitialized memory.
  [#121](https://github.com/PennyLaneAI/catalyst/pull/121)

* Fix file renaming within pass pipelines.
  [#126](https://github.com/PennyLaneAI/catalyst/pull/126)

* Fix the issue with the `do_queue` deprecation warnings in PennyLane.
  [#146](https://github.com/PennyLaneAI/catalyst/pull/146)

* Fix the issue with gradients failing to work with hybrid functions that
  contain constant `jnp.array` objects. This will enable PennyLane operators
  that have data in the form of a `jnp.array`, such as a Hamiltonian, to be
  included in a qjit-compiled function.
  [#152](https://github.com/PennyLaneAI/catalyst/pull/152)

  An example of a newly supported workflow:

  ```python
  coeffs = jnp.array([0.1, 0.2])
  terms = [qml.PauliX(0) @ qml.PauliZ(1), qml.PauliZ(0)]
  H = qml.Hamiltonian(coeffs, terms)

  @qjit
  @qml.qnode(qml.device("lightning.qubit", wires=2))
  def circuit(x):
    qml.RX(x[0], wires=0)
    qml.RY(x[1], wires=0)
    qml.CNOT(wires=[0, 1])
    return qml.expval(H)

  params = jnp.array([0.3, 0.4])
  jax.grad(circuit)(params)
  ```

<h3>Contributors</h3>

This release contains contributions from (in alphabetical order):

Ali Asadi,
David Ittah,
Erick Ochoa Lopez,
Jacob Mai Peng,
Romain Moyard,
Sergei Mironov.

# Release 0.1.2

<h3>New features</h3>

* Add an option to print verbose messages explaining the compilation process.
  [#68](https://github.com/PennyLaneAI/catalyst/pull/68)

* Allow ``catalyst.grad`` to be used on any traceable function (within a qjit context).
  This means the operation is no longer restricted to acting on ``qml.qnode``s only.
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
