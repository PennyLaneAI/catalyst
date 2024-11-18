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
