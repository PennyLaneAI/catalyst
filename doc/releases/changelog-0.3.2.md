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

