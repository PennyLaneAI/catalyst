# Release 0.7.0

<h3>New features</h3>

* Add support for accelerating classical processing via JAX with `catalyst.accelerate`.
  [(#805)](https://github.com/PennyLaneAI/catalyst/pull/805)

  Classical code that can be just-in-time compiled with JAX can now be seamlessly executed
  on GPUs or other accelerators with `catalyst.accelerate`, right inside of QJIT-compiled functions.

  ```python
  @accelerate(dev=jax.devices("gpu")[0])
  def classical_fn(x):
      return jnp.sin(x) ** 2

  @qjit
  def hybrid_fn(x):
      y = classical_fn(jnp.sqrt(x)) # will be executed on a GPU
      return jnp.cos(y)
  ```

  Available devices can be retrieved via
  `jax.devices()`. If not provided, the default value of
  `jax.devices()[0]` as determined by JAX will be used.

* Catalyst callback functions, such as `pure_callback`, `debug.callback`, and `debug.print`, now
  all support auto-differentiation.
  [(#706)](https://github.com/PennyLaneAI/catalyst/pull/706)
  [(#782)](https://github.com/PennyLaneAI/catalyst/pull/782)
  [(#822)](https://github.com/PennyLaneAI/catalyst/pull/822)
  [(#834)](https://github.com/PennyLaneAI/catalyst/pull/834)
  [(#882)](https://github.com/PennyLaneAI/catalyst/pull/882)
  [(#907)](https://github.com/PennyLaneAI/catalyst/pull/907)

  - When using callbacks that do not return any values, such as `catalyst.debug.callback` and
    `catalyst.debug.print`, these functions are marked as 'inactive' and do not contribute to or
    affect the derivative of the function:

    ```python
    import logging

    log = logging.getLogger(__name__)
    log.setLevel(logging.INFO)

    @qml.qjit
    @catalyst.grad
    def f(x):
        y = jnp.cos(x)
        catalyst.debug.print("Debug print: y = {0:.4f}", y)
        catalyst.debug.callback(lambda _: log.info("Value of y = %s", _))(y)
        return y ** 2
    ```

    ```pycon
    >>> f(0.54)
    INFO:__main__:Value of y = 0.8577086813638242
    Debug print: y = 0.8577
    array(-0.88195781)
    ```

  - Callbacks that *do* return values and may affect the qjit-compiled functions
    computation, such as `pure_callback`, may have custom derivatives manually
    registered with the Catalyst compiler in order to support differentiation.

    This can be done via the `pure_callback.fwd` and `pure_callback.bwd` methods, to specify how the
    forwards and backwards pass (the vector-Jacobian product) of the callback should be computed:

    ```python
    @catalyst.pure_callback
    def callback_fn(x) -> float:
        return np.sin(x[0]) * x[1]

    @callback_fn.fwd
    def callback_fn_fwd(x):
        # returns the evaluated function as well as residual
        # values that may be useful for the backwards pass
        return callback_fn(x), x

    @callback_fn.bwd
    def callback_fn_vjp(res, dy):
        # Accepts residuals from the forward pass, as well
        # as (one or more) cotangent vectors dy, and returns
        # a tuple of VJPs corresponding to each input parameter.

        def vjp(x, dy) -> (jax.ShapeDtypeStruct((2,), jnp.float64),):
            return (np.array([np.cos(x[0]) * dy * x[1], np.sin(x[0]) * dy]),)

        # The VJP function can also be a pure callback
        return catalyst.pure_callback(vjp)(res, dy)

    @qml.qjit
    @catalyst.grad
    def f(x):
        y = jnp.array([jnp.cos(x[0]), x[1]])
        return jnp.sin(callback_fn(y))
    ```

    ```pycon
    >>> x = jnp.array([0.1, 0.2])
    >>> f(x)
    array([-0.01071923,  0.82698717])
    ```

* Catalyst now supports the 'dynamic one shot' method for simulating circuits with mid-circuit
  measurements, which compared to other methods, may be advantageous for circuits with many
  mid-circuit measurements executed for few shots.
  [(#5617)](https://github.com/PennyLaneAI/pennylane/pull/5617)
  [(#798)](https://github.com/PennyLaneAI/catalyst/pull/798)

  The dynamic one shot method evaluates dynamic circuits by executing them one shot at a time via
  `catalyst.vmap`, sampling a dynamic execution path for each shot. This method only works for a
  QNode executing with finite shots, and it requires the device to support mid-circuit measurements
  natively.

  This new mode can be specified by using the `mcm_method` argument of the QNode:

  ```python
  dev = qml.device("lightning.qubit", wires=5, shots=20)

  @qml.qjit(autograph=True)
  @qml.qnode(dev, mcm_method="one-shot")
  def circuit(x):

      for i in range(10):
          qml.RX(x, 0)
          m = catalyst.measure(0)

          if m:
              qml.RY(x ** 2, 1)

          x = jnp.sin(x)

      return qml.expval(qml.Z(1))
  ```

  Catalyst's existing method for simulating mid-circuit measurements remains
  available via `mcm_method="single-branch-statistics"`.

  When using `mcm_method="one-shot"`, the `postselect_mode` keyword argument can also
  be used to specify whether the returned result should include `shots`-number of
  postselected measurements (`"fill-shots"`), or whether results should
  include all results, including invalid postselections (`"hw_like"`):

  ```python
  @qml.qjit
  @qml.qnode(dev, mcm_method="one-shot", postselect_mode="hw-like")
  def func(x):
      qml.RX(x, wires=0)
      m_0 = catalyst.measure(0, postselect=1)
      return qml.sample(wires=0)
  ```

  ```pycon
  >>> res = func(0.9)
  >>> res
  array([-2147483648, -2147483648,           1, -2147483648, -2147483648,
         -2147483648, -2147483648,           1, -2147483648, -2147483648,
         -2147483648, -2147483648,           1, -2147483648, -2147483648,
         -2147483648, -2147483648, -2147483648, -2147483648, -2147483648])
  >>> jnp.delete(res, jnp.where(res == np.iinfo(np.int32).min)[0])
  Array([1, 1, 1], dtype=int64)
  ```

  Note that invalid shots will not be discarded, but will be replaced by `np.iinfo(np.int32).min`.
  They will not be used for processing final results (like expectation values), but
  they will appear in the output of QNodes that return samples directly.

  For more details, see the [dynamic quantum circuit documentation](https://docs.pennylane.ai/en/latest/introduction/dynamic_quantum_circuits.html).

* Catalyst now has support for returning `qml.sample(m)` where `m` is the result of a mid-circuit
  measurement.
  [(#731)](https://github.com/PennyLaneAI/catalyst/pull/731)

  When used with `mcm_method="one-shot"`, this will return an array with one measurement
  result for each shot:

  ```python
  dev = qml.device("lightning.qubit", wires=2, shots=10)

  @qml.qjit
  @qml.qnode(dev, mcm_method="one-shot")
  def func(x):
      qml.RX(x, wires=0)
      m = catalyst.measure(0)
      qml.RX(x ** 2, wires=0)
      return qml.sample(m), qml.expval(qml.PauliZ(0))
  ```

  ```pycon
  >>> func(0.9)
  (array([0, 1, 0, 0, 0, 0, 1, 0, 0, 0]), array(0.4))
  ```

  In `mcm_method="single-branch-statistics"` mode, it will be equivalent to
  returning `m` directly from the quantum function --- that is, it will return
  a single boolean corresponding to the measurement in the branch selected:

  ```python
  @qml.qjit
  @qml.qnode(dev, mcm_method="single-branch-statistics")
  def func(x):
      qml.RX(x, wires=0)
      m = catalyst.measure(0)
      qml.RX(x ** 2, wires=0)
      return qml.sample(m), qml.expval(qml.PauliZ(0))
  ```

  ```pycon
  >>> func(0.9)
  (array(False), array(0.8))
  ```

* A new function, `catalyst.value_and_grad`, returns both the result of a function and
  its gradient with a single forward and backwards pass.
  [(#804)](https://github.com/PennyLaneAI/catalyst/pull/804)
  [(#859)](https://github.com/PennyLaneAI/catalyst/pull/859)

  This can be more efficient, and reduce overall quantum executions, compared to separately
  executing the function and then computing its gradient.

  For example:

  ```py
  dev = qml.device("lightning.qubit", wires=3)

  @qml.qnode(dev)
  def circuit(x):
      qml.RX(x, wires=0)
      qml.CNOT(wires=[0, 1])
      qml.RX(x, wires=2)
      return qml.probs()

  @qml.qjit
  @catalyst.value_and_grad
  def cost(x):
      return jnp.sum(jnp.cos(circuit(x)))
  ```

  ```pycon
  >>> cost(0.543)
  (array(7.64695856), array(0.33413963))
  ```

* Autograph now supports single index JAX array assignments
  [(#717)](https://github.com/PennyLaneAI/catalyst/pull/717)

  When using Autograph, syntax of the form `x[i] = y` where `i` is a single integer
  will now be automatically converted to the JAX equivalent of `x = x.at(i).set(y)`:

  ```python
  @qml.qjit(autograph=True)
  def f(array):
      result = jnp.ones(array.shape, dtype=array.dtype)

      for i, x in enumerate(array):
          result[i] = result[i] + x * 3

      return result
  ```

  ```pycon
  >>> f(jnp.array([-0.1, 0.12, 0.43, 0.54]))
  array([0.7 , 1.36, 2.29, 2.62])
  ```

* Catalyst now supports dynamically-shaped arrays in control-flow primitives. Arrays with dynamic
  shapes can now be used with `for_loop`, `while_loop`, and `cond` primitives.
  [(#775)](https://github.com/PennyLaneAI/catalyst/pull/775)
  [(#777)](https://github.com/PennyLaneAI/catalyst/pull/777)
  [(#830)](https://github.com/PennyLaneAI/catalyst/pull/830)

  ``` python
  @qjit
  def f(shape):
      a = jnp.ones([shape], dtype=float)

      @for_loop(0, 10, 2)
      def loop(i, a):
          return a + i

      return loop(a)
  ```
  ``` pycon
  >>> f(3)
  array([21., 21., 21.])
  ```

* Support has been added for disabling Autograph for specific functions.
  [(#705)](https://github.com/PennyLaneAI/catalyst/pull/705)
  [(#710)](https://github.com/PennyLaneAI/catalyst/pull/710)

  The decorator `catalyst.disable_autograph` allows one to disable Autograph
  from auto-converting specific external functions when called within a qjit-compiled
  function with `autograph=True`:

  ```python
  def approximate_e(n):
      num = 1.
      fac = 1.
      for i in range(1, n + 1):
          fac *= i
          num += 1. / fac
      return num

  @qml.qjit(autograph=True)
  def g(x: float, N: int):

      for i in range(N):
          x = x + catalyst.disable_autograph(approximate_e)(10) / x ** i

      return x
  ```

  ```pycon
  >>> g(0.1, 10)
  array(4.02997319)
  ```

  Note that for Autograph to be disabled, the decorated function must be
  defined **outside** the qjit-compiled function. If it is defined within
  the qjit-compiled function, it will continue to be converted with Autograph.

  In addition, Autograph can also be disabled for all externally defined functions
  within a qjit-compiled function via the context manager syntax:

  ```python
  @qml.qjit(autograph=True)
  def g(x: float, N: int):

      for i in range(N):
          with catalyst.disable_autograph:
            x = x + approximate_e(10) / x ** i

      return x
  ```

* Support for including a list of (sub)modules to be allowlisted for autograph conversion.
  [(#725)](https://github.com/PennyLaneAI/catalyst/pull/725)

  Although library code is not meant to be targeted by Autograph conversion,
  it sometimes make sense to enable it for specific submodules that might
  benefit from such conversion:

  ```py
  @qjit(autograph=True, autograph_include=["excluded_module.submodule"])
  def f(x):
    return excluded_module.submodule.func(x)

  ```

  For example, this might be useful if importing functionality from PennyLane (such as
  a transform or decomposition), and would like to have Autograph capture and convert
  associated control flow.

* Controlled operations that do not have a matrix representation defined are now supported via
  applying PennyLane's decomposition.
  [(#831)](https://github.com/PennyLaneAI/catalyst/pull/831)

  ``` python
  @qjit
  @qml.qnode(qml.device("lightning.qubit", wires=2))
  def circuit():
      qml.Hadamard(0)
      qml.ctrl(qml.TrotterProduct(H, time=2.4, order=2), control=[1])
      return qml.state()
  ```

* Catalyst has now officially support on Linux aarch64, with pre-built binaries
  available on PyPI; simply `pip install pennylane-catalyst` on Linux aarch64 systems.
  [(#767)](https://github.com/PennyLaneAI/catalyst/pull/767)

<h3>Improvements</h3>

* Validation is now performed for observables and operations to ensure that provided circuits
  are compatible with the devices for execution.
  [(#626)](https://github.com/PennyLaneAI/catalyst/pull/626)
  [(#783)](https://github.com/PennyLaneAI/catalyst/pull/783)

  ```python
  dev = qml.device("lightning.qubit", wires=2, shots=10000)

  @qjit
  @qml.qnode(dev)
  def circuit(x):
      qml.Hadamard(wires=0)
      qml.CRX(x, wires=[0, 1])
      return qml.var(qml.PauliZ(1))
  ```

  ```pycon
  >>> circuit(0.43)
  DifferentiableCompileError: Variance returns are forbidden in gradients
  ```

* Catalyst's adjoint and ctrl methods are now fully compatible with the PennyLane equivalent when
  applied to a single Operator. This should lead to improved compatibility with PennyLane library
  code, as well when reusing quantum functions with both Catalyst and PennyLane.
  [(#768)](https://github.com/PennyLaneAI/catalyst/pull/768)
  [(#771)](https://github.com/PennyLaneAI/catalyst/pull/771)
  [(#802)](https://github.com/PennyLaneAI/catalyst/pull/802)

* Controlled operations defined via specialized classes (like `Toffoli` or `ControlledQubitUnitary`)
  are now implemented as controlled versions of their base operation if the device supports it.
  In particular, `MultiControlledX` is no longer executed as a `QubitUnitary` with Lightning.
  [(#792)](https://github.com/PennyLaneAI/catalyst/pull/792)

* The Catalyst frontend now supports Python logging through PennyLane's `qml.logging` module.
  For more details, please see the [logging documentation](https://docs.pennylane.ai/en/stable/introduction/logging.html).
  [(#660)](https://github.com/PennyLaneAI/catalyst/pull/660)

* Catalyst now performs a stricter validation of the wire requirements for devices. In particular,
  only integer, continuous wire labels starting at 0 are allowed.
  [(#784)](https://github.com/PennyLaneAI/catalyst/pull/784)

* Catalyst no longer disallows quantum circuits with 0 qubits.
  [(#784)](https://github.com/PennyLaneAI/catalyst/pull/784)

* Added support for `IsingZZ` as a native gate in Catalyst. Previously, the IsingZZ gate would be
  decomposed into a CNOT and RZ gates, even if a device supported it.
  [(#730)](https://github.com/PennyLaneAI/catalyst/pull/730)

* All decorators in Catalyst, including `vmap`, `qjit`, `mitigate_with_zne`,
  as well as gradient decorators `grad`, `jacobian`, `jvp`, and `vjp`, can now be used
  both with and without keyword arguments as a decorator without the need for
  `functools.partial`:
  [(#758)](https://github.com/PennyLaneAI/catalyst/pull/758)
  [(#761)](https://github.com/PennyLaneAI/catalyst/pull/761)
  [(#762)](https://github.com/PennyLaneAI/catalyst/pull/762)
  [(#763)](https://github.com/PennyLaneAI/catalyst/pull/763)

  ```python
  @qjit
  @grad(method="fd")
  def fn1(x):
      return x ** 2

  @qjit(autograph=True)
  @grad
  def fn2(x):
      return jnp.sin(x)
  ```

  ```pycon
  >>> fn1(0.43)
  array(0.8600001)
  >>> fn2(0.12)
  array(0.99280864)
  ```

* The built-in instrumentation with `detailed` output will no longer report the cumulative time for
  MLIR pipelines, since the cumulative time was being reported as just another step alongside
  individual timings for each pipeline.
  [(#772)](https://github.com/PennyLaneAI/catalyst/pull/772)

* Raise a better error message when no shots are specified and `qml.sample` or `qml.counts` is used.
  [(#786)](https://github.com/PennyLaneAI/catalyst/pull/786)

* The finite difference method for differentiation is now always allowed, even on functions with
  mid-circuit measurements, callbacks without custom derivates, or other operations that cannot
  be differentiated via traditional autodiff.
  [(#789)](https://github.com/PennyLaneAI/catalyst/pull/789)

* A `non_commuting_observables` flag has been added to the device TOML schema, indicating whether or
  not the device supports measuring non-commuting observables. If `false`, non-commuting
  measurements will be split into multiple executions.
  [(#821)](https://github.com/PennyLaneAI/catalyst/pull/821)

* The underlying PennyLane `Operation` objects for `cond`, `for_loop`, and `while_loop` can now be
  accessed directly via `body_function.operation`.
  [(#711)](https://github.com/PennyLaneAI/catalyst/pull/711)

  This can be beneficial when, among other things,
  writing transforms without using the queuing mechanism:

  ```python
  @qml.transform
  def my_quantum_transform(tape):
      ops = tape.operations.copy()

      @for_loop(0, 4, 1)
      def f(i, sum):
          qml.Hadamard(0)
          return sum+1

      res = f(0)
      ops.append(f.operation)   # This is now supported!

      def post_processing_fn(results):
          return results
      modified_tape = qml.tape.QuantumTape(ops, tape.measurements)
      print(res)
      print(modified_tape.operations)
      return [modified_tape], post_processing_fn

  @qml.qjit
  @my_quantum_transform
  @qml.qnode(qml.device("lightning.qubit", wires=2))
  def main():
      qml.Hadamard(0)
      return qml.probs()
  ```

  ```pycon
  >>> main()
  Traced<ShapedArray(int64[], weak_type=True)>with<DynamicJaxprTrace(level=2/1)>
  [Hadamard(wires=[0]), ForLoop(tapes=[[Hadamard(wires=[0])]])]
  (array([0.5, 0. , 0.5, 0. ]),)
  ```

<h3>Breaking changes</h3>

* Binary distributions for Linux are now based on `manylinux_2_28` instead of `manylinux_2014`.
  As a result, Catalyst will only be compatible on systems with `glibc` versions `2.28` and above
  (e.g., Ubuntu 20.04 and above).
  [(#663)](https://github.com/PennyLaneAI/catalyst/pull/663)

<h3>Bug fixes</h3>

* Functions that have been annotated with return type
  annotations will now correctly compile with `@qjit`.
  [(#751)](https://github.com/PennyLaneAI/catalyst/pull/751)

* An issue in the Lightning backend for the Catalyst runtime has been fixed that would only compute
  approximate probabilities when implementing mid-circuit measurements. As a result, low shot numbers
  would lead to unexpected behaviours or projections on zero probability states.
  Probabilities for mid-circuit measurements are now always computed analytically.
  [(#801)](https://github.com/PennyLaneAI/catalyst/pull/801)

* The Catalyst runtime now raises an error if a qubit is accessed out of bounds from the allocated
  register.
  [(#784)](https://github.com/PennyLaneAI/catalyst/pull/784)

* `jax.scipy.linalg.expm` is now supported within qjit-compiled functions.
  [(#733)](https://github.com/PennyLaneAI/catalyst/pull/733)
  [(#752)](https://github.com/PennyLaneAI/catalyst/pull/752)

  This required correctly linking openblas routines necessary for `jax.scipy.linalg.expm`.
  In this bug fix, four openblas routines were newly linked and are now discoverable by
  `stablehlo.custom_call@<blas_routine>`. They are `blas_dtrsm`, `blas_ztrsm`, `lapack_dgetrf`,
  `lapack_zgetrf`.

* Fixes a bug where QNodes that contained `QubitUnitary` with a complex matrix
  would error during gradient computation.
  [(#778)](https://github.com/PennyLaneAI/catalyst/pull/778)

* Callbacks can now return types which can be flattened and unflattened.
  [(#812)](https://github.com/PennyLaneAI/catalyst/pull/812)

* `catalyst.qjit` and `catalyst.grad` now work correctly on
  functions that have been wrapped with `functools.partial`.
  [(#820)](https://github.com/PennyLaneAI/catalyst/pull/820)

<h3>Internal changes</h3>

* Catalyst uses the `collapse` method of Lightning simulators in `Measure` to select a state vector
  branch and normalize.
  [(#801)](https://github.com/PennyLaneAI/catalyst/pull/801)

* Measurement process primitives for Catalyst's JAXPR representation now have a standardized
  call signature so that `shots` and `shape` can both be provided as keyword arguments.
  [(#790)](https://github.com/PennyLaneAI/catalyst/pull/790)

* The `QCtrl` class in Catalyst has been renamed to `HybridCtrl`, indicating its capability
  to contain a nested scope of both quantum and classical operations.
  Using `ctrl` on a single operation will now directly dispatch to the equivalent PennyLane class.
  [(#771)](https://github.com/PennyLaneAI/catalyst/pull/771)

* The `Adjoint` class in Catalyst has been renamed to `HybridAdjoint`, indicating its capability
  to contain a nested scope of both quantum and classical operations.
  Using `adjoint` on a single operation will now directly dispatch to the equivalent PennyLane class.
  [(#768)](https://github.com/PennyLaneAI/catalyst/pull/768)
  [(#802)](https://github.com/PennyLaneAI/catalyst/pull/802)

* Add support to use a locally cloned PennyLane Lightning repository with the runtime.
  [(#732)](https://github.com/PennyLaneAI/catalyst/pull/732)

* The `qjit_device.py` and `preprocessing.py` modules have been refactored into the sub-package
  `catalyst.device`.
  [(#721)](https://github.com/PennyLaneAI/catalyst/pull/721)

* The `ag_autograph.py` and `autograph.py` modules have been refactored into the sub-package
  `catalyst.autograph`.
  [(#722)](https://github.com/PennyLaneAI/catalyst/pull/722)

* Callback refactoring. This refactoring creates the classes `FlatCallable`
  and `MemrefCallable`.
  [(#742)](https://github.com/PennyLaneAI/catalyst/pull/742)

  The `FlatCallable` class is a `Callable` that is
  initialized by providing some parameters and kwparameters that match the
  the expected shapes that will be received at the callsite. Instead of taking
  shaped `*args` and `**kwargs`, it receives flattened arguments. The flattened
  arguments are unflattened with the shapes with which the function was
  initialized. The `FlatCallable` return values will allways be flattened
  before returning to the caller.

  The `MemrefCallable` is a subclass of `FlatCallable`. It takes a result type
  parameter during initialization that corresponds to the expected return type.
  This class is expected to be called only from the Catalyst runtime. It
  expects all arguments to be `void*` to memrefs. These `void*` are casted
  to MemrefStructDescriptors using ctypes, numpy arrays, and finally jax
  arrays. These flat jax arrays are then sent to the `FlatCallable`.
  `MemrefCallable` is again expected to be called only from within the Catalyst
  runtime. And the return values match those expected by Catalyst runtime.

  This separation allows for a better separation of concerns, provides a nicer
  interface and allows for multiple `MemrefCallable` to be defined for a single
  callback, which is necessary for custom gradient of `pure_callbacks`.

* A new `catalyst::gradient::GradientOpInterface` is available when querying the gradient method in
  the mlir c++ api.
  [(#800)](https://github.com/PennyLaneAI/catalyst/pull/800)

  `catalyst::gradient::GradOp`, `ValueAndGradOp`, `JVPOp`, and `VJPOp` now inherits traits in this
  new `GradientOpInterface`. The supported attributes are now `getMethod()`, `getCallee()`,
  `getDiffArgIndices()`, `getDiffArgIndicesAttr()`, `getFiniteDiffParam()`, and
  `getFiniteDiffParamAttr()`.

  - There are operations that could potentially be used as `GradOp`, `ValueAndGradOp`, `JVPOp` or
    `VJPOp`. When trying to get the gradient method, instead of doing
    ```C++
          auto gradOp = dyn_cast<GradOp>(op);
          auto jvpOp = dyn_cast<JVPOp>(op);
          auto vjpOp = dyn_cast<VJPOp>(op);

          llvm::StringRef MethodName;
          if (gradOp)
              MethodName = gradOp.getMethod();
          else if (jvpOp)
              MethodName = jvpOp.getMethod();
          else if (vjpOp)
              MethodName = vjpOp.getMethod();
    ```
    to identify which op it actually is and protect against segfaults (calling
    `nullptr.getMethod()`), in the new interface we just do
    ```C++
          auto gradOpInterface = cast<GradientOpInterface>(op);
          llvm::StringRef MethodName = gradOpInterface.getMethod();
    ```

  - Another advantage is that any concrete gradient operation object can behave like a
    `GradientOpInterface`:
    ```C++
    GradOp op; // or ValueAndGradOp op, ...
    auto foo = [](GradientOpInterface op){
      llvm::errs() << op.getCallee();
    };
    foo(op);  // this works!
    ```

  - Finally, concrete op specific methods can still be called by "reinterpret"-casting the interface
    back to a concrete op (provided the concrete op type is correct):
    ```C++
    auto foo = [](GradientOpInterface op){
      size_t numGradients = cast<ValueAndGradOp>(&op)->getGradients().size();
    };
    ValueAndGradOp op;
    foo(op);  // this works!
    ```

<h3>Contributors</h3>

This release contains contributions from (in alphabetical order):

Ali Asadi,
Lillian M.A. Frederiksen,
David Ittah,
Christina Lee,
Erick Ochoa,
Haochen Paul Wang,
Lee James O'Riordan,
Mehrdad Malekmohammadi,
Vincent Michaud-Rioux,
Mudit Pandey,
Raul Torres,
Sergei Mironov,
Tzung-Han Juang.
