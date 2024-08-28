# Release 0.8.0

<h3>New features</h3>

* JAX-compatible functions that run on classical accelerators, such as GPUs, via `catalyst.accelerate` now support autodifferentiation.
  [(#920)](https://github.com/PennyLaneAI/catalyst/pull/920)

  For example,

  ```python
  from catalyst import qjit, grad

  @qjit
  @grad
  def f(x):
      expm = catalyst.accelerate(jax.scipy.linalg.expm)
      return jnp.sum(expm(jnp.sin(x)) ** 2)
  ```

  ```pycon
  >>> x = jnp.array([[0.1, 0.2], [0.3, 0.4]])
  >>> f(x)
  Array([[2.80120452, 1.67518663],
         [1.61605839, 4.42856163]], dtype=float64)
  ```

* Assertions can now be raised at runtime via the `catalyst.debug_assert` function.
  [(#925)](https://github.com/PennyLaneAI/catalyst/pull/925)

  Python-based exceptions (via `raise`) and assertions (via `assert`)
  will always be evaluated at program capture time, before certain runtime information
  may be available.

  Use `debug_assert` to instead raise assertions at runtime, including
  assertions that depend on values of dynamic variables.

  For example,

  ```python
  from catalyst import debug_assert

  @qjit
  def f(x):
      debug_assert(x < 5, "x was greater than 5")
      return x * 8
  ```

  ```pycon
  >>> f(4)
  Array(32, dtype=int64)
  >>> f(6)
  RuntimeError: x was greater than 5
  ```

  Assertions can be disabled globally for a qjit-compiled function
  via the ``disable_assertions`` keyword argument:

  ```python
  @qjit(disable_assertions=True)
  def g(x):
      debug_assert(x < 5, "x was greater than 5")
      return x * 8
  ```

  ```pycon
  >>> g(6)
  Array(48, dtype=int64)
  ```

* Mid-circuit measurement results when using `lightning.qubit` and `lightning.kokkos`
  can now be seeded via the new `seed` argument of the `qjit` decorator.
  [(#936)](https://github.com/PennyLaneAI/catalyst/pull/936)

  The seed argument accepts an unsigned 32-bit integer, which is used to initialize the pseudo-random
  state at the beginning of each execution of the compiled function.
  Therefor, different `qjit` objects with the same seed (including repeated calls to the same `qjit`)
  will always return the same sequence of mid-circuit measurement results.

  ```python
  dev = qml.device("lightning.qubit", wires=1)

  @qml.qnode(dev)
  def circuit(x):
      qml.RX(x, wires=0)
      m = measure(0)

      if m:
          qml.Hadamard(0)

      return qml.probs()

  @qjit(seed=37, autograph=True)
  def workflow(x):
      return jnp.stack([circuit(x) for i in range(4)])
  ```

  Repeatedly calling the `workflow` function above will always
  result in the same values:

  ```pycon
  >>> workflow(1.8)
  Array([[1. , 0. ],
       [1. , 0. ],
       [1. , 0. ],
       [0.5, 0.5]], dtype=float64)
  >>> workflow(1.8)
  Array([[1. , 0. ],
       [1. , 0. ],
       [1. , 0. ],
       [0.5, 0.5]], dtype=float64)
  ```

  Note that setting the seed will *not* avoid shot-noise stochasticity in terminal measurement
  statistics such as `sample` or `expval`:

  ```python
  dev = qml.device("lightning.qubit", wires=1, shots=10)

  @qml.qnode(dev)
  def circuit(x):
      qml.RX(x, wires=0)
      m = measure(0)

      if m:
          qml.Hadamard(0)

      return qml.expval(qml.PauliZ(0))

  @qjit(seed=37, autograph=True)
  def workflow(x):
      return jnp.stack([circuit(x) for i in range(4)])
  ```
  ```pycon
  >>> workflow(1.8)
  Array([1. , 1. , 1. , 0.4], dtype=float64)
  >>> workflow(1.8)
  Array([ 1. ,  1. ,  1. , -0.2], dtype=float64)
  ```

* Exponential fitting is now a supported method of zero-noise extrapolation when performing
  error mitigation in Catalyst using `mitigate_with_zne`.
  [(#953)](https://github.com/PennyLaneAI/catalyst/pull/953)

  This new functionality fits the data from noise-scaled circuits with an exponential function,
  and returns the zero-noise value:

  ```py
  from pennylane.transforms import exponential_extrapolate
  from catalyst import mitigate_with_zne

  dev = qml.device("lightning.qubit", wires=2, shots=100000)

  @qml.qnode(dev)
  def circuit(weights):
      qml.StronglyEntanglingLayers(weights, wires=[0, 1])
      return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))

  @qjit
  def workflow(weights, s):
      zne_circuit = mitigate_with_zne(circuit, scale_factors=s, extrapolate=exponential_extrapolate)
      return zne_circuit(weights)
  ```

  ```pycon
  >>> weights = jnp.ones([3, 2, 3])
  >>> scale_factors = jnp.array([1, 2, 3])
  >>> workflow(weights, scale_factors)
  Array(-0.19946598, dtype=float64)
  ```

* A new module is available, `catalyst.passes`, which provides Python decorators
  for enabling and configuring Catalyst MLIR compiler passes.
  [(#911)](https://github.com/PennyLaneAI/catalyst/pull/911)
  [(#1037)](https://github.com/PennyLaneAI/catalyst/pull/1037)

  The first pass available is `catalyst.passes.cancel_inverses`,
  which enables the `-removed-chained-self-inverse` MLIR pass that
  cancels two neighbouring Hadamard gates.

  ```python
  from catalyst.debug import get_compilation_stage
  from catalyst.passes import cancel_inverses

  dev = qml.device("lightning.qubit", wires=1)

  @qml.qnode(dev)
  def circuit(x: float):
      qml.RX(x, wires=0)
      qml.Hadamard(wires=0)
      qml.Hadamard(wires=0)
      return qml.expval(qml.PauliZ(0))

  @qjit(keep_intermediate=True)
  def workflow(x):
      optimized_circuit = cancel_inverses(circuit)
      return circuit(x), optimized_circuit(x)
  ```

* Catalyst now has debug functions `get_compilation_stage` and `replace_ir` to acquire and
  recompile the IR from a given pipeline pass for functions compiled with
  `keep_intermediate=True`.
  [(#981)](https://github.com/PennyLaneAI/catalyst/pull/981)

  For example, consider the following function:

  ```python
  @qjit(keep_intermediate=True)
  def f(x):
      return x**2
  ```

  ```pycon
  >>> f(2.0)
  4.0
  ```

  Here we use `get_compilation_stage` to acquire the IR, and then modify
  `%2 = arith.mulf %in, %in_0 : f64` to turn the square function into a cubic one
  via `replace_ir`:

  ```python
  from catalyst.debug import get_compilation_stage, replace_ir

  old_ir = get_compilation_stage(f, "HLOLoweringPass")
  new_ir = old_ir.replace(
      "%2 = arith.mulf %in, %in_0 : f64\n",
      "%t = arith.mulf %in, %in_0 : f64\n    %2 = arith.mulf %t, %in_0 : f64\n"
  )
  replace_ir(f, "HLOLoweringPass", new_ir)
  ```

  The recompilation starts after the given checkpoint stage:

  ```pycon
  >>> f(2.0)
  8.0
  ```

  Either function can also be used independently of each other. Note that
  `get_compilation_stage` replaces the `print_compilation_stage` function;
  please see the Breaking Changes section for more details.

* Catalyst now supports generating executables from compiled functions for the native host architecture using
  `catalyst.debug.compile_executable`.
  [(#1003)](https://github.com/PennyLaneAI/catalyst/pull/1003)

  ```pycon
  >>> @qjit
  ... def f(x):
  ...     y = x * x
  ...     catalyst.debug.print_memref(y)
  ...     return y
  >>> f(5)
  MemRef: base@ = 0x31ac22580 rank = 0 offset = 0 sizes = [] strides = [] data =
  25
  Array(25, dtype=int64)
  ```

  We can use ``compile_executable`` to compile this function to a binary:

  ```pycon
  >>> from catalyst.debug import compile_executable
  >>> binary = compile_executable(f, 5)
  >>> print(binary)
  /path/to/executable
  ```

  Executing this function from a shell environment:

  ```console
  $ /path/to/executable
  MemRef: base@ = 0x64fc9dd5ffc0 rank = 0 offset = 0 sizes = [] strides = [] data =
  25
  ```

<h3>Improvements</h3>

* Catalyst has been updated to work with JAX v0.4.28 (exact version match required).
  [(#931)](https://github.com/PennyLaneAI/catalyst/pull/931)
  [(#995)](https://github.com/PennyLaneAI/catalyst/pull/995)

* Catalyst now supports keyword arguments for qjit-compiled functions.
  [(#1004)](https://github.com/PennyLaneAI/catalyst/pull/1004)

  ```pycon
  >>> @qjit
  ... @grad
  ... def f(x, y):
  ...     return x * y
  >>> f(3., y=2.)
  Array(2., dtype=float64)
  ```

  Note that the `static_argnums` argument to the `qjit` decorator
  is not supported when passing argument values as keyword arguments.

* Support has been added for the `jax.numpy.argsort`
  function within qjit-compiled functions.
  [(#901)](https://github.com/PennyLaneAI/catalyst/pull/901)

* Autograph now supports in-place array assignments with static slices.
  [(#843)](https://github.com/PennyLaneAI/catalyst/pull/843)

  For example,

  ```python
  @qjit(autograph=True)
  def f(x, y):
      y[1:10:2] = x
      return y
  ```

  ```pycon
  >>> f(jnp.ones(5), jnp.zeros(10))
  Array([0., 1., 0., 1., 0., 1., 0., 1., 0., 1.], dtype=float64)
  ```

* Autograph now works when `qjit` is applied to a function decorated with
  `vmap`, `cond`, `for_loop` or `while_loop`. Previously, stacking the
  autograph-enabled qjit decorator directly on top of other Catalyst
  decorators would lead to errors.
  [(#835)](https://github.com/PennyLaneAI/catalyst/pull/835)
  [(#938)](https://github.com/PennyLaneAI/catalyst/pull/938)
  [(#942)](https://github.com/PennyLaneAI/catalyst/pull/942)

  ```python
  from catalyst import vmap, qjit

  dev = qml.device("lightning.qubit", wires=2)

  @qml.qnode(dev)
  def circuit(x):
      qml.RX(x, wires=0)
      return qml.expval(qml.PauliZ(0))
  ```

  ```pycon
  >>> x = jnp.array([0.1, 0.2, 0.3])
  >>> qjit(vmap(circuit), autograph=True)(x)
  Array([0.99500417, 0.98006658, 0.95533649], dtype=float64)
  ```

* Runtime memory usage, and compilation complexity, has been reduced by eliminating some scalar
  tensors from the IR. This has been done by adding a `linalg-detensorize` pass at the end of the
  HLO lowering pipeline.
  [(#1010)](https://github.com/PennyLaneAI/catalyst/pull/1010)

* Program verification is extended to confirm that the measurements included in QNodes
  are compatible with the specified device and settings.
  [(#945)](https://github.com/PennyLaneAI/catalyst/pull/945)
  [(#962)](https://github.com/PennyLaneAI/catalyst/pull/962)

  ```pycon
  >>> dev = qml.device("lightning.qubit", wires=2, shots=None)
  >>> @qjit
  ... @qml.qnode(dev)
  ... def circuit(params):
  ...     qml.RX(params[0], wires=0)
  ...     qml.RX(params[1], wires=1)
  ...     return {
  ...         "sample": qml.sample(wires=[0, 1]),
  ...         "expval": qml.expval(qml.PauliZ(0))
  ...     }
  >>> circuit([0.1, 0.2])
  CompileError: Sample-based measurements like sample(wires=[0, 1])
  cannot work with shots=None. Please specify a finite number of shots.
  ```

* On devices that support it, initial state preparation routines `qml.StatePrep` and `qml.BasisState`
  are no longer decomposed when using Catalyst, improving compilation and runtime performance.
  [(#955)](https://github.com/PennyLaneAI/catalyst/pull/955)
  [(#1047)](https://github.com/PennyLaneAI/catalyst/pull/1047)
  [(#1062)](https://github.com/PennyLaneAI/catalyst/pull/1062)

* Improved type validation and error messaging has been added to both the `catalyst.jvp`
  and `catalyst.vjp` functions to ensure that the (co)tangent and parameter types are compatible.
  [(#1020)](https://github.com/PennyLaneAI/catalyst/pull/1020)
  [(#1030)](https://github.com/PennyLaneAI/catalyst/pull/1030)
  [(#1031)](https://github.com/PennyLaneAI/catalyst/pull/1031)

  For example, providing an integer tangent for a function with float64 parameters
  will result in an error:

  ```pycon
  >>> f = lambda x: (2 * x, x * x)
  >>> f_jvp = lambda x: catalyst.jvp(f, params=(x,), tangents=(1,))
  >>> qjit(f_jvp)(0.5)
  TypeError: function params and tangents arguments to catalyst.jvp do not match;
  dtypes must be equal. Got function params dtype float64 and so expected tangent
  dtype float64, but got tangent dtype int64 instead.
  ```

  Ensuring that the types match will resolve the error:

  ```pycon
  >>> f_jvp = lambda x: catalyst.jvp(f, params=(x,), tangents=(1.0,))
  >>> qjit(f_jvp)(0.5)
  ((Array(1., dtype=float64), Array(0.25, dtype=float64)),
   (Array(2., dtype=float64), Array(1., dtype=float64)))
  ```

* Add a script for setting up a Frontend-Only Development Environment that does not require
  compilation, as it uses the TestPyPI wheel shared libraries.
  [(#1022)](https://github.com/PennyLaneAI/catalyst/pull/1022)

<h3>Breaking changes</h3>

* The `argnum` keyword argument in the `grad`, `jacobian`, `value_and_grad`,
  `vjp`, and `jvp` functions has been renamed to `argnums` to better match JAX.
  [(#1036)](https://github.com/PennyLaneAI/catalyst/pull/1036)

* Return values of qjit-compiled functions that were previously `numpy.ndarray` are now of type
  `jax.Array` instead. This should have minimal impact, but code that depends on the output of
  qjit-compiled function being NumPy arrays will need to be updated.
  [(#895)](https://github.com/PennyLaneAI/catalyst/pull/895)

* The `print_compilation_stage` function has been renamed `get_compilation_stage`.
  It no longer prints the IR to the standard output, instead it simply returns
  the IR as a string.
  [(#981)](https://github.com/PennyLaneAI/catalyst/pull/981)

  ```pycon
  >>> @qjit(keep_intermediate=True)
  ... def func(x: float):
  ...     return x
  >>> print(get_compilation_stage(func, "HLOLoweringPass"))
  module @func {
    func.func public @jit_func(%arg0: tensor<f64>)
    -> tensor<f64> attributes {llvm.emit_c_interface} {
      return %arg0 : tensor<f64>
    }
    func.func @setup() {
      quantum.init
      return
    }
    func.func @teardown() {
      quantum.finalize
      return
    }
  }
  ```

* Support for TOML files in Schema 1 has been disabled.
  [(#960)](https://github.com/PennyLaneAI/catalyst/pull/960)

* The `mitigate_with_zne` function no longer accepts a `degree` parameter for polynomial fitting
  and instead accepts a callable to perform extrapolation. Any qjit-compatible extrapolation
  function is valid. Keyword arguments can be passed to this function using the
  `extrapolate_kwargs` keyword argument in `mitigate_with_zne`.
  [(#806)](https://github.com/PennyLaneAI/catalyst/pull/806)

* The QuantumDevice API has now added the functions `SetState` and `SetBasisState`
  for simulators that may benefit from instructions that directly set the state.
  Implementing these methods is optional, and device support can be indicated via
  the `initial_state_prep` flag in the TOML configuration file.
  [(#955)](https://github.com/PennyLaneAI/catalyst/pull/955)

<h3>Bug fixes</h3>

* Fix a bug where LegacyDevice number of shots is not correctly extracted when using the legacyDeviceFacade.
  [(#1035)](https://github.com/PennyLaneAI/catalyst/pull/1035)

* Catalyst no longer generates a `QubitUnitary` operation during decomposition if a device doesn't
  support it. Instead, the operation that would lead to a `QubitUnitary` is either decomposed or
  raises an error.
  [(#1002)](https://github.com/PennyLaneAI/catalyst/pull/1002)

* Catalyst now preserves output PyTrees in QNodes executed with `mcm_method="one-shot"`.
  [(#957)](https://github.com/PennyLaneAI/catalyst/pull/957)

  For example:

  ```python
  dev = qml.device("lightning.qubit", wires=1, shots=20)
  @qml.qjit
  @qml.qnode(dev, mcm_method="one-shot")
  def func(x):
      qml.RX(x, wires=0)
      m_0 = catalyst.measure(0, postselect=1)
      return {"hi": qml.expval(qml.Z(0))}
  ```

  ```pycon
  >>> func(0.9)
  {'hi': Array(-1., dtype=float64)}
  ```

* Fixes a bug where scatter did not work correctly with list indices.
  [(#982)](https://github.com/PennyLaneAI/catalyst/pull/982)

  ```python
  A = jnp.ones([3, 3]) * 2

  def update(A):
      A = A.at[[0, 1], :].set(jnp.ones([2, 3]), indices_are_sorted=True, unique_indices=True)
      return A
  ```

  ```pycon
  >>> update
  [[1. 1. 1.]
   [1. 1. 1.]
   [2. 2. 2.]]
  ```

* Static arguments can now be passed through a QNode when specified
  with the `static_argnums` keyword argument.
  [(#932)](https://github.com/PennyLaneAI/catalyst/pull/932)

  ```python
  dev = qml.device("lightning.qubit", wires=1)

  @qjit(static_argnums=(1,))
  @qml.qnode(dev)
  def circuit(x, c):
      print("Inside QNode:", c)
      qml.RY(c, 0)
      qml.RX(x, 0)
      return qml.expval(qml.PauliZ(0))
  ```

  When executing the qjit-compiled function above, `c` will
  be a static variable with value known at compile time:

  ```pycon
  >>> circuit(0.5, 0.5)
  "Inside QNode: 0.5"
  Array(0.77015115, dtype=float64)
  ```

  Changing the value of `c` will result in re-compilation:

  ```pycon
  >>> circuit(0.5, 0.8)
  "Inside QNode: 0.8"
  Array(0.61141766, dtype=float64)
  ```

* Fixes a bug where Catalyst would fail to apply quantum transforms and preserve
  QNode configuration settings when Autograph was enabled.
  [(#900)](https://github.com/PennyLaneAI/catalyst/pull/900)

* `pure_callback` will no longer cause a crash in the compiler if the return type
  signature is declared incorrectly and the callback function is differentiated.
  [(#916)](https://github.com/PennyLaneAI/catalyst/pull/916)

  Instead, this is caught early and a useful error message returned:

  ```python
  @catalyst.pure_callback
  def callback_fn(x) -> jax.ShapeDtypeStruct((2,), jnp.float32):
      return np.array([np.sin(x), np.cos(x)])
  
  callback_fn.fwd(lambda x: (callback_fn(x), x))
  callback_fn.bwd(lambda x, dy: (jnp.array([jnp.cos(x), -jnp.sin(x)]) @ dy,))
  
  @qjit
  @catalyst.grad
  def f(x):
      return jnp.sum(callback_fn(jnp.sin(x)))
  ```

  ```pycon
  >>> f(0.54)
  TypeError: Callback callback_fn expected type ShapedArray(float32[2]) but observed ShapedArray(float64[2]) in its return value
  ```

* AutoGraph will now correctly convert conditional statements where the condition is a non-boolean
  static value.
  [(#944)](https://github.com/PennyLaneAI/catalyst/pull/944)

  Internally, statically known non-boolean predicates (such as `1`) will be
  converted to `bool`:

  ```python
  @qml.qjit(autograph=True)
  def workflow(x):
      n = 1

      if n:
          y = x ** 2
      else:
          y = x

      return y
  ```

* `value_and_grad` will now correctly differentiate functions with multiple arguments.
  Previously, attempting to differentiate functions with multiple arguments, or pass
  the ``argnums`` argument, would result in an error.
  [(#1034)](https://github.com/PennyLaneAI/catalyst/pull/1034)

  ```python
  @qjit
  def g(x, y, z):
      def f(x, y, z):
          return x * y ** 2 * jnp.sin(z)
      return catalyst.value_and_grad(f, argnums=[1, 2])(x, y, z)
  ```

  ```pycon
  >>> g(0.4, 0.2, 0.6)
  (Array(0.00903428, dtype=float64),
   (Array(0.0903428, dtype=float64), Array(0.01320537, dtype=float64)))
  ```

* A bug is fixed in `catalyst.debug.get_cmain` to support multi-dimensional arrays as
  function inputs.
  [(#1003)](https://github.com/PennyLaneAI/catalyst/pull/1003)

<h3>Documentation</h3>

* A page has been added to the documentation, listing devices that are
  Catalyst compatible.
  [(#966)](https://github.com/PennyLaneAI/catalyst/pull/966)

<h3>Internal changes</h3>

* Adds `catalyst.from_plxpr.from_plxpr` for converting a PennyLane variant jaxpr into a
  Catalyst variant jaxpr.
  [(#837)](https://github.com/PennyLaneAI/catalyst/pull/837)

* Catalyst now uses Enzyme `v0.0.130`
  [(#898)](https://github.com/PennyLaneAI/catalyst/pull/898)

* When memrefs have no identity layout, memrefs copy operations are replaced by the linalg copy operation.
  It does not use a runtime function but instead lowers to scf and standard dialects. It also ensures
  a better compatibility with Enzyme.
  [(#917)](https://github.com/PennyLaneAI/catalyst/pull/917)

* LLVM's O2 optimization pipeline and Enzyme's AD transformations are now only run in the presence
  of gradients, significantly improving compilation times for programs without derivatives.
  Similarly, LLVM's coroutine lowering passes only run when `async_qnodes` is enabled in the QJIT decorator.
  [(#968)](https://github.com/PennyLaneAI/catalyst/pull/968)

* The function `inactive_callback` was renamed `__catalyst_inactive_callback`.
  [(#899)](https://github.com/PennyLaneAI/catalyst/pull/899)

* The function `__catalyst_inactive_callback` has the nofree attribute.
  [(#898)](https://github.com/PennyLaneAI/catalyst/pull/898)

* `catalyst.dynamic_one_shot` uses `postselect_mode="pad-invalid-samples"` in favour of `interface="jax"` when processing results.
  [(#956)](https://github.com/PennyLaneAI/catalyst/pull/956)

* Callbacks now have nicer identifiers in their MLIR representation. The identifiers include
  the name of the Python function being called back into.
  [(#919)](https://github.com/PennyLaneAI/catalyst/pull/919)

* Fix tracing of `SProd` operations to bring Catalyst in line with PennyLane v0.38.
  [(#935)](https://github.com/PennyLaneAI/catalyst/pull/935)

  After some changes in PennyLane, `Sprod.terms()` returns the terms as leaves
  instead of a tree. This means that we need to manually trace each term and
  finally multiply it with the coefficients to create a Hamiltonian.

* The function `mitigate_with_zne` accomodates a `folding` input argument for specifying the type of
  circuit folding technique to be used by the error-mitigation routine
  (only `global` value is supported to date.)
  [(#946)](https://github.com/PennyLaneAI/catalyst/pull/946)

* Catalyst's implementation of Lightning Kokkos plugin has been removed in favor of Lightning's one.
  [(#974)](https://github.com/PennyLaneAI/catalyst/pull/974)

* The `validate_device_capabilities` function is considered obsolete. Hence, it has been removed.
  [(#1045)](https://github.com/PennyLaneAI/catalyst/pull/1045)

<h3>Contributors</h3>

This release contains contributions from (in alphabetical order):

Joey Carter,
Alessandro Cosentino,
Lillian M. A. Frederiksen,
Josh Izaac,
Christina Lee,
Kunwar Maheep Singh,
Mehrdad Malekmohammadi,
Romain Moyard,
Erick Ochoa Lopez,
Mudit Pandey,
Nate Stemen,
Raul Torres,
Tzung-Han Juang,
Paul Haochen Wang,