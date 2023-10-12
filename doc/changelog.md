# Release 0.3.1-dev

<h3>New features</h3>

* Add lowering to tensor dialect for MHLO scatter. It unlocks indexing and updating jax arrays.
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

* Catalyst users can now use Python for loop statements in their programs without having to
  explicitly use the functional `catalyst.for_loop` form!
  [#258](https://github.com/PennyLaneAI/catalyst/pull/258)

  This feature extends the existing AutoGraph support for Python if statements with Python for
  loops. The following example is now supported:

  ```python
  dev = qml.device("lightning.qubit", wires=n)

  @qjit(autograph=True)
  @qml.qnode(dev)
  def f(n):
      for i in range(n):
          qml.Hadamard(wires=i)

      ...

      return qml.expval(qml.PauliZ(0))
  ```

* The quantum control operation can now be used in conjunction with Catalyst control flow, such as
  loops and conditionals. For this purpose a new instruction, `catalyst.ctrl`, has been added.
  [(#282)](https://github.com/PennyLaneAI/catalyst/pull/282)

  `catalyst.ctrl` can wrap around quantum functions which contain the Catalyst `cond`,
  `for_loop`, and `while_loop` primitives.

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

<h3>Improvements</h3>

* Update the Lightning backend device to work with the PL-Lightning monorepo.
  [(#259)](https://github.com/PennyLaneAI/catalyst/pull/259)

* Move to an alternate compiler driver in C++. This improves compile-time performance by
  avoiding *round-tripping*, which is when the entire program being compiled is dumped to
  a textual form and re-parsed by another tool.

  This is also a requirement for providing custom metadata at the LLVM level, which is
  necessary for better integration with tools like Enzyme. Finally, this makes it more natural
  to improve error messages originating from C++ when compared to the prior subprocess-based
  approach.
  [(#216)](https://github.com/PennyLaneAI/catalyst/pull/216)

* Build both `"lightning.qubit"` and `"lightning.kokkos"` against the PL-Lightning monorepo.
  [(#277)](https://github.com/PennyLaneAI/catalyst/pull/277)

* Support the `braket.devices.Devices` enum class and `s3_destination_folder`
  for AWS Braket remove devices.
  [(#278)](https://github.com/PennyLaneAI/catalyst/pull/278)

* Avoid building `opt` from the build process as it is unnecessary.
  Avoid downloading the `wheel` Python package as it is unnecessary.
  [(#298)](https://github.com/PennyLaneAI/catalyst/pull/298)

<h3>Breaking changes</h3>

<h3>Bug fixes</h3>

* Remove undocumented package dependency on the zlib/zstd compression library.
  [(#308)](https://github.com/PennyLaneAI/catalyst/pull/308)

* Add support for applying the `adjoint` operation to `QubitUnitary` gates.
  `QubitUnitary` was not able to be `adjoint`ed when the variable holding the unitary matrix might
  change. This can happen, for instance, inside of a for loop.
  To solve this issue, the unitary matrix gets stored in the array list via push and pops.
  The unitary matrix is later reconstructed from the array list and `QubitUnitary` can be executed
  in the `adjoint`ed context.
  [(#304)](https://github.com/PennyLaneAI/catalyst/pull/304)

<h3>Contributors</h3>

This release contains contributions from (in alphabetical order):

Ali Asadi,
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
