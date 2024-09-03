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
