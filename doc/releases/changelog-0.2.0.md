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
