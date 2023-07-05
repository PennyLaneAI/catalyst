# Release 0.2.0

<h3>New features</h3>

* Catalyst programs can now be used inside of a larger JAX workflow which uses
  JIT compilation, automatic differentiation, and other JAX transforms.[#96]
  (https://github.com/PennyLaneAI/catalyst/pull/96)[#123]
  (https://github.com/PennyLaneAI/catalyst/pull/123)

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

  ``` python
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

  For a list of available devices, please see the [PennyLane-Braket]
  (https://amazon-braket-pennylane-plugin-python.readthedocs.io/en/latest/)
  documentation.

  Internally, the quantum instructions are generating OpenQASM3 kernels at
  runtime; these are then executed on both local (``braket.local.qubit``) and
  remote(``braket.aws.qubit``) devices backed by Amazon Braket Python SDK,
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

  Use `catalyst.jvp` to compute the forward-pass value and VJP:

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

* Add a Backprop operation for using AD at the LLVM level with Enzyme AD. It has a 
  bufferization pattern and a lowering to LLVM.
  [#107](https://github.com/PennyLaneAI/catalyst/pull/107)
  [#116](https://github.com/PennyLaneAI/catalyst/pull/116)

* Add the end-to-end support for multiple backend devices. The compilation flag
  ``ENABLE_LIGHTNING_KOKKOS=ON`` builds the runtime with support for PennyLane's
  ``lightning.kokkos``. Both ``lightning.qubit`` and ``lightning.kokkos`` can be
  chosen as available backend devices from the frontend.
  [#89](https://github.com/PennyLaneAI/catalyst/pull/89)

* Add support for ``var`` of general observables
  [#124](https://github.com/PennyLaneAI/catalyst/pull/124)


<h3>Improvements</h3>

* Improving error handling by throwing descriptive and unified expressions for runtime
  errors and assertions.
  [#92](https://github.com/PennyLaneAI/catalyst/pull/92)

* Improve interface for adding and re-using flags to quantum-opt commands.
  These are called pipelines, as they contain multiple passes.
  [#38](https://github.com/PennyLaneAI/catalyst/pull/38)

* Improve python compatibility by providing a stable signature for user generated functions.
  [#106](https://github.com/PennyLaneAI/catalyst/pull/106)

* Handle C++ exceptions without unwinding the whole stack.
  [#99](https://github.com/PennyLaneAI/catalyst/pull/99)

* Support constant negative step sizes in ``@for_loop`` loops.
  [#129](https://github.com/PennyLaneAI/catalyst/pull/129)

* Reduce the number of classical invocations by counting the number of gate parameters in
  the ``argmap`` function.
  [#136](https://github.com/PennyLaneAI/catalyst/pull/136)

  Prior to this, the computation of hybrid gradients executed all of the classical code
  being differentiated in a ``pcount`` function that solely counted the number of gate
  parameters in the quantum circuit. This was so ``argmap`` and other downstream
  functions could allocate memrefs large enough to store all gate parameters.

  Now, instead of counting the number of parameters separately, a dynamically-resizable
  array is used in the ``argmap`` function directly to store the gate parameters. This
  removes one invocation of all of the classical code being differentiated.

* Use Tablegen to define MLIR passes instead of C++ to reduce overhead of adding new passes.
  [#157](https://github.com/PennyLaneAI/catalyst/pull/157)

* Update JAX to `v0.4.10`.
  [#143](https://github.com/PennyLaneAI/catalyst/pull/143)

* Perform constant folding on wire indices for ``quantum.insert`` and ``quantum.extract`` ops,
  used when writing (resp. reading) qubits to (resp. from) quantum registers.
  [#161](https://github.com/PennyLaneAI/catalyst/pull/161)

* Use the differentiation method on the QNode rather than the ``grad`` function to better
  match PennyLane's differentiation API.

  This is an intermediate step towards differentiating functions that internally call multiple
  QNodes, and towards supporting differentiation of classical postprocessing.
  [#163](https://github.com/PennyLaneAI/catalyst/pull/163)

* Represent known named observables as members of an MLIR Enum rather than a raw integer.
  This improves IR readability.
  [#165](https://github.com/PennyLaneAI/catalyst/pull/165)

<h3>Breaking changes</h3>

<h3>Bug fixes</h3>

* Fix a bug in the mapping from logical to concrete qubits for mid-circuit measurements.
  [#80](https://github.com/PennyLaneAI/catalyst/pull/80)

* Fix a bug in the way gradient result type is inferred.
  [#84](https://github.com/PennyLaneAI/catalyst/pull/84)

* Fix a memory regression and reduce memory footprint by removing unnecessary temporary buffers.
  [#100](https://github.com/PennyLaneAI/catalyst/pull/100)

* Provide a new abstraction to the ``QuantumDevice`` interface in the runtime called ``MemRefView``.
  C++ implementations of the interface can iterate through and directly store results into the
  ``MemRefView`` independant of the underlying memory layout. This can eliminate redundant buffer
  copies at the interface boundaries, which has been applied to existing devices.
  [#109](https://github.com/PennyLaneAI/catalyst/pull/109)

* Reduce memory utilization by transferring ownership of buffers from the runtime to Python instead
  of copying them. This includes adding a compiler pass that copies global buffers into the heap
  as global buffers cannot be transferred to Python.
  [#112](https://github.com/PennyLaneAI/catalyst/pull/112)

* Temporary fix of use-after-free and dependency of uninitialized memory.
  [#121](https://github.com/PennyLaneAI/catalyst/pull/121)

* Fixes file renaming within pass pipelines.
  [#126](https://github.com/PennyLaneAI/catalyst/pull/126)

* Fixes the issue with the ``do_queue`` deprecation warnings in PennyLane.
  [#146](https://github.com/PennyLaneAI/catalyst/pull/146)

* Fixes the issue with gradients failing to work with ``jnp.array`` as constants.
  [#152](https://github.com/PennyLaneAI/catalyst/pull/152)
  
  An example of a newly supported workflow:
  
  ``` python
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
  This means the operation is no longer resticted to acting on ``qml.qnode``s only.
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
