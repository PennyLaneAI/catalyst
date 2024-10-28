# Release 0.9.0 (current release)

<h3>New features</h3>

* Experimental integration of the PennyLane capture module is available. It currently only supports
  quantum gates, without control flow.
  [(#1109)](https://github.com/PennyLaneAI/catalyst/pull/1109)

  To trigger the PennyLane pipeline for capturing the program as a Jaxpr, simply set
  `experimental_capture=True` in the qjit decorator.

  ```python
  import pennylane as qml
  from catalyst import qjit

  dev = qml.device("lightning.qubit", wires=1)

  @qjit(experimental_capture=True)
  @qml.qnode(dev)
  def circuit():
      qml.Hadamard(0)
      qml.CNOT([0, 1])
      return qml.expval(qml.Z(0))
  ```

* Shot-vector support for Catalyst: Introduces support for shot-vectors in Catalyst, currently
  available for `qml.sample` measurements in the `lightning.qubit` device. Shot-vectors now allow
  elements of the form `((20, 5),)`, which is equivalent to `(20,)*5` or `(20, 20, 20, 20, 20)`.
  Furthermore, multiple `qml.sample` calls can now be returned from the same program, and can be
  structured using Python containers. For example, a program can return a dictionary like
  `return {"first": qml.sample(), "second": qml.sample()}`.
  [(#1051)](https://github.com/PennyLaneAI/catalyst/pull/1051)

  For example,

  ```python
  import pennylane as qml
  from catalyst import qjit

  dev = qml.device("lightning.qubit", wires=1, shots=((5, 2), 7))

  @qjit
  @qml.qnode(dev)
  def circuit():
      qml.Hadamard(0)
      return qml.sample()
  ```

  ```pycon
  >>> circuit()
  (Array([[0], [1], [0], [1], [1]], dtype=int64),
  Array([[0], [1], [1], [0], [1]], dtype=int64),
  Array([[1], [0], [1], [1], [0], [1],[0]], dtype=int64))
  ```

* A new function `catalyst.passes.pipeline` allows the quantum-circuit-transformation pass pipeline
  for QNodes within a qjit-compiled workflow to be configured.
  [(#1131)](https://github.com/PennyLaneAI/catalyst/pull/1131)

  ```python
  my_passes = {
      "cancel_inverses": {},
      "my_circuit_transformation_pass": {"my-option" : "my-option-value"},
  }
  dev = qml.device("lightning.qubit", wires=2)

  @pipeline(my_passes)
  @qnode(dev)
  def circuit(x):
      qml.RX(x, wires=0)
      return qml.expval(qml.PauliZ(0))

  @qjit
  def fn(x):
      return jnp.sin(circuit(x ** 2))
  ```

  `pipeline` can also be used to specify different pass pipelines for different parts of the
  same qjit-compiled workflow:

  ```python
  my_pipeline = {
      "cancel_inverses": {},
      "my_circuit_transformation_pass": {"my-option" : "my-option-value"},
  }

  my_other_pipeline = {"cancel_inverses": {}}

  @qjit
  def fn(x):
      circuit_pipeline = pipeline(my_pipeline)(circuit)
      circuit_other = pipeline(my_other_pipeline)(circuit)
      return jnp.abs(circuit_pipeline(x) - circuit_other(x))
  ```

  The pass pipeline order and options can be configured *globally* for a qjit-compiled function, by
  using the `circuit_transform_pipeline` argument of the :func:`~.qjit` decorator.

  ```python
  my_passes = {
      "cancel_inverses": {},
      "my_circuit_transformation_pass": {"my-option" : "my-option-value"},
  }

  @qjit(circuit_transform_pipeline=my_passes)
  def fn(x):
      return jnp.sin(circuit(x ** 2))
  ```

  Global and local (via `@pipeline`) configurations can coexist, however local pass pipelines will
  always take precedence over global pass pipelines.

  The available MLIR passes are listed and documented in the
  [`catalyst.passes` module documentation](https://docs.pennylane.ai/projects/catalyst/en/stable/code/__init__.html#module-catalyst.passes).

* A peephole merge rotations pass is now available in MLIR. It can be added to `catalyst.passes.pipeline`,
  or the Python function `catalyst.passes.merge_rotations` can be directly called on a `QNode`.
  [(#1162)](https://github.com/PennyLaneAI/catalyst/pull/1162)
  [(#1205)](https://github.com/PennyLaneAI/catalyst/pull/1205)
  [(#1206)](https://github.com/PennyLaneAI/catalyst/pull/1206)

  Using the pipeline, one can run:

  ```python
  from catalys.passes import pipeline

  my_passes = {
      "merge_rotations": {}
  }

  @qjit(circuit_transform_pipeline=my_passes)
  @qml.qnode(qml.device("lightning.qubit", wires=1))
  def g(x: float):
      qml.RX(x, wires=0)
      qml.RX(x, wires=0)
      qml.Hadamard(wires=0)
      return qml.expval(qml.PauliZ(0))
  ```

  Using the Python function, one can run:

  ```python
  from catalys.passes import merge_rotations

  @qjit
  @merge_rotations
  @qml.qnode(qml.device("lightning.qubit", wires=1))
  def g(x: float):
      qml.RX(x, wires=0)
      qml.RX(x, wires=0)
      qml.Hadamard(wires=0)
      return qml.expval(qml.PauliZ(0))
  ```

* Catalyst Autograph now supports updating a single index or a slice of JAX arrays using Python's
  array assignment operator syntax.
  [(#769)](https://github.com/PennyLaneAI/catalyst/pull/769)
  [(#1143)](https://github.com/PennyLaneAI/catalyst/pull/1143)

  Using operator assignment syntax in favor of `at...op` expressions is now possible for the
  following operations:

  - `x[i] += y` in favor of `x.at[i].add(y)`
  - `x[i] -= y` in favor of `x.at[i].add(-y)`
  - `x[i] *= y` in favor of `x.at[i].multiply(y)`
  - `x[i] /= y` in favor of `x.at[i].divide(y)`
  - `x[i] **= y` in favor of `x.at[i].power(y)`

  ```python
  @qjit(autograph=True)
  def f(x):
      first_dim = x.shape[0]
      result = jnp.copy(x)

      for i in range(first_dim):
        result[i] *= 2  # This is now supported

      return result
  ```

  ```pycon
  >>> f(jnp.array([1, 2, 3]))
  Array([2, 4, 6], dtype=int64)
  ```

* Catalyst now has a standalone compiler tool called `catalyst-cli` that quantum compiles MLIR
  input files into an object file independent of the Python frontend.
  [(#1208)](https://github.com/PennyLaneAI/catalyst/pull/1208)

  This compiler tool combines three stages of compilation:

  1. `quantum-opt`: Performs the MLIR-level optimizations and lowers the input dialect to the LLVM dialect.
  2. `mlir-translate`: Translates the input in the LLVM dialect into LLVM IR.
  3. `llc`: Performs lower-level optimizations and creates the object file.

  `catalyst-cli` runs all three stages under the hood by default, but it also has the ability to run
  each stage individually. For example:

  ```console
  # Creates both the optimized IR and an object file
  catalyst-cli input.mlir -o output.o

  # Only performs MLIR optimizations
  catalyst-cli --tool=opt input.mlir -o llvm-dialect.mlir

  # Only lowers LLVM dialect MLIR input to LLVM IR
  catalyst-cli --tool=translate llvm-dialect.mlir -o llvm-ir.ll

  # Only performs lower-level optimizations and creates object file
  catalyst-cli --tool=llc llvm-ir.ll -o output.o
  ```

* Static arguments of a qjit-compiled function can now be indicated by a `static_argnames` argument
  to `qjit`.
  [(#1158)](https://github.com/PennyLaneAI/catalyst/pull/1158)

  ```python
  @qjit(static_argnames="y")
  def f(x, y):
      if y < 10:  # y needs to be marked as static since its concrete boolean value is needed
          return x + y

  @qjit(static_argnames=["x","y"])
  def g(x, y):
      if x < 10 and y < 10:
          return x + y

  res_f = f(1, 2)
  res_g = g(3, 4)
  print(res_f, res_g)
  ```

  ```pycon
  3 7
  ```

<h3>Improvements</h3>

* Adjoint canonicalization is now available in MLIR for `CustomOp` and `MultiRZOp`. It can be used
  with the `--canonicalize` pass in `quantum-opt`.
  [(#1205)](https://github.com/PennyLaneAI/catalyst/pull/1205)

* Implements a Catalyst runtime plugin that mocks out all functions in the QuantumDevice interface.
  [(#1179)](https://github.com/PennyLaneAI/catalyst/pull/1179)

* Scalar tensors are eliminated from control flow operations in the program, and are replaced with
  bare scalars instead. This improves compilation time and memory usage at runtime by avoiding heap
  allocations and reducing the amount of instructions.
  [(#1075)](https://github.com/PennyLaneAI/catalyst/pull/1075)

  A new MLIR pass `detensorize-scf` is added that works in conjunction with the existing
  `linalg-detensorize` pass to detensorize input programs. The IR generated by JAX wraps all values
  in the program in tensors, including scalars, leading to unnecessary memory allocations for
  programs compiled to CPU via the MLIR-to-LLVM pipeline.

* Bufferization of `gradient.ForwardOp` and `gradient.ReverseOp` now requires three steps:
  `gradient-preprocessing`, `gradient-bufferize`, and `gradient-postprocessing`.
  `gradient-bufferize` has a new rewrite for `gradient.ReturnOp`.
  [(#1139)](https://github.com/PennyLaneAI/catalyst/pull/1139)

* The decorator `self_inverses` now supports all Hermitian Gates.
  [(#1136)](https://github.com/PennyLaneAI/catalyst/pull/1136)

  The full list of supported gates are as follows:

  One-qubit Gates:
  - [`qml.Hadamard`](https://docs.pennylane.ai/en/stable/code/api/pennylane.Hadamard.html)
  - [`qml.PauliX`](https://docs.pennylane.ai/en/stable/code/api/pennylane.PauliX.html)
  - [`qml.PauliY`](https://docs.pennylane.ai/en/stable/code/api/pennylane.PauliY.html)
  - [`qml.PauliZ`](https://docs.pennylane.ai/en/stable/code/api/pennylane.PauliZ.html)

  Two-qubit Gates:
  - [`qml.CNOT`](https://docs.pennylane.ai/en/stable/code/api/pennylane.CNOT.html)
  - [`qml.CY`](https://docs.pennylane.ai/en/stable/code/api/pennylane.CY.html)
  - [`qml.CZ`](https://docs.pennylane.ai/en/stable/code/api/pennylane.CZ.html)
  - [`qml.SWAP`](https://docs.pennylane.ai/en/stable/code/api/pennylane.SWAP.html)

  Three-qubit Gates:
  - [`qml.Toffoli`](https://docs.pennylane.ai/en/stable/code/api/pennylane.Toffoli.html)

* Support is expanded for backend devices that exclusively return samples in the measurement basis.
  Pre- and post-processing now allows `qjit` to be used on these devices with `qml.expval`,
  `qml.var` and `qml.probs` measurements in addition to `qml.sample`, using the
  `measurements_from_samples` transform.
  [(#1106)](https://github.com/PennyLaneAI/catalyst/pull/1106)

* Catalyst now supports numpy 2.0
  [(#1119)](https://github.com/PennyLaneAI/catalyst/pull/1119)
  [(#1182)](https://github.com/PennyLaneAI/catalyst/pull/1182)

* Importing Catalyst will now pollute less of JAX's global variables by using `LoweringParameters`.
  [(#1152)](https://github.com/PennyLaneAI/catalyst/pull/1152)

* Compiling `qnode`s to asynchronous functions will no longer print to stderr in case of an error.
  [(#645)](https://github.com/PennyLaneAI/catalyst/pull/645)

* Cached primitive lowerings is used instead of a custom cache structure.
  [(#1159)](https://github.com/PennyLaneAI/catalyst/pull/1159)

* Calling gradients twice (with same GradParams) will now only lower to a single MLIR function.
  [(#1172)](https://github.com/PennyLaneAI/catalyst/pull/1172)

* Samples on lightning.qubit/kokkos can now be seeded with `qjit(seed=...)`.
  [(#1164)](https://github.com/PennyLaneAI/catalyst/pull/1164)

* The compiler pass `-remove-chained-self-inverse` can now also cancel adjoints of arbitrary unitary
  operations (in addition to the named Hermitian gates).
  [(#1186)](https://github.com/PennyLaneAI/catalyst/pull/1186)
  [(#1211)](https://github.com/PennyLaneAI/catalyst/pull/1211)

<h3>Breaking changes</h3>

* Remove `static_size` field from `AbstractQreg` class.
  [(#1113)](https://github.com/PennyLaneAI/catalyst/pull/1113)

  This reverts a previous breaking change.

* Nesting qnodes now raises an error.
  [(#1176)](https://github.com/PennyLaneAI/catalyst/pull/1176)

  This is unlikely to affect users since only under certain conditions did
  nesting qnodes worked successfully.

* Removes `debug.compile_from_mlir`.
  [(#1181)](https://github.com/PennyLaneAI/catalyst/pull/1181)

  Please use `debug.replace_ir`.

* Removes `compiler.last_compiler_output`.
  [(#1208)](https://github.com/PennyLaneAI/catalyst/pull/1208)

  Please use `compiler.get_output_of("last", workspace)`

<h3>Bug fixes</h3>

* Fixes a bug in `catalyst.mitigate_with_zne` that would lead to incorrectly extrapolated results.
  [(#1213)](https://github.com/PennyLaneAI/catalyst/pull/1213)

* Fixes a bug preventing the target of `qml.adjoint` and `qml.ctrl` calls from being transformed by
  AutoGraph.
  [(#1212)](https://github.com/PennyLaneAI/catalyst/pull/1212)

* Resolves a bug where `mitigate_with_zne` does not work properly with shots and devices supporting
  only Counts and Samples (e.g. Qrack). (transform: `measurements_from_sample`).
  [(#1165)](https://github.com/PennyLaneAI/catalyst/pull/1165)

* Resolves a bug in the `vmap` function when passing shapeless values to the target.
  [(#1150)](https://github.com/PennyLaneAI/catalyst/pull/1150)

* Fixes a bug that resulted in an error message when using `qml.cond` on callables with arguments.
  [(#1151)](https://github.com/PennyLaneAI/catalyst/pull/1151)

* Fixes a bug that prevented taking the gradient of nested accelerate callbacks.
  [(#1156)](https://github.com/PennyLaneAI/catalyst/pull/1156)

* Fixes some small issues with scatter lowering:
  [(#1216)](https://github.com/PennyLaneAI/catalyst/pull/1216)
  [(#1217)](https://github.com/PennyLaneAI/catalyst/pull/1217)

  - Registers the func dialect as a requirement for running the scatter lowering pass.
  - Emits error if `%input`, `%update` and `%result` are not of length 1 instead of segfaulting.

* Fixes a performance issue with vmap with its root cause in the lowering of the scatter operation.
  [(#1214)](https://github.com/PennyLaneAI/catalyst/pull/1214)

<h3>Internal changes</h3>

* Removes deprecated PennyLane code across the frontend.
  [(#1168)](https://github.com/PennyLaneAI/catalyst/pull/1168)

* Updates Enzyme to version `v0.0.149`.
  [(#1142)](https://github.com/PennyLaneAI/catalyst/pull/1142)

* Removes the `MemMemCpyOptPass` in llvm O2 (applied for Enzyme), which reduces bugs when running
  gradient-like functions.
  [(#1063)](https://github.com/PennyLaneAI/catalyst/pull/1063)

* Functions with multiple tapes are now split with a new mlir pass `--split-multiple-tapes`, with
  one tape per function. The reset routine that makes a measurement between tapes and inserts an X
  gate if measured one is no longer used.
  [(#1017)](https://github.com/PennyLaneAI/catalyst/pull/1017)
  [(#1130)](https://github.com/PennyLaneAI/catalyst/pull/1130)

* Prefer creating new `qml.devices.ExecutionConfig` objects over using the global
  `qml.devices.DefaultExecutionConfig`. Doing so helps avoid unexpected bugs and test failures in
  case the `DefaultExecutionConfig` object becomes modified from its original state.
  [(#1137)](https://github.com/PennyLaneAI/catalyst/pull/1137)

* Remove the old `QJITDevice` API.
  [(#1138)](https://github.com/PennyLaneAI/catalyst/pull/1138)

* The device-capability loading mechanism has been moved into the `QJITDevice` constructor.
  [(#1141)](https://github.com/PennyLaneAI/catalyst/pull/1141)

* Several functions related to device capabilities have been refactored.
  [(#1149)](https://github.com/PennyLaneAI/catalyst/pull/1149)

  In particular, the signatures of `get_device_capability`, `catalyst_decompose`,
 `catalyst_acceptance`, and `QJITDevice.__init__` have changed, and the `pennylane_operation_set`
  function has been removed entirely.

* Catalyst now generates nested modules denoting quantum programs.
  [(#1144)](https://github.com/PennyLaneAI/catalyst/pull/1144)

  Similar to MLIR's `gpu.launch_kernel` function, Catalyst, now supports a `call_function_in_module`.
  This allows Catalyst to call functions in modules and have modules denote a quantum kernel. This
  will allow for device-specific optimizations and compilation pipelines.

  At the moment, no one is using this. This is just the necessary scaffolding to support device-
  specific transformations. As such, the module will be inlined to preserve current semantics.
  However, in the future, we will explore lowering this nested module into other IRs/binary formats
  and lowering `call_function_in_module` to something that can dispatch calls to another runtime/VM.


<h3>Contributors</h3>

This release contains contributions from (in alphabetical order):

Joey Carter,
Spencer Comin,
Amintor Dusko,
Lillian M.A. Frederiksen,
Sengthai Heng,
David Ittah,
Mehrdad Malekmohammadi,
Vincent Michaud-Rioux,
Romain Moyard,
Erick Ochoa Lopez,
Daniel Strano,
Raul Torres,
Paul Haochen Wang.
