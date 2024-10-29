# Release 0.9.0 (current release)

<h3>New features</h3>

* Catalyst now supports the specification of shot-vectors when used with
  `qml.sample` measurements on the `lightning.qubit` device.
  [(#1051)](https://github.com/PennyLaneAI/catalyst/pull/1051)

  Shot-vectors allow shots to be specified as a list of shots, `[20, 1, 100]`,
  or as a tuple of the form `((num_shots, repetitions), ...)` such that
  `((20, 3), (1, 100))` is equivalent to `shots=[20, 20, 20, 1, 1, ..., 1]`.

  This can result in more efficient quantum execution, as a single job representing
  the total number of shots is executed on the quantum device, with the measurement
  post-processing then course-grained with respect to the shot-vector.

  For example,

  ```python
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
  Array([[1], [0], [1], [1], [0], [1], [0]], dtype=int64))
  ```

  Note that other measurement types, such as `expval` and `probs`, currently
  do not support shot-vectors.

* A new function `catalyst.passes.pipeline` allows the quantum-circuit-transformation pass pipeline
  for QNodes within a qjit-compiled workflow to be configured.
  [(#1131)](https://github.com/PennyLaneAI/catalyst/pull/1131)

  ```python
  my_passes = {
      "cancel_inverses": {},
      "my_circuit_transformation_pass": {"my-option" : "my-option-value"},
  }
  dev = qml.device("lightning.qubit", wires=2)

  @pipeline(pass_pipeline=my_passes)
  @qml.qnode(dev)
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
      circuit_pipeline = pipeline(pass_pipeline=my_pipeline)(circuit)
      circuit_other = pipeline(pass_pipeline=my_other_pipeline)(circuit)
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
  [passes module documentation](https://docs.pennylane.ai/projects/catalyst/en/stable/code/__init__.html#module-catalyst.passes).

* A peephole merge rotations pass, which acts similarly to the Python-based [PennyLane merge rotations transform](https://docs.pennylane.ai/en/stable/code/api/pennylane.transforms.merge_rotations.html),
  is now available in MLIR and can be applied to QNodes within a qjit-compiled function.
  [(#1162)](https://github.com/PennyLaneAI/catalyst/pull/1162)
  [(#1205)](https://github.com/PennyLaneAI/catalyst/pull/1205)
  [(#1206)](https://github.com/PennyLaneAI/catalyst/pull/1206)

  The `merge_rotations` pass can be provided to the `catalyst.passes.pipeline`
  decorator:

  ```python
  from catalyst.passes import pipeline

  my_passes = {
      "merge_rotations": {}
  }

  dev = qml.device("lightning.qubit", wires=1)

  @qjit(circuit_transform_pipeline=my_passes)
  @qml.qnode(dev)
  def g(x: float):
      qml.RX(x, wires=0)
      qml.RX(x, wires=0)
      qml.Hadamard(wires=0)
      return qml.expval(qml.PauliX(0))
  ```

  It can also be applied directly to qjit-compiled QNodes via
  the `catalyst.passes.merge_rotations` Python decorator:

  ```python
  from catalys.passes import merge_rotations

  @qjit
  @merge_rotations
  @qml.qnode(dev)
  def g(x: float):
      qml.RX(x, wires=0)
      qml.RX(x, wires=0)
      qml.Hadamard(wires=0)
      return qml.expval(qml.PauliX(0))
  ```

* Static arguments of a qjit-compiled function can now be indicated by name via a `static_argnames` argument
  to the `qjit` decorator.
  [(#1158)](https://github.com/PennyLaneAI/catalyst/pull/1158)

  Specified static argument names will be treated as compile-time static
  values, allowing any hashable Python object to be passed to this
  function argument during compilation.

  ```pycon
  >>> @qjit(static_argnames="y")
  ... def f(x, y):
  ...     print(f"Compiling with y={y}")
  ...     return x + y
  >>> f(0.5, 0.3)
  Compiling with y=0.3
  ```

  The function will only be re-compiled if the hash value of the static arguments change. Otherwise,
  re-using previous static argument values will result in no re-compilation:

  ```pycon
  Array(0.8, dtype=float64)
  >>> f(0.1, 0.3)  # no re-compilation occurs
  Array(0.4, dtype=float64)
  >>> f(0.1, 0.4)  # y changes, re-compilation
  Compiling with y=0.4
  Array(0.5, dtype=float64)
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

  Note that `catalyst-cli` is only available when Catalyst is built from
  source, and is not included when installing Catalyst via pip or from
  wheels.

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

<h3>Improvements</h3>

* Multiple `qml.sample` calls can now be returned from the same program, and can be
  structured using Python containers. For example, a program can return a dictionary of the form
  `return {"first": qml.sample(), "second": qml.sample()}`.
  [(#1051)](https://github.com/PennyLaneAI/catalyst/pull/1051)

* Catalyst now ships with `null.qubit`, a Catalyst runtime plugin that mocks out all
  functions in the QuantumDevice interface. This device is provided as a convenience for testing
  and benchmarking purposes.
  [(#1179)](https://github.com/PennyLaneAI/catalyst/pull/1179)

  ```python
  qml.device("null.qubit", wires=1)

  @qml.qjit
  @qml.qnode(dev)
  def g(x):
      qml.RX(x, wires=0)
      return qml.probs(wires=[0])
  ```

* Setting the `seed` argument in the `qjit` decorator will now seed sampled results, in addition
  to mid-circuit measurement results.
  [(#1164)](https://github.com/PennyLaneAI/catalyst/pull/1164)

  ```python
  dev = qml.device("lightning.qubit", wires=1, shots=10)

  @qml.qnode(dev)
  def circuit(x):
      qml.RX(x, wires=0)
      m = catalyst.measure(0)

      if m:
          qml.Hadamard(0)

      return qml.sample()

  @qml.qjit(seed=37, autograph=True)
  def workflow(x):
      return jnp.squeeze(jnp.stack([circuit(x) for i in range(4)]))
  ```
  ```pycon
  >>> workflow(1.8)
  Array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
       [1, 1, 0, 0, 1, 1, 0, 0, 1, 0],
       [0, 0, 1, 0, 1, 1, 0, 0, 1, 1],
       [1, 1, 1, 0, 0, 1, 1, 0, 1, 1]], dtype=int64)
  >>> workflow(1.8)
  Array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [1, 1, 0, 0, 1, 1, 0, 0, 1, 0],
         [0, 0, 1, 0, 1, 1, 0, 0, 1, 1],
         [1, 1, 1, 0, 0, 1, 1, 0, 1, 1]], dtype=int64)
  ```

  Note that statistical measurement processes such as `expval`, `var`, and `probs`
  are currently not affected by seeding when shot noise is present.

* The `cancel_inverses` MLIR compilation pass (`-remove-chained-self-inverse`) now supports
  cancelling all Hermitian gates, as well as adjoints of arbitrary unitary
  operations.
  [(#1136)](https://github.com/PennyLaneAI/catalyst/pull/1136)
  [(#1186)](https://github.com/PennyLaneAI/catalyst/pull/1186)
  [(#1211)](https://github.com/PennyLaneAI/catalyst/pull/1211)

  For the full list of supported Hermitian gates please see the `cancel_inverses` documentation
  in `catalyst.passes`.

* Support is expanded for backend devices that exclusively return samples in the measurement basis.
  Pre- and post-processing now allows `qjit` to be used on these devices with `qml.expval`,
  `qml.var` and `qml.probs` measurements in addition to `qml.sample`, using the
  `measurements_from_samples` transform.
  [(#1106)](https://github.com/PennyLaneAI/catalyst/pull/1106)

* Scalar tensors are eliminated from control flow operations in the program, and are replaced with
  bare scalars instead. This improves compilation time and memory usage at runtime by avoiding heap
  allocations and reducing the amount of instructions.
  [(#1075)](https://github.com/PennyLaneAI/catalyst/pull/1075)

* Catalyst now supports NumPy 2.0.
  [(#1119)](https://github.com/PennyLaneAI/catalyst/pull/1119)
  [(#1182)](https://github.com/PennyLaneAI/catalyst/pull/1182)

* Compiling QNodes to asynchronous functions will no longer print to `stderr` in case of an error.
  [(#645)](https://github.com/PennyLaneAI/catalyst/pull/645)

* Gradient computations have been made more efficient, as calling gradients twice
  (with the same gradient parameters) will now only lower to a single MLIR function.
  [(#1172)](https://github.com/PennyLaneAI/catalyst/pull/1172)

<h3>Breaking changes</h3>

* Remove `static_size` field from `AbstractQreg` class.
  [(#1113)](https://github.com/PennyLaneAI/catalyst/pull/1113)

  This reverts a previous breaking change.

* Nesting QNodes within one another now raises an error.
  [(#1176)](https://github.com/PennyLaneAI/catalyst/pull/1176)

* The `debug.compile_from_mlir` function has been removed;
  please use `debug.replace_ir` instead.
  [(#1181)](https://github.com/PennyLaneAI/catalyst/pull/1181)

* The `compiler.last_compiler_output` function has been removed;
  please use `compiler.get_output_of("last", workspace)` instead.
  [(#1208)](https://github.com/PennyLaneAI/catalyst/pull/1208)

<h3>Bug fixes</h3>

* Fixes a bug in `catalyst.mitigate_with_zne` that would lead to incorrectly extrapolated results.
  [(#1213)](https://github.com/PennyLaneAI/catalyst/pull/1213)

* Fixes a bug preventing the target of `qml.adjoint` and `qml.ctrl` calls from being transformed by
  AutoGraph.
  [(#1212)](https://github.com/PennyLaneAI/catalyst/pull/1212)

* Resolves a bug where `mitigate_with_zne` does not work properly with shots and devices supporting
  only counts and samples (e.g., Qrack).
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

* Fixes a performance issue with `catalyst.vmap` with its root cause in
  the lowering of the scatter operation.
  [(#1214)](https://github.com/PennyLaneAI/catalyst/pull/1214)

<h3>Internal changes</h3>

* Removes deprecated PennyLane code across the frontend.
  [(#1168)](https://github.com/PennyLaneAI/catalyst/pull/1168)

* Updates Enzyme to version `v0.0.149`.
  [(#1142)](https://github.com/PennyLaneAI/catalyst/pull/1142)

* Adjoint canonicalization is now available in MLIR for `CustomOp` and `MultiRZOp`. It can be used
  with the `--canonicalize` pass in `quantum-opt`.
  [(#1205)](https://github.com/PennyLaneAI/catalyst/pull/1205)

* Removes the `MemMemCpyOptPass` in llvm O2 (applied for Enzyme), which reduces bugs when running
  gradient-like functions.
  [(#1063)](https://github.com/PennyLaneAI/catalyst/pull/1063)

* Bufferization of `gradient.ForwardOp` and `gradient.ReverseOp` now requires three steps:
  `gradient-preprocessing`, `gradient-bufferize`, and `gradient-postprocessing`.
  `gradient-bufferize` has a new rewrite for `gradient.ReturnOp`.
  [(#1139)](https://github.com/PennyLaneAI/catalyst/pull/1139)

* A new MLIR pass `detensorize-scf` is added that works in conjunction with the existing
  `linalg-detensorize` pass to detensorize input programs. The IR generated by JAX wraps all values
  in the program in tensors, including scalars, leading to unnecessary memory allocations for
  programs compiled to CPU via the MLIR-to-LLVM pipeline.
  [(#1075)](https://github.com/PennyLaneAI/catalyst/pull/1075)

* Importing Catalyst will now pollute less of JAX's global variables by using `LoweringParameters`.
  [(#1152)](https://github.com/PennyLaneAI/catalyst/pull/1152)

* Cached primitive lowerings is used instead of a custom cache structure.
  [(#1159)](https://github.com/PennyLaneAI/catalyst/pull/1159)

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
