# Release 0.9.0-dev

<h3>New features</h3>

* Experimental integration of the PennyLane capture module is available. It currently only supports
  quantum gates, without control flow.
  [(#1109)](https://github.com/PennyLaneAI/catalyst/pull/1109)

  To trigger the PennyLane pipeline for capturing the program as a JaxPR, one needs to simply
  set `experimental_capture=True` in the qjit decorator.

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

* A new function `catalyst.passes.pipeline` allows the quantum circuit transformation pass pipeline
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

  For a list of available passes, please see the
  [catalyst.passes module documentation](https://docs.pennylane.ai/projects/catalyst/en/stable/code/__init__.html#module-catalyst.passes).

  The pass pipeline order and options can be configured *globally* for a qjit-compiled
  function, by using the `circuit_transform_pipeline` argument of the :func:`~.qjit`
  decorator.

  ```python
    my_passes = {
        "cancel_inverses": {},
        "my_circuit_transformation_pass": {"my-option" : "my-option-value"},
    }

    @qjit(circuit_transform_pipeline=my_passes)
    def fn(x):
        return jnp.sin(circuit(x ** 2))
  ```

  Global and local (via `@pipeline`) configurations can coexist, however local pass pipelines
  will always take precedence over global pass pipelines.

  Available MLIR passes are now documented and available within the
  [catalyst.passes module documentation](https://docs.pennylane.ai/projects/catalyst/en/stable/code/__init__.html#module-catalyst.passes).

* Catalyst Autograph now supports updating a single index or a slice of JAX arrays using Python's
  array assignment operator syntax.
  [(#769)](https://github.com/PennyLaneAI/catalyst/pull/769)
  [(#1143)](https://github.com/PennyLaneAI/catalyst/pull/1143)

  Using operator assignment syntax in favor of `at...op` expressions is now possible for the
  following operations:
  * `x[i] += y` in favor of `x.at[i].add(y)`
  * `x[i] -= y` in favor of `x.at[i].add(-y)`
  * `x[i] *= y` in favor of `x.at[i].multiply(y)`
  * `x[i] /= y` in favor of `x.at[i].divide(y)`
  * `x[i] **= y` in favor of `x.at[i].power(y)`

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

<h3>Improvements</h3>

* Scalar tensors are eliminated from control flow operations in the program, and are replaced with
  bare scalars instead. This improves compilation time and memory usage at runtime by avoiding heap
  allocations and reducing the amount of instructions.
  [(#1075)](https://github.com/PennyLaneAI/catalyst/pull/1075)

  A new MLIR pass `detensorize-scf` is added that works in conjuction with the existing
  `linalg-detensorize` pass to detensorize input programs. The IR generated by JAX wraps all values
  in the program in tensors, including scalars, leading to unnecessary memory allocations for
  programs compiled to CPU via MLIR to LLVM pipeline.

* Bufferization of `gradient.ForwardOp` and `gradient.ReverseOp` now requires 3 steps:
  `gradient-preprocessing`, `gradient-bufferize`, and `gradient-postprocessing`.
  `gradient-bufferize` has a new rewrite for `gradient.ReturnOp`.
  [(#1139)](https://github.com/PennyLaneAI/catalyst/pull/1139)

* The decorator `self_inverses` now supports all Hermitian Gates.
  [(#1136)](https://github.com/PennyLaneAI/catalyst/pull/1136)

  The full list of supported gates are as follows:

  One-bit Gates:
  - [`qml.Hadamard`](https://docs.pennylane.ai/en/stable/code/api/pennylane.Hadamard.html)
  - [`qml.PauliX`](https://docs.pennylane.ai/en/stable/code/api/pennylane.PauliX.html)
  - [`qml.PauliY`](https://docs.pennylane.ai/en/stable/code/api/pennylane.PauliY.html)
  - [`qml.PauliZ`](https://docs.pennylane.ai/en/stable/code/api/pennylane.PauliZ.html)

  Two-bit Gates:
  - [`qml.CNOT`](https://docs.pennylane.ai/en/stable/code/api/pennylane.CNOT.html)
  - [`qml.CY`](https://docs.pennylane.ai/en/stable/code/api/pennylane.CY.html)
  - [`qml.CZ`](https://docs.pennylane.ai/en/stable/code/api/pennylane.CZ.html)
  - [`qml.SWAP`](https://docs.pennylane.ai/en/stable/code/api/pennylane.SWAP.html)

  Three-bit Gates: Toffoli
  - [`qml.Toffoli`](https://docs.pennylane.ai/en/stable/code/api/pennylane.Toffoli.html)

* Support is expanded for backend devices that exculsively return samples in the measurement
  basis. Pre- and post-processing now allows `qjit` to be used on these devices with `qml.expval`,
  `qml.var` and `qml.probs` measurements in addiiton to `qml.sample`, using the
  `measurements_from_samples` transform.
  [(#1106)](https://github.com/PennyLaneAI/catalyst/pull/1106)

* Catalyst now supports numpy 2.0
  [(#1119)](https://github.com/PennyLaneAI/catalyst/pull/1119)

* Importing Catalyst will now pollute less of JAX's global variables by using `LoweringParameters`.
  [(#1152)](https://github.com/PennyLaneAI/catalyst/pull/1152)

* Compiling `qnode`s to asynchronous functions will no longer print to stderr in case of an error.
  [(#645)](https://github.com/PennyLaneAI/catalyst/pull/645)

* Cached primitive lowerings is used instead of a custom cache structure.
  [(#1159)](https://github.com/PennyLaneAI/catalyst/pull/1159)

<h3>Breaking changes</h3>

* Remove `static_size` field from `AbstractQreg` class.
  [(#1113)](https://github.com/PennyLaneAI/catalyst/pull/1113)

  This reverts a previous breaking change.

<h3>Bug fixes</h3>

* Resolve a bug in the `vmap` function when passing shapeless values to the target.
  [(#1150)](https://github.com/PennyLaneAI/catalyst/pull/1150)

* Fix error message displayed when using `qml.cond` on callables with arguments.
  [(#1151)](https://github.com/PennyLaneAI/catalyst/pull/1151)

* Fixes taking gradient of nested accelerate callbacks.
  [(#1156)](https://github.com/PennyLaneAI/catalyst/pull/1156)

<h3>Internal changes</h3>

* Update Enzyme to version `v0.0.149`.
  [(#1142)](https://github.com/PennyLaneAI/catalyst/pull/1142)

* Remove the `MemMemCpyOptPass` in llvm O2 (applied for Enzyme), this reduces bugs when
  running gradient like functions.
  [(#1063)](https://github.com/PennyLaneAI/catalyst/pull/1063)

* Functions with multiple tapes are now split with a new mlir pass `--split-multiple-tapes`, with
  one tape per function. The reset routine that makes a maeasurement between tapes and inserts a
  X gate if measured one is no longer used.
  [(#1017)](https://github.com/PennyLaneAI/catalyst/pull/1017)
  [(#1130)](https://github.com/PennyLaneAI/catalyst/pull/1130)

* Prefer creating new `qml.devices.ExecutionConfig` objects over using the global
  `qml.devices.DefaultExecutionConfig`. Doing so helps avoid unexpected bugs and test failures in
  case the `DefaultExecutionConfig` object becomes modified from its original state.
  [(#1137)](https://github.com/PennyLaneAI/catalyst/pull/1137)

* Remove the old `QJITDevice` API.
  [(#1138)](https://github.com/PennyLaneAI/catalyst/pull/1138)

* The device capability loading mechanism has been moved into the `QJITDevice` constructor.
  [(#1141)](https://github.com/PennyLaneAI/catalyst/pull/1141)

* Several functions related to device capabilities have been refactored.
  [(#1149)](https://github.com/PennyLaneAI/catalyst/pull/1149)

  In particular, the signatures of `get_device_capability`, `catalyst_decompose`,
 `catalyst_acceptance`, and `QJITDevice.__init__` have changed, and the `pennylane_operation_set`
  function has been removed entirely.

<h3>Contributors</h3>

This release contains contributions from (in alphabetical order):

Joey Carter,
Spencer Comin,
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
