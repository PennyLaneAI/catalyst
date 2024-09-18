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

* A new function `catalyst.passes.pipeline` allows the quantum circuit transformation pass pipeline for QNodes within a qjit-compiled workflow to be configured.
  [(#1131)](https://github.com/PennyLaneAI/catalyst/pull/1131)

  ```python
    my_pass_pipeline = {
        "cancel_inverses": {},
        "merge_rotations": {},
    }
    dev = qml.device("lightning.qubit", wires=2)

    @pipeline(my_pass_pipeline)
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
        "merge_rotations": {},
    }

    no_rotation_pipeline = {"cancel_inverses": {}}

    @qjit
    def fn(x):
        circuit_pipeline = pipeline(my_pipeline)(circuit)
        circuit_no_rotation = pipeline(no_rotation_pipeline)(circuit)
        return jnp.abs(circuit_pipeline(x) - circuit_no_rotation(x))
  ```

  For a list of available passes, please see the [catalyst.passes module documentation](https://docs.pennylane.ai/projects/catalyst/en/stable/code/__init__.html#module-catalyst.passes).

  The pass pipeline order and options can be configured *globally* for a
  qjit-compiled function, by using the `circuit_transform_pipeline` argument of the :func:`~.qjit` decorator.

  ```python
    my_pass_pipeline = {
        "cancel_inverses": {},
        "merge_rotations": {},
    }

    @qjit(circuit_transform_pipeline=my_pass_pipeline)
    def fn(x):
        return jnp.sin(circuit(x ** 2))
  ```

  Global and local (via `@pipeline`) configurations can coexist, however local pass pipelines
  will always take precedence over global pass pipelines.

  Available MLIR passes are now documented and available within the
  [catalyst.passes module documentation](https://docs.pennylane.ai/projects/catalyst/en/stable/code/__init__.html#module-catalyst.passes).

<h3>Improvements</h3>

* Bufferization of `gradient.ForwardOp` and `gradient.ReverseOp` now requires 3 steps: `gradient-preprocessing`, 
  `gradient-bufferize`, and `gradient-postprocessing`. `gradient-bufferize` has a new rewrite for `gradient.ReturnOp`. 
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
  `qml.var` and `qml.probs` measurements in addiiton to `qml.sample`, using the `measurements_from_samples` transform.
  [(#1106)](https://github.com/PennyLaneAI/catalyst/pull/1106)

<h3>Breaking changes</h3>

* Remove `static_size` field from `AbstractQreg` class.
  [(#1113)](https://github.com/PennyLaneAI/catalyst/pull/1113)

  This reverts a previous breaking change.

<h3>Bug fixes</h3>

<h3>Internal changes</h3>

* Remove the `MemMemCpyOptPass` in llvm O2 (applied for Enzyme), this reduces bugs when 
  running gradient like functions.
  [(#1063)](https://github.com/PennyLaneAI/catalyst/pull/1063)

* Functions with multiple tapes are now split with a new mlir pass `--split-multiple-tapes`, with one tape per function. 
  The reset routine that makes a maeasurement between tapes and inserts a X gate if measured one is no longer used.
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

<h3>Contributors</h3>

This release contains contributions from (in alphabetical order):

Joey Carter,
Lillian M.A. Frederiksen,
Romain Moyard,
Erick Ochoa Lopez,
Paul Haochen Wang,
Sengthai Heng,
Daniel Strano,
Raul Torres.
