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

* Catalyst now supports numpy 2.0
  [(#1119)](https://github.com/PennyLaneAI/catalyst/pull/1119)

<h3>Breaking changes</h3>

* Remove `static_size` field from `AbstractQreg` class.
  [(#1113)](https://github.com/PennyLaneAI/catalyst/pull/1113)

  This reverts a previous breaking change.

<h3>Bug fixes</h3>

<h3>Internal changes</h3>

* Update Enzyme to version `v0.0.149`.
  [(#1142)](https://github.com/PennyLaneAI/catalyst/pull/1142)

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
Mehrdad Malekmohammadi,
Paul Haochen Wang,
Sengthai Heng,
Daniel Strano,
Raul Torres.
