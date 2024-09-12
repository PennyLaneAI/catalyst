# Release 0.9.0-dev

<h3>New features</h3>

* Shot-vector support for Catalyst: Introduces support for shot-vectors in Catalyst, currently available for `qml.sample` measurements in the `lightning.qubit` device. Shot-vectors now allow elements of the form `((20, 5),)`, which is equivalent to `(20,)*5` or `(20, 20, 20, 20, 20)`. Furthermore, multiple `qml.sample` calls can now be returned from the same program, and can be structured using Python containers. For example, a program can return a dictionary like `return {"first": qml.sample(), "second": qml.sample()}`.
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

* Treat `qml.devices.DefaultExecutionConfig` as if it were immutable and make copies of it when
  modifications are required. Doing so helps avoid unexpected bugs and test failures when the
  `DefaultExecutionConfig` object becomes modified from its original state.
  [(#1137)](https://github.com/PennyLaneAI/catalyst/pull/1137)

<h3>Contributors</h3>

This release contains contributions from (in alphabetical order):

Joey Carter,
Lillian M.A. Frederiksen,
Romain Moyard,
Erick Ochoa Lopez,
Paul Haochen Wang,
Sengthai Heng,
Daniel Strano.
