# Release 0.9.0-dev

<h3>New features</h3>

* Shot-vector support for Catalyst: Introduces support for shot-vectors in Catalyst, currently available for `qml.sample` measurements in the `lightning.qubit` device. Shot-vectors now allow elements of the form `((20, 5),)`, which is equivalent to `(20,)*5` or `(20, 20, 20, 20, 20)`. Furthermore, multiple `qml.sample` calls can now be returned from the same program, and can be structured using Python containers. For example, a program can return a dictionary like `return {"first": qml.sample(), "second": qml.sample()}`.[(#1051)](https://github.com/PennyLaneAI/catalyst/pull/1051)

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

* Fixes an issue where certain JAX linear algebra functions from `jax.scipy.linalg` gave incorrect
  results when invoked from within a qjit block, and adds full support for other `jax.scipy.linalg`
  functions.
  [(#1097)](https://github.com/PennyLaneAI/catalyst/pull/1097)

  The supported linear algebra functions include, but are not limited to:

  - [`jax.scipy.linalg.cholesky`](https://jax.readthedocs.io/en/latest/_autosummary/jax.scipy.linalg.cholesky.html)
  - [`jax.scipy.linalg.expm`](https://jax.readthedocs.io/en/latest/_autosummary/jax.scipy.linalg.expm.html)
  - [`jax.scipy.linalg.funm`](https://jax.readthedocs.io/en/latest/_autosummary/jax.scipy.linalg.funm.html)
  - [`jax.scipy.linalg.hessenberg`](https://jax.readthedocs.io/en/latest/_autosummary/jax.scipy.linalg.hessenberg.html)
  - [`jax.scipy.linalg.lu`](https://jax.readthedocs.io/en/latest/_autosummary/jax.scipy.linalg.lu.html)
  - [`jax.scipy.linalg.lu_solve`](https://jax.readthedocs.io/en/latest/_autosummary/jax.scipy.linalg.lu_solve.html)
  - [`jax.scipy.linalg.polar`](https://jax.readthedocs.io/en/latest/_autosummary/jax.scipy.linalg.polar.html)
  - [`jax.scipy.linalg.qr`](https://jax.readthedocs.io/en/latest/_autosummary/jax.scipy.linalg.qr.html)
  - [`jax.scipy.linalg.schur`](https://jax.readthedocs.io/en/latest/_autosummary/jax.scipy.linalg.schur.html)
  - [`jax.scipy.linalg.solve`](https://jax.readthedocs.io/en/latest/_autosummary/jax.scipy.linalg.solve.html)
  - [`jax.scipy.linalg.sqrtm`](https://jax.readthedocs.io/en/latest/_autosummary/jax.scipy.linalg.sqrtm.html)
  - [`jax.scipy.linalg.svd`](https://jax.readthedocs.io/en/latest/_autosummary/jax.scipy.linalg.svd.html)

<h3>Breaking changes</h3>

<h3>Bug fixes</h3>

* Those functions calling the `gather_p` primitive (like `jax.scipy.linalg.expm`)
  can now be used in multiple qjits in a single program.
  [(#1096)](https://github.com/PennyLaneAI/catalyst/pull/1096)

<h3>Internal changes</h3>

* Remove the `MemMemCpyOptPass` in llvm O2 (applied for Enzyme), this reduces bugs when 
  running gradient like functions.
  
  [(#1063)](https://github.com/PennyLaneAI/catalyst/pull/1063)

<h3>Contributors</h3>

This release contains contributions from (in alphabetical order):

Joey Carter,
Romain Moyard,
Paul Haochen Wang,
Sengthai Heng,