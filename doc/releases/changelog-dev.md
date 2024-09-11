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

* Zero-Noise Extrapolation (ZNE) local folding: Introduces the option to fold gates locally as well as the existing method of globally. Global folding (as in previous versions) applies the scale factor by forming the inverse of the entire quantum circuit (without measurements) and repeating the circuit with its inverse; local folding inserts per-gate folding sequences directly in place of each gate in the original circuit instead of applying the scale factor to the entire circuit at once. [(#1006)](https://github.com/PennyLaneAI/catalyst/pull/1006)

  For example,

  ```python
  import jax
  import pennylane as qml
  from catalyst import qjit, mitigate_with_zne
  from pennylane.transforms import exponential_extrapolate

  dev = qml.device("lightning.qubit", wires=4, shots=5)

  @qml.qnode(dev)
  def circuit():
    qml.Hadamard(wires=0)
    qml.CNOT(wires=[0, 1])
    return qml.expval(qml.PauliY(wires=0))

  @qjit(keep_intermediate=True)
  def mitigated_circuit():
    s = jax.numpy.array([1, 2, 3])
    return mitigate_with_zne(
      circuit,
      scale_factors=s,
      extrapolate=exponential_extrapolate,
      folding="all" # "all" for local or "global" for the original method (default being "global")
    )()
  ```

  ```pycon
  >>> circuit()
  >>> mitigated_circuit()
  ```

<h3>Improvements</h3>

* Support is expanded for backend devices that exculsively return samples in the measurement 
  basis. Pre- and post-processing now allows `qjit` to be used on these devices with `qml.expval`, 
  `qml.var` and `qml.probs` measurements in addiiton to `qml.sample`, using the `measurements_from_samples` transform.
  [(#1106)](https://github.com/PennyLaneAI/catalyst/pull/1106)

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

* Remove `static_size` field from `AbstractQreg` class.
  [(#1113)](https://github.com/PennyLaneAI/catalyst/pull/1113)

  This reverts a previous breaking change.

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
Lillian M.A. Frederiksen,
Romain Moyard,
Erick Ochoa Lopez,
Paul Haochen Wang,
Sengthai Heng,
Daniel Strano
