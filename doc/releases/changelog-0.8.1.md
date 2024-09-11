# Release 0.8.1

<h3>New features</h3>

* The `catalyst.mitigate_with_zne` error mitigation compilation pass now supports
  the option to fold gates locally as well as the existing method of globally.
  [(#1006)](https://github.com/PennyLaneAI/catalyst/pull/1006)
  [(#1129)](https://github.com/PennyLaneAI/catalyst/pull/1129)

  While global folding applies the scale factor by forming the inverse of the
  entire quantum circuit (without measurements) and repeating
  the circuit with its inverse, local folding instead inserts per-gate folding sequences directly in place
  of each gate in the original circuit.

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
      folding="local-all" # "local-all" for local on all gates or "global" for the original method (default being "global")
    )()
  ```

  ```pycon
  >>> circuit()
  >>> mitigated_circuit()
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

* The argument `scale_factors` of `mitigate_with_zne` function now follows the proper literature
  definition. It now needs to be a list of positive odd integers, as we don't support the fractional
  part.
  [(#1120)](https://github.com/PennyLaneAI/catalyst/pull/1120)

<h3>Contributors</h3>

This release contains contributions from (in alphabetical order):

Joey Carter,
Alessandro Cosentino,
David Ittah,
Romain Moyard,
Daniel Strano,
Raul Torres.
