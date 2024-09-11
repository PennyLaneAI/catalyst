# Release 0.8.1

<h3>New features</h3>

* Zero-Noise Extrapolation (ZNE) local folding: Introduces the option to fold gates locally as well
  as the existing method of globally. Global folding (as in previous versions) applies the scale
  factor by forming the inverse of the entire quantum circuit (without measurements) and repeating
  the circuit with its inverse; local folding inserts per-gate folding sequences directly in place
  of each gate in the original circuit instead of applying the scale factor to the entire circuit
  at once. 
  [(#1006)](https://github.com/PennyLaneAI/catalyst/pull/1006)
  [(#1129)](https://github.com/PennyLaneAI/catalyst/pull/1129)

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

<h3>Breaking changes</h3>

* The argument `scale_factors` of `mitigate_with_zne` function now follows the proper literature
  definition. It now needs to be a list of positive odd integers, as we don't support the fractional
  part.
  [(#1120)](https://github.com/PennyLaneAI/catalyst/pull/1120)

<h3>Contributors</h3>

This release contains contributions from (in alphabetical order):

Joey Carter,
David Ittah,
Romain Moyard,
Daniel Strano,
Raul Torres.
