# Release 0.12.0 (development release)

<h3>New features since last release</h3>

<h3>Improvements 🛠</h3>

* The behaviour of measurement processes executed on `null.qubit` with QJIT is now more in line with
  their behaviour on `null.qubit` *without* QJIT.
  [(#1598)](https://github.com/PennyLaneAI/catalyst/pull/1598)

  Previously, measurement processes like `qml.sample()`, `qml.counts()`, `qml.probs()`, etc.
  returned values from uninitialized memory when executed on `null.qubit` with QJIT. This change
  ensures that measurement processes on `null.qubit` always return the value 0 or the result
  corresponding to the '0' state, depending on the context.

<h3>Breaking changes 💔</h3>

* Catalyst has removed the `experimental_capture` keyword from its `qjit` function in favor of a
  unified program capture behaviour. 
  [(#1657)](https://github.com/PennyLaneAI/catalyst/pull/1657)

  Program capture has to be enabled before the definition of the function to be qjitted.
  
  For AOT compilation, program capture can be disabled right after the qjit usage and before execution.
  
  ```python
  import pennylane as qml
  from catalyst import qjit

  dev = qml.device("lightning.qubit", wires=1)

  qml.capture.enable()

  @qjit()
  @qml.qnode(dev)
  def circuit(x: float):
      qml.Hadamard(0)
      qml.CNOT([0, 1])
      return qml.expval(qml.Z(0))

  qml.capture.disable()

  circuit(0.1)
  ```

  But for JIT compilation, program capture cannot be disabled before execution,
  otherwise the capture will not take place:

  ```python
  import pennylane as qml
  from catalyst import qjit

  dev = qml.device("lightning.qubit", wires=1)

  qml.capture.enable()

  @qjit()
  @qml.qnode(dev)
  def circuit(x):
      qml.Hadamard(0)
      qml.CNOT([0, 1])
      return qml.expval(qml.Z(0))

  circuit(0.1)

  qml.capture.disable()
  ```

<h3>Deprecations 👋</h3>

<h3>Bug fixes 🐛</h3>

* Catalyst now correctly supports `qml.StatePrep()` and `qml.BasisState()` operations in the
  experimental PennyLane program-capture pipeline.
  [(#1631)](https://github.com/PennyLaneAI/catalyst/pull/1631)

<h3>Internal changes ⚙️</h3>

<h3>Documentation 📝</h3>

<h3>Contributors ✍️</h3>

This release contains contributions from (in alphabetical order):

Joey Carter,
Erick Ochoa Lopez.
