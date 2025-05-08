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

* Catalyst has removed the `experimental_capture` keyword from the `qjit` decorator in favour of
  unified behaviour with PennyLane.
  [(#1657)](https://github.com/PennyLaneAI/catalyst/pull/1657)

  Instead of enabling program capture with Catalyst via `qjit(experimental_capture=True)`, program capture
  can be enabled via the global toggle `qml.capture.enable()`:

  ```python
  import pennylane as qml
  from catalyst import qjit

  dev = qml.device("lightning.qubit", wires=2)

  qml.capture.enable()

  @qjit
  @qml.qnode(dev)
  def circuit(x):
      qml.Hadamard(0)
      qml.CNOT([0, 1])
      return qml.expval(qml.Z(0))

  circuit(0.1)
  ```

  Disabling program capture can be done with `qml.capture.disable()`.

<h3>Deprecations 👋</h3>

<h3>Bug fixes 🐛</h3>

* Fix Boolean arguments/results not working with the debugging functions `debug.get_cmain` and
  `debug.compile_executable`.
  [(#1687)](https://github.com/PennyLaneAI/catalyst/pull/1687)

* Fix AutoGraph fallback for valid iteration targets with constant data but no length, for example
  `itertools.product(range(2), repeat=2)`.
  [(#1665)](https://github.com/PennyLaneAI/catalyst/pull/1665)

* Catalyst now correctly supports `qml.StatePrep()` and `qml.BasisState()` operations in the
  experimental PennyLane program-capture pipeline.
  [(#1631)](https://github.com/PennyLaneAI/catalyst/pull/1631)

<h3>Internal changes ⚙️</h3>

* Stop overriding the `num_wires` property when the operator can exist on `AnyWires`. This allows the deprecation
  of `WiresEnum` in pennylane.
  [(#1667)](https://github.com/PennyLaneAI/catalyst/pull/1667)

* Catalyst now includes an experimental `mbqc` dialect for representing measurement-based
  quantum-computing protocols in MLIR.
  [(#1663)](https://github.com/PennyLaneAI/catalyst/pull/1663)

* The utility function `EnsureFunctionDeclaration` is refactored into the `Utils` of the `Catalyst` dialect, instead of being duplicated in each individual dialect.
  [(#1683)](https://github.com/PennyLaneAI/catalyst/pull/1683)

<h3>Documentation 📝</h3>

<h3>Contributors ✍️</h3>

This release contains contributions from (in alphabetical order):

Joey Carter,
David Ittah,
Christina Lee,
Erick Ochoa Lopez,
Paul Haochen Wang.
