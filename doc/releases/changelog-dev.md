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

* (Device Developers Only) The `QuantumDevice` interface in the Catalyst Runtime plugin system
  has been modified, which requires recompiling plugins for binary compatibility.
  [(#1680)](https://github.com/PennyLaneAI/catalyst/pull/1680)

  As announced in the [0.10.0 release](https://docs.pennylane.ai/projects/catalyst/en/stable/dev/release_notes.html#release-0-10-0),
  the `shots` argument has been removed from the `Sample` and `Counts` methods in the interface,
  since it unnecessarily duplicated this information. Additionally, `shots` will no longer be
  supplied by Catalyst through the `kwargs` parameter of the device constructor. The shot value must
  now be obtained through the `SetDeviceShots` method.

  Further, the documentation for the interface has been overhauled and now describes the
  expected behaviour of each method in detail. A quality of life improvement is that optional
  methods are now clearly marked as such and also come with a default implementation in the base
  class, so device plugins need only override the methods they wish to support.

  Finally, the `PrintState` and the `One`/`Zero` utility functions have been removed, since they
  did not serve a convincing purpose.

* Catalyst has removed the `experimental_capture` keyword from the `qjit` decorator in favour of
  unified behaviour with PennyLane.
  [(#1657)](https://github.com/PennyLaneAI/catalyst/pull/1657)

  Instead of enabling program capture with Catalyst via `qjit(experimental_capture=True)`, program
  capture can be enabled via the global toggle `qml.capture.enable()`:

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

* Fix AutoGraph fallback for valid iteration targets with constant data but no length, for example
  `itertools.product(range(2), repeat=2)`.
  [(#1665)](https://github.com/PennyLaneAI/catalyst/pull/1665)

* Catalyst now correctly supports `qml.StatePrep()` and `qml.BasisState()` operations in the
  experimental PennyLane program-capture pipeline.
  [(#1631)](https://github.com/PennyLaneAI/catalyst/pull/1631)

<h3>Internal changes ⚙️</h3>

* Stop overriding the `num_wires` property when the operator can exist on `AnyWires`. This allows
  the deprecation of `WiresEnum` in pennylane.
  [(#1667)](https://github.com/PennyLaneAI/catalyst/pull/1667)

* Catalyst now includes an experimental `mbqc` dialect for representing measurement-based
  quantum-computing protocols in MLIR.
  [(#1663)](https://github.com/PennyLaneAI/catalyst/pull/1663)

<h3>Documentation 📝</h3>

<h3>Contributors ✍️</h3>

This release contains contributions from (in alphabetical order):

Joey Carter,
David Ittah,
Christina Lee,
Erick Ochoa Lopez.
