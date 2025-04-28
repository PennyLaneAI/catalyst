# Release 0.12.0 (development release)

<h3>New features since last release</h3>

* A new compilation pass has been added to Catalyst to decompose non-Clifford Pauli product rotations (PPRs) into Pauli product measurements (PPMs) that aim to consume magic states. Two decomposition strategies are supported: *inject magic state* and *auto-corrected :math:`\pi/8` rotation method* as described in [arXiv:1808.02892](https://arxiv.org/abs/1808.02892v3).
  [(#1664)](https://github.com/PennyLaneAI/catalyst/pull/1664)

  The new compilation pass can be accessed from the :mod:`~.passes` module or in :func:`~.pipeline`:
  * :func:`catalyst.passes.ppr_to_ppm <~.passes.ppr_to_ppm>`: Decomposes non-Clifford PPRs into PPMs, consuming magic states using the auto-corrected :math:`\pi/8` rotation method by default.

  The following new and extended operations in intermediate representation are available:
  * :func:`qec.select.ppm <~.ops.qec.select.ppm>`: Performs a conditional PPM based on a boolean control.
  * :func:`qec.prepare <~.ops.qec.prepare>`: Prepares logical qubits in specified initial states.
  * :func:`qec.ppr <~.ops.qec.ppr>` and :func:`qec.ppm <~.ops.qec.ppm>`: Now support conditional execution.

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

* The `ppr_to_ppm` pass has been renamed to `merge_ppr_ppm` (same functionality). A new `ppr_to_ppm` will handle direct decomposition of PPRs into PPMs.
  [(#1688)](https://github.com/PennyLaneAI/catalyst/pull/1688)

<h3>Deprecations 👋</h3>

<h3>Bug fixes 🐛</h3>

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
  [(#1679)](https://github.com/PennyLaneAI/catalyst/pull/1679)

* The utility function `EnsureFunctionDeclaration` is refactored into the `Utils` of the `Catalyst` dialect, instead of being duplicated in each individual dialect.
  [(#1683)](https://github.com/PennyLaneAI/catalyst/pull/1683)

<h3>Documentation 📝</h3>

<h3>Contributors ✍️</h3>

This release contains contributions from (in alphabetical order):

Joey Carter,
Sengthai Heng,
David Ittah,
Christina Lee,
Erick Ochoa Lopez,
Paul Haochen Wang.
