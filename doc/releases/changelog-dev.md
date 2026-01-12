# Release 0.15.0 (development release)

<h3>New features since last release</h3>

* The tree-traversal MCM method (accessed with `mcm_method="tree-traversal"` when creating a QNode) is now
  compatible with Catalyst. This MCM method works in analytic mode and with finite-shots, providing great scaling
  in _both_ memory consumption and execution time.
  [(#2384)](https://github.com/PennyLaneAI/catalyst/pull/2384)

  When using `mcm_method="tree-traversal"` within a qjit'd function, there are limitations to consider:

  * Postselecting and resetting qubits is not supported when performing an MCM with :func:`~.measure <qml.measure>` (i.e., the `postselect` and `reset` arguments).
  * Statevector-based terminal measurements are not supported (e.g., :func:`~.probs` and :func:`~.state`).
  * Multiple terminal measurements are not supported (e.g., ``return qml.expval(Z(0)), qml.expval(Y(1), qml.expval(X(2)))``).
  * Terminal measurements acting on MCM values are not supported (e.g., ``return qml.expval(m1)``, where ``m1`` is the result of an MCM with ``qml.measure``).
  * For loops with a dynamic range are not supported.

<h3>Improvements 🛠</h3>

<h3>Breaking changes 💔</h3>

<h3>Deprecations 👋</h3>

<h3>Bug fixes 🐛</h3>

<h3>Internal changes ⚙️</h3>

<h3>Documentation 📝</h3>

<h3>Contributors ✍️</h3>

This release contains contributions from (in alphabetical order):

Luis Alfredo Nuñez Meneses,
Hongsheng Zheng
