# Release 0.13.0 (development release)

<h3>New features since last release</h3>

<h3>Improvements 🛠</h3>

<h3>Breaking changes 💔</h3>

<h3>Deprecations 👋</h3>

<h3>Bug fixes 🐛</h3>

* Fix errors in AutoGraph transformed functions when `qml.prod` is used together with other operator
  transforms (e.g. `qml.adjoint`).
  [(#1910)](https://github.com/PennyLaneAI/catalyst/pull/1910)

<h3>Internal changes ⚙️</h3>

* `from_plxpr` now supports adjoint and ctrl operations and transforms,
  `Hermitian` observables, `for_loop` outside qnodes, and `while_loop` outside QNode's.
  [(#1844)](https://github.com/PennyLaneAI/catalyst/pull/1844)
  [(#1850)](https://github.com/PennyLaneAI/catalyst/pull/1850)
  [(#1903)](https://github.com/PennyLaneAI/catalyst/pull/1903)
  [(#1896)](https://github.com/PennyLaneAI/catalyst/pull/1896)


<h3>Documentation 📝</h3>

<h3>Contributors ✍️</h3>

This release contains contributions from (in alphabetical order):

David Ittah,
Christina Lee
