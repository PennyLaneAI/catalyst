# Release 0.8.2

<h3>New features</h3>

<h3>Improvements</h3>

* The function `mitigate_with_zne` is not restricted to be applied only on :class:`QNode`.
  It can be applied on a classical function containing quantum calls.
  [(#1169)](https://github.com/PennyLaneAI/catalyst/pull/1169)

<h3>Breaking changes</h3>


<h3>Bug fixes</h3>

* Resolve a bug in the `vmap` function when passing shapeless values to the target.
  [(#1150)](https://github.com/PennyLaneAI/catalyst/pull/1150)

* Resolve a bug where `mitigate_with_zne` does not work properly with shots and devices
  supporting only Counts and Samples (e.g. Qrack). (transform: `measurements_from_sample`).
  [(#1165)](https://github.com/PennyLaneAI/catalyst/pull/1165)

* Resolve a bug where `mitigate_with_zne` does not work properly with qnode using `mcm_method="one-shot"`,
  the transform is correctly applied.
  [(#1169)](https://github.com/PennyLaneAI/catalyst/pull/1169)

<h3>Contributors</h3>

This release contains contributions from (in alphabetical order):

David Ittah,
Romain Moyard,
Raul Torres.
