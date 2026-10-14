# Release 0.14.2 (current release)

<h3>Bug fixes</h3>

* Fixed a compilation error, `cannot handle (reverse) unknown intrinsic llvm.atan2`, when
  differentiating programs that use `atan2` (e.g. `jnp.arctan2` or `qml.AmplitudeEmbedding`).
  Upstream Enzyme support for the `llvm.atan2` intrinsic
  ([EnzymeAD/Enzyme#3222](https://github.com/EnzymeAD/Enzyme/pull/3222)) is backported as a
  source patch on top of the pinned Enzyme version, without changing the pinned LLVM or Enzyme
  versions.
  [#3325](https://github.com/PennyLaneAI/catalyst/issues/3325)

<h3>Contributors</h3>

This release contains contributions from (in alphabetical order):

David Ittah.
