# Release 0.14.0 (development release)

<h3>New features since last release</h3>

<h3>Improvements 🛠</h3>

<h3>Breaking changes 💔</h3>

<h3>Deprecations 👋</h3>

<h3>Bug fixes 🐛</h3>

* Fixes the translation of plxpr control flow for edge cases where the `consts` were being
  reordered.
  [(#2128)](https://github.com/PennyLaneAI/catalyst/pull/2128)
  [(#2133)](https://github.com/PennyLaneAI/catalyst/pull/2133)

<h3>Internal changes ⚙️</h3>

* Several MLIR passes (`to-ppr`, `commute-ppr`, `merge-ppr-ppm`, `pprm-to-mbqc` and `reduce-t-depth`)
  are mapped to corresponding primitive names in PLxPR so that they can be included in programs generated
  with `qml.capture` enabled.
  [(#2139)](https://github.com/PennyLaneAI/catalyst/pull/2139)

<h3>Documentation 📝</h3>

<h3>Contributors ✍️</h3>

This release contains contributions from (in alphabetical order):

Lillian Frederiksen,
Christina Lee,
Paul Haochen Wang.
