# Release 0.16.0 (development release)

<h3>New features since last release</h3>

* A new, experimental compiler pass `convert-qecp-to-quantum` has been added to lower operations
  from the QEC Physical (`qecp`) dialect into the Quantum (`quantum`) dialect.
  [(#2822)](https://github.com/PennyLaneAI/catalyst/pull/2822)

<h3>Improvements 🛠</h3>

* A new `catalyst.debug.compile_mlir` function has been added, allowing standalone MLIR files to be
  compiled through the full Catalyst pipeline and returned as a callable Python object.
  [(#2832)](https://github.com/PennyLaneAI/catalyst/pull/2832)

<h3>Breaking changes 💔</h3>

<h3>Deprecations 👋</h3>

<h3>Bug fixes 🐛</h3>

* Fixed a bug where using `keep_intermediate=True` with `target="mlir"` resulted in an empty workspace
  folder being created and the files printed outside in the main directory.
  [(#2807)](https://github.com/PennyLaneAI/catalyst/pull/2807)

<h3>Internal changes ⚙️</h3>

* Update RC nightly builds to read version number from the `_version.py` file 
  [(#2797)](https://github.com/PennyLaneAI/catalyst/pull/2797)

* Fix build failures when using clang with GCC ≤ 13 libstdc++ by replacing
  `std::views::filter`/`std::views::transform` with `std::copy_if`/`std::transform`
  [(#2801)](https://github.com/PennyLaneAI/catalyst/pull/2801)

* The experimental compiler pass `convert-qecl-to-qecp` has been extended to lower 
  transversal gate operations from the QEC Logical (`qecl`) dialect into the QEC 
  Physical (`qecp`) dialect.
  [(#2776)](https://github.com/PennyLaneAI/catalyst/pull/2776)

<h3>Documentation 📝</h3>

<h3>Contributors ✍️</h3>

This release contains contributions from (in alphabetical order):

Lillian Frederiksen,
Mehrdad Malekmohammadi,
Shuli Shu,
