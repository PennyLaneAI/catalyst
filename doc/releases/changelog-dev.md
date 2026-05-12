# Release 0.16.0 (development release)

<h3>New features since last release</h3>

* A new, experimental compiler pass `convert-qecp-to-quantum` has been added to lower operations
  from the QEC Physical (`qecp`) dialect into the Quantum (`quantum`) dialect.
  [(#2822)](https://github.com/PennyLaneAI/catalyst/pull/2822)

<h3>Improvements 🛠</h3>

<h3>Breaking changes 💔</h3>

<h3>Deprecations 👋</h3>

<h3>Bug fixes 🐛</h3>

<h3>Internal changes ⚙️</h3>

* Added a generic memory-space attribute in the Catalyst dialect that can be used in the
  `memory_space` slot of `memref` types (or as a discardable attribute) to tag values with
  a logical memory domain plus an optional LLVM-level address space within that domain.
  [(#2750)](https://github.com/PennyLaneAI/catalyst/pull/2750)

* Update RC nightly builds to read version number from the `_version.py` file 
  [(#2797)](https://github.com/PennyLaneAI/catalyst/pull/2797)

* The experimental compiler pass `convert-qecl-to-qecp` has been extended to lower 
  transversal gate operations from the QEC Logical (`qecl`) dialect into the QEC 
  Physical (`qecp`) dialect.
  [(#2776)](https://github.com/PennyLaneAI/catalyst/pull/2776)

<h3>Documentation 📝</h3>

<h3>Contributors ✍️</h3>

This release contains contributions from (in alphabetical order):

Lillian Frederiksen,
Shuli Shu,
