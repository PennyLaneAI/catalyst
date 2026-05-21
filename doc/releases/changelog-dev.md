# Release 0.16.0 (development release)

<h3>New features since last release</h3>

* A new, experimental compiler pass `convert-qecp-to-quantum` has been added to lower operations
  from the QEC Physical (`qecp`) dialect into the Quantum (`quantum`) dialect.
  [(#2822)](https://github.com/PennyLaneAI/catalyst/pull/2822)
  [(#2809)](https://github.com/PennyLaneAI/catalyst/pull/2809)
  [(#2824)](https://github.com/PennyLaneAI/catalyst/pull/2824)
  [(#2835)](https://github.com/PennyLaneAI/catalyst/pull/2835)
  [(#2839)](https://github.com/PennyLaneAI/catalyst/pull/2839)
  [(#2849)](https://github.com/PennyLaneAI/catalyst/pull/2849)


<h3>Improvements 🛠</h3>

* The `--decompose-lowering` pass can now handle decomposition rule functions whose quantum register
  argument is at an arbitrary position in the argument list.
  [(#2836)](https://github.com/PennyLaneAI/catalyst/pull/2836)

* The `--decompose-lowering` pass can now handle null decomposition rules, which are rule functions
  that do not have any quantum values as arguments or results. Gates with null decomposition rules
  are simply removed.
  [(#2855)](https://github.com/PennyLaneAI/catalyst/pull/2855)

<h3>Breaking changes 💔</h3>

* Catalyst's xDSL dependencies have been updated to `xdsl` 0.63.0 and `xdsl-jax` 0.5.2.
  [(#2840)](https://github.com/PennyLaneAI/catalyst/pull/2840)

* Removes support for `Transform.plxpr_transform` from the `qp.qjit(capture=True)` capture pipeline.
  All transforms must now have a MLIR or XDSL implementation and a corresponding `pass_name`.

<h3>Deprecations 👋</h3>

<h3>Bug fixes 🐛</h3>

* Fixed a bug where using `keep_intermediate=True` with `target="mlir"` resulted in an empty workspace
  folder being created and the files printed outside in the main directory.
  [(#2807)](https://github.com/PennyLaneAI/catalyst/pull/2807)

<h3>Internal changes ⚙️</h3>

* Fixed ``KeyError`` in autograph when using ``qp.prod`` as a decorator with PennyLane >= 0.45.
  [(#2844)](https://github.com/PennyLaneAI/catalyst/pull/2844)

* Update RC nightly builds to read version number from the `_version.py` file
  [(#2797)](https://github.com/PennyLaneAI/catalyst/pull/2797)

* Fix build failures when using clang with GCC ≤ 13 libstdc++ by replacing
  `std::views::filter`/`std::views::transform` with `std::copy_if`/`std::transform`
  [(#2801)](https://github.com/PennyLaneAI/catalyst/pull/2801)

* The experimental compiler pass `convert-qecl-to-qecp` has been extended to lower
  transversal gate operations from the QEC Logical (`qecl`) dialect into the QEC
  Physical (`qecp`) dialect.
  [(#2776)](https://github.com/PennyLaneAI/catalyst/pull/2776)

* The reference semantics Pauli Product Measurement operation `pbc.ref.ppm` was added.
  [(#2773)](https://github.com/PennyLaneAI/catalyst/pull/2773)

* Part of the new, experimental QEC pipeline, the `convert-qecp-to-llvm` compiler pass has been
  added to lower operations and types in the QEC physical dialect to the LLVM dialect.
  [(#2780)](https://github.com/PennyLaneAI/catalyst/pull/2780)
  [(#2772)](https://github.com/PennyLaneAI/catalyst/pull/2772)

* The strategy to decode physical measurements in the `convert-qecl-to-qecp` pass has been updated
  to perform the decoding directly in the IR rather than offloading to a pre-compiled runtime
  function.
  [(#2813)](https://github.com/PennyLaneAI/catalyst/pull/2813)

* Resolved a bug in the QEC-cycle subroutine within the `convert-qecl-to-qecp` pass where the SSA
  values of the `scf.yield` op were incorrectly returned instead of the `scf.for` op results. Also,
  the `qec_code` pass option is now given as a `str` rather than a `QecCode` object to ensure
  compatibility with Catalyst's compiler infrastructure.
  [(#2837)](https://github.com/PennyLaneAI/catalyst/pull/2837)

* The constructors of xDSL ops that accept index attributes have been updated to ensure that the
  resulting attribute has the correct type. These ops include `quantum.{extract, insert}`,
  `qecl.{extract_block, insert_block, measure, <gates>}`, and
  `qecp.{extract_block, insert_block, extract, insert}`.
  [(#2846)](https://github.com/PennyLaneAI/catalyst/pull/2846)

* A new, experimental compiler pipeline `qec_pipeline` has been added to the `ftqc.pipelines` module.
  [(#2852)](https://github.com/PennyLaneAI/catalyst/pull/2852)

* The reference semantics MBQC operations have been moved from the `qref` dialect to the `mbqc`
  dialect. They are now accessible as `mbqc.ref.measure_in_basis` and `mbqc.ref.graph_state_prep`.
  [(#2829)](https://github.com/PennyLaneAI/catalyst/pull/2829)

<h3>Documentation 📝</h3>

<h3>Contributors ✍️</h3>

This release contains contributions from (in alphabetical order):

Joey Carter,
Yushao Chen,
Lillian Frederiksen,
Christina Lee,
Mehrdad Malekmohammadi,
Shuli Shu,
Paul Haochen Wang.
