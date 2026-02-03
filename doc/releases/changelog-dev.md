# Release 0.15.0 (development release)

<h3>New features since last release</h3>

<h3>Improvements 🛠</h3>

* The default mcm_method for the finite-shots setting (dynamic one-shot) no longer silently falls
  back to single-branch statistics in most cases. Instead, an error message is raised pointing out
  alternatives, like explicitly selecting single-branch statistics.
  [(#2398)](https://github.com/PennyLaneAI/catalyst/pull/2398)

  Importantly, single-branch statistics only explores one branch of the MCM decision tree, meaning
  program outputs are typically probabilistic and statistics produced by measurement processes are
  conditional on the selected decision tree path.

* Two new verifiers were added to the `quantum.paulirot` operation. They verify that the Pauli word
  length and the number of qubit operands are the same, and that all of the Pauli words are legal.
  [(#2405)](https://github.com/PennyLaneAI/catalyst/pull/2405)

* `qml.vjp` can now be used with Catalyst and program capture.
  [(#2279)](https://github.com/PennyLaneAI/catalyst/pull/2279)

<h3>Breaking changes 💔</h3>

* (Compiler integrators only) The versions of StableHLO/LLVM/Enzyme used by Catalyst have been updated.
  [(#2415)](https://github.com/PennyLaneAI/catalyst/pull/2415)
  [(#2416)](https://github.com/PennyLaneAI/catalyst/pull/2416)
  [(#2444)](https://github.com/PennyLaneAI/catalyst/pull/2444)
  [(#2445)](https://github.com/PennyLaneAI/catalyst/pull/2445)

  - The StableHLO version has been updated to
  [v1.13.7](https://github.com/openxla/stablehlo/tree/v1.13.7).
  - The LLVM version has been updated to
  [commit 8f26458](https://github.com/llvm/llvm-project/tree/8f264586d7521b0e305ca7bb78825aa3382ffef7).
  - The Enzyme version has been updated to
  [v0.0.238](https://github.com/EnzymeAD/Enzyme/releases/tag/v0.0.238).

* When an integer argnums is provided to `catalyst.vjp`, a singleton dimension is now squeezed
  out. This brings the behaviour in line with that of `grad` and `jacobian`.
  [(#2279)](https://github.com/PennyLaneAI/catalyst/pull/2279)

* Dropped support for NumPy 1.x following its end-of-life. NumPy 2.0 or higher is now required.
  [(#2407)](https://github.com/PennyLaneAI/catalyst/pull/2407)

<h3>Deprecations 👋</h3>

<h3>Bug fixes 🐛</h3>

* Fix `replace_ir` for certain stages when used with gradients.
  [(#2436)](https://github.com/PennyLaneAI/catalyst/pull/2436)

* Restore the ability to differentiate multiple (expectation value) QNode results with the
  adjoint-differentiation method.
  [(#2428)](https://github.com/PennyLaneAI/catalyst/pull/2428)

* Fixed the angle conversion when lowering `qec.ppr` and `qec.ppr.arbitrary` operations to
  `__catalyst__qis__PauliRot` runtime calls. The PPR rotation angle is now correctly multiplied
  by 2 to match the PauliRot convention (`PauliRot(φ) == PPR(φ/2)`).
  [(#2414)](https://github.com/PennyLaneAI/catalyst/pull/2414)

* Fixed the `catalyst` CLI tool silently listening to stdin when run without an input file, even when given flags like `--list-passes` that should override this behaviour.
  [(2447)](https://github.com/PennyLaneAI/catalyst/pull/2447)
  
* Fixing incorrect lowering of PPM into CAPI calls when the PPM is in the negative basis.
  [(#2422)](https://github.com/PennyLaneAI/catalyst/pull/2422)

* Fixed the GlobalPhase discrepancies when executing gridsynth in the PPR basis.
  [(#2433)](https://github.com/PennyLaneAI/catalyst/pull/2433)


<h3>Internal changes ⚙️</h3>

* `catalyst.python_interface.xdsl_universe.XDSL_UNIVERSE` has been renamed to `CATALYST_XDSL_UNIVERSE`.
  [(#2435)](https://github.com/PennyLaneAI/catalyst/pull/2435)

* The private helper `_extract_passes` of `qfunc.py` uses `BoundTransform.tape_transform`
  instead of the deprecated `BoundTransform.transform`.
  `jax_tracer.py` and `tracing.py` also updated accordingly.
  [(#2440)](https://github.com/PennyLaneAI/catalyst/pull/2440)

* Autograph is no longer applied to decomposition rules based on whether it's applied to the workflow itself.
  Operator developers now need to manually apply autograph to decomposition rules when needed.
  [(#2421)](https://github.com/PennyLaneAI/catalyst/pull/2421)

* The quantum dialect MLIR and TableGen source has been refactored to place type and attribute
  definitions in separate file scopes.
  [(#2329)](https://github.com/PennyLaneAI/catalyst/pull/2329)

* Added lowering of `qec.ppm`, `qec.ppr`, and `quantum.paulirot` to the runtime CAPI and QuantumDevice C++ API.
  [(#2348)](https://github.com/PennyLaneAI/catalyst/pull/2348)
  [(#2413)](https://github.com/PennyLaneAI/catalyst/pull/2413)

* Added LLVM conversion patterns to lower QEC dialect operations to their corresponding runtime
  CAPI calls.
  This includes `qec.ppr` and `qec.ppr.arbitrary` (lowered to `__catalyst__qis__PauliRot`),
  `qec.ppm` (lowered to `__catalyst__qis__PauliMeasure`). This enables device execution of QEC
  operations through the Catalyst runtime.
  [(#2389)](https://github.com/PennyLaneAI/catalyst/pull/2389)

* A new compiler pass `unroll-conditional-ppr-ppm` for lowering conditional PPR and PPMs
  into normal PPR and PPMs with SCF dialect to support runtime execution.
  [(#2390)](https://github.com/PennyLaneAI/catalyst/pull/2390)

* Increased format size for the `--mlir-timing` flag, displaying more decimals for better timing precision.
  [(#2423)](https://github.com/PennyLaneAI/catalyst/pull/2423)
  
* Added global phase tracking to the `to-ppr` compiler pass. When converting quantum gates to
  Pauli Product Rotations (PPR), the pass now emits `quantum.gphase` operations to preserve
  global phase correctness.
  [(#2419)](https://github.com/PennyLaneAI/catalyst/pull/2419)

* New qubit-type specializations have been added to Catalyst's MLIR type system. These new qubit
  types include `!quantum.bit<logical>`, `!quantum.bit<qec>` and `!quantum.bit<physical>`. The
  original `!quantum.bit` type continues to be supported and used as the default qubit type.
  [(#2369)](https://github.com/PennyLaneAI/catalyst/pull/2369)

  The corresponding register-type specializations have also been added.
  [(#2431)](https://github.com/PennyLaneAI/catalyst/pull/2431)

* The upstream MLIR `Test` dialect is now available via the `catalyst` command line tool.
  [(#2417)](https://github.com/PennyLaneAI/catalyst/pull/2417)

* A new compiler pass `lower-qec-init-ops` has been added to lower QEC initialization operations
  to Quantum dialect operations. This pass converts `qec.prepare` to `quantum.custom` and
  `qec.fabricate` to `quantum.alloc_qb` + `quantum.custom`, enabling runtime execution of
  QEC state preparation operations.
  [(#2424)](https://github.com/PennyLaneAI/catalyst/pull/2424)

<h3>Documentation 📝</h3>

* Updated the Unified Compiler Cookbook to be compatible with the latest versions of PennyLane and Catalyst.
  [(#2406)](https://github.com/PennyLaneAI/catalyst/pull/2406)

* Updated the changelog and builtin_passes.py to link to https://pennylane.ai/compilation/pauli-based-computation instead.
  [(#2409)](https://github.com/PennyLaneAI/catalyst/pull/2409)

<h3>Contributors ✍️</h3>

This release contains contributions from (in alphabetical order):
Ali Asadi,
Joey Carter,
Yushao Chen,
Sengthai Heng,
David Ittah,
Jeffrey Kam,
Mehrdad Malekmohammadi,
River McCubbin,
Mudit Pandey,
Andrija Paurevic,
David D.W. Ren,
Paul Haochen Wang.
