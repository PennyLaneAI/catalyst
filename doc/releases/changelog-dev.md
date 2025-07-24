# Release 0.13.0 (development release)

<h3>New features since last release</h3>

<h3>Improvements 🛠</h3>

<h3>Breaking changes 💔</h3>

* The JAX version used by Catalyst is updated to 0.6.2.
  [(#1897)](https://github.com/PennyLaneAI/catalyst/pull/1897)

* The version of LLVM, mlir-hlo, and Enzyme used by Catalyst has been updated.
  [(#1916)](https://github.com/PennyLaneAI/catalyst/pull/1916)

  The LLVM version has been updated to
  [commit f8cb798](https://github.com/llvm/llvm-project/tree/f8cb7987c64dcffb72414a40560055cb717dbf74).
  The mlir-hlo version has been updated to
  [commit 1dd2e71](https://github.com/tensorflow/mlir-hlo/tree/1dd2e71331014ae0373f6bf900ce6be393357190).
  The Enzyme version has been updated to
  [v0.0.186](https://github.com/EnzymeAD/Enzyme/releases/tag/v0.0.186).

<h3>Deprecations 👋</h3>

<h3>Bug fixes 🐛</h3>

* A bug in the `NullQubit::ReleaseQubit()` method that prevented the deallocation of individual
  qubits on the `"null.qubit"` device has been fixed.
  [(#1926)](https://github.com/PennyLaneAI/catalyst/pull/1926)

<h3>Internal changes ⚙️</h3>

* Raise warning when subroutines are used without capture enabled.
  [(#1930)](https://github.com/PennyLaneAI/catalyst/pull/1930)

* Update imports for noise transforms from `pennylane.transforms` to `pennylane.noise`.
  [(#1918)](https://github.com/PennyLaneAI/catalyst/pull/1918)

* Improve error message for quantum subroutines when used outside a quantum context.
  [(#1932)](https://github.com/PennyLaneAI/catalyst/pull/1932)

* `from_plxpr` now supports adjoint and ctrl operations and transforms, operator
  arithemtic observables, `Hermitian` observables, `for_loop` outside qnodes,
  and `while_loop` outside QNode's.
  [(#1844)](https://github.com/PennyLaneAI/catalyst/pull/1844)
  [(#1850)](https://github.com/PennyLaneAI/catalyst/pull/1850)
  [(#1903)](https://github.com/PennyLaneAI/catalyst/pull/1903)
  [(#1896)](https://github.com/PennyLaneAI/catalyst/pull/1896)
  [(#1889)](https://github.com/PennyLaneAI/catalyst/pull/1889)

* The `qec.layer` and `qec.yield` operations have been added to the QEC dialect to represent a group
  of QEC operations. The main use case is to analyze the depth of a circuit.
  Also, this is a preliminary step towards supporting parallel execution of QEC layers.
  [(#1917)](https://github.com/PennyLaneAI/catalyst/pull/1917)

* Conversion patterns for the single-qubit `quantum.alloc_qb` and `quantum.dealloc_qb` operations
  have been added for lowering to the LLVM dialect. These conversion patterns allow for execution of
  programs containing these operations.
  [(#1920)](https://github.com/PennyLaneAI/catalyst/pull/1920)

<h3>Documentation 📝</h3>

<h3>Contributors ✍️</h3>

This release contains contributions from (in alphabetical order):

Joey Carter,
Sengthai Heng,
Christina Lee,
Andrija Paurevic,
Paul Haochen Wang.
