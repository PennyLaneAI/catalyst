# Release 0.13.0 (development release)

<h3>New features since last release</h3>

<h3>Improvements 🛠</h3>

* Workflows `for_loop`, `while_loop` and `cond` now error out if `qml.capture` is enabled.
  [(#1945)](https://github.com/PennyLaneAI/catalyst/pull/1945)

*  Displays Catalyst version in `quantum-opt --version` output.
  [(#1922)](https://github.com/PennyLaneAI/catalyst/pull/1922)

* Snakecased keyword arguments to :func:`catalyst.passes.apply_pass()` are now correctly parsed
  to kebab-case pass options [(#1954)](https://github.com/PennyLaneAI/catalyst/pull/1954).
  For example:

  ```python
  @qjit(target="mlir")
  @catalyst.passes.apply_pass("some-pass", "an-option", maxValue=1, multi_word_option=1)
  @qml.qnode(qml.device("null.qubit", wires=1))
  def example():
      return qml.state()
  ```

  which looks like the following line in the MLIR:

  ```pycon
  %0 = transform.apply_registered_pass "some-pass" with options = {"an-option" = true, "maxValue" = 1 : i64, "multi-word-option" = 1 : i64}
  ```

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

* Fix type promotion on branch
  [(#1977)](https://github.com/PennyLaneAI/catalyst/pull/1977)

* Fix errors in AutoGraph transformed functions when `qml.prod` is used together with other operator
  transforms (e.g. `qml.adjoint`).
  [(#1910)](https://github.com/PennyLaneAI/catalyst/pull/1910)

* A bug in the `NullQubit::ReleaseQubit()` method that prevented the deallocation of individual
  qubits on the `"null.qubit"` device has been fixed.
  [(#1926)](https://github.com/PennyLaneAI/catalyst/pull/1926)

<h3>Internal changes ⚙️</h3>

* Updates use of `qml.transforms.dynamic_one_shot.parse_native_mid_circuit_measurements` to improved signature.
  [(#1953)](https://github.com/PennyLaneAI/catalyst/pull/1953)

* When capture is enabled, `qjit(autograph=True)` will use capture autograph instead of catalyst autograph.
  [(#1960)](https://github.com/PennyLaneAI/catalyst/pull/1960)

* QJitDevice helper `extract_backend_info` removed its redundant `capabilities` argument.
  [(#1956)](https://github.com/PennyLaneAI/catalyst/pull/1956)

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

* The default compilation pipeline is now available as `catalyst.pipelines.default_pipeline()`. The
  function `catalyst.pipelines.get_stages()` has also been removed as it was not used and duplicated
  the `CompileOptions.get_stages()` method.
  [(#1941)](https://github.com/PennyLaneAI/catalyst/pull/1941)

* Utility functions for modifying an existing compilation pipeline have been added to the
  `catalyst.pipelines` module.
  [(#1941)](https://github.com/PennyLaneAI/catalyst/pull/1941)

  These functions provide a simple interface to insert passes and stages into a compilation
  pipeline. The available functions are `insert_pass_after`, `insert_pass_before`,
  `insert_stage_after`, and `insert_stage_before`. For example,

  ```pycon
  >>> from catalyst.pipelines import insert_pass_after
  >>> pipeline = ["pass1", "pass2"]
  >>> insert_pass_after(pipeline, "new_pass", ref_pass="pass1")
  >>> pipeline
  ['pass1', 'new_pass', 'pass2']
  ```

* A new built-in compilation pipeline for experimental MBQC workloads has been added, available as
  `catalyst.ftqc.mbqc_pipeline()`.
  [(#1942)](https://github.com/PennyLaneAI/catalyst/pull/1942)

  The output of this function can be used directly as input to the `pipelines` argument of
  :func:`~.qjit`, for example,

  ```python
  from catalyst.ftqc import mbqc_pipeline

  @qjit(pipelines=mbqc_pipeline())
  @qml.qnode(dev)
  def workload():
      ...
  ```

* The `mbqc.graph_state_prep` operation has been added to the MBQC dialect. This operation prepares
  a graph state with arbitrary qubit connectivity, specified by an input adjacency-matrix operand,
  for use in MBQC workloads.
  [(#1965)](https://github.com/PennyLaneAI/catalyst/pull/1965)

* `catalyst.accelerate`, `catalyst.debug.callback`, and `catalyst.pure_callback`, `catalyst.debug.print`, and `catalyst.debug.print_memref` now work when capture is enabled.
  [(#1902)](https://github.com/PennyLaneAI/catalyst/pull/1902)

* The merge rotation pass in Catalyst (:func:`~.passes.merge_rotations`) now also considers
  `qml.Rot` and `qml.CRot`.
  [(#1955)](https://github.com/PennyLaneAI/catalyst/pull/1955)

<h3>Documentation 📝</h3>

<h3>Contributors ✍️</h3>

This release contains contributions from (in alphabetical order):

Joey Carter,
Sengthai Heng,
David Ittah,
Christina Lee,
Andrija Paurevic,
Roberto Turrado,
Paul Haochen Wang.
