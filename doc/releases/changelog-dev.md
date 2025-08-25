# Release 0.13.0 (development release)

<h3>New features since last release</h3>

* Catalyst now provides native support for `SingleExcitation`, `DoubleExcitation`,
  and `PCPhase` on compatible devices like Lightning simulators.
  This enhancement avoids unnecessary gate decomposition,
  leading to reduced compilation time and improved overall performance.
  [(#1980)](https://github.com/PennyLaneAI/catalyst/pull/1980)
  [(#1987)](https://github.com/PennyLaneAI/catalyst/pull/1987)

  For example, the code below is captured with `PCPhase` avoiding the
  decomposition to many `MultiControlledX` and `PhaseShift` gates:

  ```python
  dev = qml.device("lightning.qubit", wires=3)

  @qml.qnode(dev)
  def circuit():
      qml.ctrl(qml.PCPhase(0.5, dim=1, wires=[0, 2]), control=[1])
      return qml.probs()
  ```

<h3>Improvements 🛠</h3>

* Adjoint differentiation is used by default when executing on lightning devices, significantly reduces gradient computation time.
  [(#1961)](https://github.com/PennyLaneAI/catalyst/pull/1961)

* Added `detensorizefunctionboundary` pass to remove scalar tensors across function boundaries and enabled `symbol-dce` pass to remove dead functions, reducing the number of instructions for compilation.
  [(#1904)](https://github.com/PennyLaneAI/catalyst/pull/1904)

* Workflows `for_loop`, `while_loop` and `cond` now error out if `qml.capture` is enabled.
  [(#1945)](https://github.com/PennyLaneAI/catalyst/pull/1945)

* Displays Catalyst version in `quantum-opt --version` output.
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

*  `Commuting Clifford Pauli Product Rotation (PPR) operations, past non-Clifford PPRs, now supports P(π/2) Cliffords in addition to P(π/4)`
   [(#1966)](https://github.com/PennyLaneAI/catalyst/pull/1966)

<h3>Breaking changes 💔</h3>

* The `shots` property has been removed from `OQDDevice`. The number of shots for a qnode execution is now set directly on the qnode via `qml.set_shots`,
  either used as decorator `@qml.set_shots(num_shots)` or directly on the qnode `qml.set_shots(qnode, shots=num_shots)`.
  [(#1988)](https://github.com/PennyLaneAI/catalyst/pull/1988)

* The JAX version used by Catalyst is updated to 0.6.2.
  [(#1897)](https://github.com/PennyLaneAI/catalyst/pull/1897)

* The version of LLVM and Enzyme used by Catalyst has been updated.
  The mlir-hlo dependency has been replaced with stablehlo.
  [(#1916)](https://github.com/PennyLaneAI/catalyst/pull/1916)
  [(#1921)](https://github.com/PennyLaneAI/catalyst/pull/1921)

  The LLVM version has been updated to
  [commit f8cb798](https://github.com/llvm/llvm-project/tree/f8cb7987c64dcffb72414a40560055cb717dbf74).
  The stablehlo version has been updated to
  [commit 69d6dae](https://github.com/openxla/stablehlo/commit/69d6dae46e1c7de36e6e6973654754f05353cba5).
  The Enzyme version has been updated to
  [v0.0.186](https://github.com/EnzymeAD/Enzyme/releases/tag/v0.0.186).

<h3>Deprecations 👋</h3>

* Deprecated usages of `Device.shots` along with setting `device(..., shots=...)`.
  Heavily adjusted frontend pipelines within qfunc, tracer, verification and QJITDevice to account for this change.
  [(#1952)](https://github.com/PennyLaneAI/catalyst/pull/1952)

<h3>Bug fixes 🐛</h3>

* Fix type promotion on conditional branches, where the return values from `cond` should be the promoted one.
  [(#1977)](https://github.com/PennyLaneAI/catalyst/pull/1977)

* Fix wrong handling of partitioned shots in the decomposition pass of `measurements_from_samples`.
  [(#1981)](https://github.com/PennyLaneAI/catalyst/pull/1981)

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
  arithmetic observables, `Hermitian` observables, `for_loop` outside qnodes, `cond` outside qnodes,
  `while_loop` outside QNode's, and `cond` with elif branches.
  [(#1844)](https://github.com/PennyLaneAI/catalyst/pull/1844)
  [(#1850)](https://github.com/PennyLaneAI/catalyst/pull/1850)
  [(#1903)](https://github.com/PennyLaneAI/catalyst/pull/1903)
  [(#1896)](https://github.com/PennyLaneAI/catalyst/pull/1896)
  [(#1889)](https://github.com/PennyLaneAI/catalyst/pull/1889)
  [(#1973)](https://github.com/PennyLaneAI/catalyst/pull/1973)

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

* Catalyst now supports *array-backed registers*, meaning that `quantum.insert` operations now allow
  for the insertion of a qubit into an arbitrary position within a register (if the qubit originally
  at that position has been deallocated).

<h3>Documentation 📝</h3>

<h3>Contributors ✍️</h3>

This release contains contributions from (in alphabetical order):

Ali Asadi,
Joey Carter,
Yushao Chen,
Sengthai Heng,
David Ittah,
Christina Lee,
Joseph Lee,
Andrija Paurevic,
Roberto Turrado,
Paul Haochen Wang.
