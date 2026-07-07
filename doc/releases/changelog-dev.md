# Release 0.16.0 (development release)

<h3>New features since last release</h3>


<h3>Improvements 🛠</h3>

* The new `pennylane.core.Operator2` can now be lowered to MLIR with program capture for operators
  without non-lowerable arguments. `Operator2` classes are now lowered to specialized operations
  where applicable, unlocking compilation and execution for these cases. `qp.specs` and the
  `ResourceAnalysis` pass now support the `quantum::OperatorOp` and `qref::OperatorOp` instructions.
  [(#2979)](https://github.com/PennyLaneAI/catalyst/pull/2979)
  [(#2969)](https://github.com/PennyLaneAI/catalyst/pull/2969)
  [(#2980)](https://github.com/PennyLaneAI/catalyst/pull/2980)
  [(#2990)](https://github.com/PennyLaneAI/catalyst/pull/2990)
  [(#2993)](https://github.com/PennyLaneAI/catalyst/pull/2993)

* The `ResourceAnalysis` pass now reports each loop body and each subroutine as its own entry
  instead of folding their gate counts into the caller. Loops with constant bounds appear as `for_loop_<N>`
  with their trip count. Loops with dynamic bounds appear as `dyn_for_loop_<N>` with a stable
  identifier, and totals across the call graph are computed on demand.
  [(#2782)](https://github.com/PennyLaneAI/catalyst/pull/2782)
  [(#2900)](https://github.com/PennyLaneAI/catalyst/pull/2900)

* The `ResourceAnalysis` pass now supports IR in reference semantics natively, rather than requiring a conversion step.
  [(#2923)](https://github.com/PennyLaneAI/catalyst/pull/2923)

* The `resource-analysis` pass JSON output now includes `depth` for worst-case PBC layer depth
  (`any_commuting_depth` / `qubit_disjoint_depth`) per function and lifted loop entry.
  [(#2967)](https://github.com/PennyLaneAI/catalyst/pull/2967)

* The `--adjoint-lowering` pass no longer turns statically bounded for loops into
  dynamically bounded ones. In this way they remain analyzable by functionality like `qp.specs`.
  [(#2959)](https://github.com/PennyLaneAI/catalyst/issues/2959)

* The `--decompose-lowering` pass can now handle decomposition rule functions whose quantum register
  argument is at an arbitrary position in the argument list.
  [(#2836)](https://github.com/PennyLaneAI/catalyst/pull/2836)

* PPRs and PPMs can now be lowered properly into MLIR directly in the non-capture workflow.
  [(#2816)](https://github.com/PennyLaneAI/catalyst/pull/2816)

* The `--decompose-lowering` pass can now handle null decomposition rules, which are rule functions
  that do not have any quantum values as arguments or results. Gates with null decomposition rules
  are simply removed.
  [(#2855)](https://github.com/PennyLaneAI/catalyst/pull/2855)

* The ``--partition-layers`` pass now supports a ``disjoint-qubit`` option to group PBC ops
  into the same layer only when they act on disjoint qubits. By default, commuting ops on
  overlapping qubits may still be merged into one layer.
  [(#2858)](https://github.com/PennyLaneAI/catalyst/pull/2858)

* The :func:`~.passes.ppm_specs` now report circuit depth as ``depth``
  (layer count) and ``depth_type`` (``0`` = commuting ops on overlapping qubits may share a
  layer; ``1`` = only ops with disjoint qubit support may share a layer). The Python API accepts
  ``only_disjoint_qubit=True`` to run ``ppm-specs{disjoint-qubit=true}``. AOT ``ppm_specs`` no
  longer requires an explicit pipeline and no longer mixes MLIR into the JSON output.
  [(#2863)](https://github.com/PennyLaneAI/catalyst/pull/2863)

* The ``depth`` field reported by :func:`~.passes.ppm_specs` is now the worst-case depth
  across ``scf.if`` and ``scf.index_switch`` branches (taking the maximum over all branches)
  and across statically-bounded ``scf.for`` loops (multiplied by the trip count).
  Previously, branches were counted sequentially and PBC ops inside ``scf.for`` produced an
  error. ``scf.while`` and dynamically-bounded ``scf.for`` still produce an error.
  [(#2876)](https://github.com/PennyLaneAI/catalyst/pull/2876)
  [(#2877)](https://github.com/PennyLaneAI/catalyst/pull/2877)
  [(#2879)](https://github.com/PennyLaneAI/catalyst/pull/2879)

* Global toggles, ``compile_without_static_conditionals`` and ``compile_without_static_loops`` have
  been added to control the capture behaviour for ``catalyst``/``pennylane`` ``cond`` and
  ``for_loop`` instructions. Setting the toggle to ``True`` will automatically remove the respective
  construct from the captured program (i.e., evaluate it in Python) whenever the predicate or bounds
  are static.
  [(#2912)](https://github.com/PennyLaneAI/catalyst/pull/2912)

  For example, consider the following circuit with a statically defined `for` loop bound.

  ```python
  import pennylane as qp
  import catalyst

  catalyst.compile_without_static_loops = True

  @qp.qjit
  @qp.qnode(qp.device("lightning.qubit", wires=2))
  def f():
      @qp.for_loop(0, 2)
      def loop(i):
          qp.H(i)
      loop()
      return qp.state()
  ```
  Using the `catalyst.compile_without_static_loops` toggle, Catalyst will evaluate
  the `for_loop` in Python, which unrolls the `for_loop`. This can be verified by printing
  the `jaxpr` representation of the circuit.
  ```pycon
  >>> print(f.jaxpr)
  ...
            b:AbstractQreg() = qalloc 2:i64[]
            c:AbstractQbit() = qextract b 0:i64[]
            d:AbstractQbit() = qinst[
              adjoint=False
              ctrl_len=0
              op=Hadamard
              params_len=0
              qubits_len=1
            ] c
            e:AbstractQbit() = qextract b 1:i64[]
            f:AbstractQbit() = qinst[
              adjoint=False
              ctrl_len=0
              op=Hadamard
              params_len=0
              qubits_len=1
            ]
  ...
  ```

* The `--decompose-lowering` pass can now handle cases where the decomposed gate act on qubit values
  extracted from different quantum register SSA values, as long as all these quantum register values
  trace back to the same allocation.
  [(#2861)](https://github.com/PennyLaneAI/catalyst/pull/2861)

* The `--adjoint-lowering` pass can now handle adjoint operations containing control flow operations
  that have multiple quantum operands, of either quantum register or qubit type.
  [(#2868)](https://github.com/PennyLaneAI/catalyst/pull/2868)

* The `--decompose-lowering` pass now supports `quantum.paulirot` operators.
  [(#2893)](https://github.com/PennyLaneAI/catalyst/pull/2893)

* Exclude more packages from AutoGraph conversion, since converting code unintentionally can lead
  to tracing errors.
  [(#2891)](https://github.com/PennyLaneAI/catalyst/pull/2891)

* Dynamically allocated wires can now be used in quantum adjoints.
  [(#2720)](https://github.com/PennyLaneAI/catalyst/pull/2720)

* Dynamic shapes with ``qp.cond`` are now supported with ``qjit(capture=True)``:
  [(#2740)](https://github.com/PennyLaneAI/catalyst/pull/2740)

* Introduced compile-time python-decompositions, allowing compiler passes to lower decomposition
  rules instantiated with static data (ex. pauli strings). Using this, the `graph-decomposition`
  pass can now decompose `quantum.paulirot` operations using the decomposition rule defined in
  PennyLane.
  [(#2769)](https://github.com/PennyLaneAI/catalyst/pull/2769)

<h3>Breaking changes 💔</h3>

* Catalyst's xDSL dependencies have been updated to `xdsl` 0.63.0 and `xdsl-jax` 0.5.2.
  [(#2840)](https://github.com/PennyLaneAI/catalyst/pull/2840)

* Removes support for `Transform.plxpr_transform` from the `qp.qjit(capture=True)` capture pipeline.
  All transforms must now have a MLIR or XDSL implementation and a corresponding `pass_name`.

<h3>Deprecations 👋</h3>

<h3>Bug fixes 🐛</h3>

* Fixed a bug where the `ResourceAnalysis` pass only analyzed functions directly contained in
  the top-level module. Functions inside nested modules, such as kernels called through
  `catalyst.launch_kernel`, are now included in the output.
  [(#2961)](https://github.com/PennyLaneAI/catalyst/pull/2961)

* Fixed a bug in `DecompRuleInterpreter.cleanup` by replacing fragile string-based operator
  checks with strict type-based checking.
  [(#2873)](https://github.com/PennyLaneAI/catalyst/pull/2873)

* Fixed support of region-based adjoint (`qp.adjoint(qfunc)()`) when used in conjunction with
  dynamic qubit allocation.
  [(#2933)](https://github.com/PennyLaneAI/catalyst/pull/2933)

  For instance, the following would previously fail:
  ```py
  def fun(w):
    with qp.allocate(1) as qs:
        qp.S(qs[0])
    qp.X(w)

  @qp.qjit(capture=True)
  @qp.qnode(qp.device("null.qubit", wires=1))
  def circuit():
      qp.adjoint(fun)(0)
      return qp.probs()
  ```
  with the error message:
  ```
  catalyst.utils.exceptions.CompileError: catalyst failed with error code 1: Failed to run pipeline: QuantumCompilationStage
  Compilation failed:
  circuit:31:9: error: Unhandled operation in adjoint region
  circuit:31:9: note: see current operation: "quantum.dealloc"(%13) : (!quantum.reg) -> ()
  ```

* Fixed a bug where using `keep_intermediate=True` with `target="mlir"` resulted in an empty workspace
  folder being created and the files printed outside in the main directory.
  [(#2807)](https://github.com/PennyLaneAI/catalyst/pull/2807)

* Fixed a bug that passed incorrect SSA values to the final register deallocation when translating
  from the `qecp` to the `quantum` dialect. This bug prevented deallocation of unneeded registers
  after magic state injection.
  [(#2897)](https://github.com/PennyLaneAI/catalyst/pull/2897)

* Fix memory bugs in the PBC passes.
  [(#2918)](https://github.com/PennyLaneAI/catalyst/pull/2918)

* Fixed incorrect ``depth`` in :func:`~.passes.ppm_specs` when a ``quantum.extract`` appeared
  after a PBC op but read from a register not updated by that op. Layer grouping now checks
  data dependencies through insert to extract chains instead of textual op ordering.
  [(#2884)](https://github.com/PennyLaneAI/catalyst/pull/2884)

* Fixed the assembly format for `quantum.adjoint` when it has no quantum operands/results.
  [(#2938)](https://github.com/PennyLaneAI/catalyst/pull/2938)

* The `--decompose-lowering` pass no longer crashes on qreg-mode decomposition rules whose wire
  indices are passed as separate scalar arguments, e.g. `(qreg, *params, idx0, idx1, ...)`.
  [(#2952)](https://github.com/PennyLaneAI/catalyst/pull/2952)

<h3>Internal changes ⚙️</h3>

* Update tests to not use global capture toggle where possible.
  [(#2964)](https://github.com/PennyLaneAI/catalyst/pull/2964)

* The `/benchmark` GitHub comment trigger can now accept additional arguments and has been renamed to `!benchmark`.
  [(#2947)](https://github.com/PennyLaneAI/catalyst/pull/2947)

* The frontend now generates MLIR in reference semantics when capture is enabled.
  [(#2663)](https://github.com/PennyLaneAI/catalyst/pull/2663)
  [(#2664)](https://github.com/PennyLaneAI/catalyst/pull/2664)
  [(#2672)](https://github.com/PennyLaneAI/catalyst/pull/2672)
  [(#2694)](https://github.com/PennyLaneAI/catalyst/pull/2694)
  [(#2717)](https://github.com/PennyLaneAI/catalyst/pull/2717)
  [(#2720)](https://github.com/PennyLaneAI/catalyst/pull/2720)
  [(#2740)](https://github.com/PennyLaneAI/catalyst/pull/2740)
  [(#2757)](https://github.com/PennyLaneAI/catalyst/pull/2757)
  [(#2781)](https://github.com/PennyLaneAI/catalyst/pull/2781)
  [(#2834)](https://github.com/PennyLaneAI/catalyst/pull/2834)
  [(#2911)](https://github.com/PennyLaneAI/catalyst/pull/2911)

* A new pass `--convert-to-reference-semantics` has been added. The pass takes in MLIR in value
  semantics `quantum` dialect, and converts them to reference semantics `qref` dialect.
  [(#2920)](https://github.com/PennyLaneAI/catalyst/pull/2920)
  [(#2930)](https://github.com/PennyLaneAI/catalyst/pull/2930)
  [(#2931)](https://github.com/PennyLaneAI/catalyst/pull/2931)
  [(#2937)](https://github.com/PennyLaneAI/catalyst/pull/2937)
  [(#2945)](https://github.com/PennyLaneAI/catalyst/pull/2945)
  [(#2948)](https://github.com/PennyLaneAI/catalyst/pull/2948)

* Removed the internal ``mlir_specs`` function which was the old backend for :func:`qp.specs`. The resource analysis pass replaces its use.
  [(#2841)](https://github.com/PennyLaneAI/catalyst/pull/2841)

* Fixed ``KeyError`` in autograph when using ``qp.prod`` as a decorator with PennyLane >= 0.45.
  [(#2844)](https://github.com/PennyLaneAI/catalyst/pull/2844)

* Update RC nightly builds to read version number from the `_version.py` file
  [(#2797)](https://github.com/PennyLaneAI/catalyst/pull/2797)

* Fix build failures when using clang with GCC ≤ 13 libstdc++ by replacing
  `std::views::filter`/`std::views::transform` with `std::copy_if`/`std::transform`
  [(#2801)](https://github.com/PennyLaneAI/catalyst/pull/2801)

* A new, experimental compiler pass `convert-qecp-to-quantum` has been added to lower operations
  from the QEC Physical (`qecp`) dialect into the Quantum (`quantum`) dialect.
  [(#2822)](https://github.com/PennyLaneAI/catalyst/pull/2822)
  [(#2809)](https://github.com/PennyLaneAI/catalyst/pull/2809)
  [(#2824)](https://github.com/PennyLaneAI/catalyst/pull/2824)
  [(#2835)](https://github.com/PennyLaneAI/catalyst/pull/2835)
  [(#2839)](https://github.com/PennyLaneAI/catalyst/pull/2839)
  [(#2849)](https://github.com/PennyLaneAI/catalyst/pull/2849)
  [(#2927)](https://github.com/PennyLaneAI/catalyst/pull/2927)
  [(#2955)](https://github.com/PennyLaneAI/catalyst/pull/2955)

* The experimental compiler pass `convert-qecl-to-qecp` has been extended to lower
  transversal gate operations from the QEC Logical (`qecl`) dialect into the QEC
  Physical (`qecp`) dialect.
  [(#2776)](https://github.com/PennyLaneAI/catalyst/pull/2776)
  [(#2871)](https://github.com/PennyLaneAI/catalyst/pull/2871)
  [(#2922)](https://github.com/PennyLaneAI/catalyst/pull/2922)

* The experimental compiler pass `convert-quantum-to-qecl` has been extended to lower the
  `quantum.custom "T"` gate to the `qecl` layer as a subroutine using a magic state (or conjugate
  magic state in the case of the adjoint).
  [(#2870)](https://github.com/PennyLaneAI/catalyst/pull/2870)
  [(#2921)](https://github.com/PennyLaneAI/catalyst/pull/2921)

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

* A new operation has been added to the Quantum dialects to represent generic and high-level
  quantum operators, including operators with frontend-end specific data.
  [(#2883)](https://github.com/PennyLaneAI/catalyst/pull/2883)
  [(#2943)](https://github.com/PennyLaneAI/catalyst/pull/2943)
  [(#2951)](https://github.com/PennyLaneAI/catalyst/pull/2951)

* In order to support T gates and π/8 PPRs in the experimental QEC pipeline, the following new
  operations have been added:

  - `qecl.fabricate`, which fabricates a logical codeblock in a specified initial state (typically a
    magic state).
    [(#2865)](https://github.com/PennyLaneAI/catalyst/pull/2865)
  - `qecl.dealloc_cb`, which deallocates a single logical codeblock.
    [(#2866)](https://github.com/PennyLaneAI/catalyst/pull/2866)
  - `qecp.alloc_cb`, which allocates a single physical codeblock.
    [(#2867)](https://github.com/PennyLaneAI/catalyst/pull/2867)
  - `qecp.dealloc_cb`, which deallocates a single physical codeblock.
    [(#2867)](https://github.com/PennyLaneAI/catalyst/pull/2867)
  - `qecp.t`, which performs a T gate on a single physical qubit.
    [(#2888)](https://github.com/PennyLaneAI/catalyst/pull/2888)

* The experimental `convert-qecl-to-qecp` pass has been extended to support lowering
  `qecl.fabricate [magic]` to a subroutine that prepares a magic state through a simple,
  non-fault tolerant encoding.
  [(#2894)](https://github.com/PennyLaneAI/catalyst/pull/2894)

* The experimental QEC pipeline now supports compilation and execution of circuits that only
  include a single wire (a previously unsupported edge-case).
  [(#2897)](https://github.com/PennyLaneAI/catalyst/pull/2897)

* The experimental QEC pipeline now only generates subroutines for operations present in the
  compiled circuit, rather than generating all QEC subroutines.
  [(#2929)](https://github.com/PennyLaneAI/catalyst/pull/2929)

* More conservative casting to tracer arrays in conditionals to preserve constant (static) values
  better. This can be useful for optimizations that depend on values being static.
  [(#2892)](https://github.com/PennyLaneAI/catalyst/pull/2892)

* The experimental QEC pipeline now supports the following control-flow operations:

  - Conditionals (`scf.if`)
    [(#2872)](https://github.com/PennyLaneAI/catalyst/pull/2872)
  - For loops (`scf.for`)
    [(#2881)](https://github.com/PennyLaneAI/catalyst/pull/2881)
  - While loops (`scf.while`)
    [(#2905)](https://github.com/PennyLaneAI/catalyst/pull/2905)

* The experimental QEC pipeline now supports programs that sample wires, where before it only
  supported sampling mid-circuit measurements.
  [(#2941)](https://github.com/PennyLaneAI/catalyst/pull/2941)

  The QEC pipeline also now supports `qp.expval`, `qp.var` and `qp.probs` measurement processes when
  used in conjunction with the `measurements-from-samples` pass.
  [(#2958)](https://github.com/PennyLaneAI/catalyst/pull/2958)

<h3>Documentation 📝</h3>

* A broken link was removed in the [Compiler Core](https://docs.pennylane.ai/projects/catalyst/en/stable/modules/mlir.html) documentation page. The link referred to where precompiled decomposition rules were implemented, which has since been refactored.
  [(#2913)](https://github.com/PennyLaneAI/catalyst/pull/2913)

<h3>Contributors ✍️</h3>

This release contains contributions from (in alphabetical order):

Ali Asadi,
Joey Carter,
Yushao Chen,
Lillian Frederiksen,
Sengthai Heng,
David Ittah,
Christina Lee,
Mehrdad Malekmohammadi,
River McCubbin,
Shuli Shu,
Paul Haochen Wang,
Jake Zaia.
