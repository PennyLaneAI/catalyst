# Release 0.16.0 (development release)

<h3>New features since last release</h3>

* ``qp.allocate`` now supports ``state="magic"`` and ``state="magic_conj"``. Both
  capture and legacy frontends emit the same ``allocate`` / ``deallocate`` jaxpr
  primitives; MLIR lowering selects ``qref.alloc`` or ``pbc.ref.fabricate`` based
  on the ``state`` argument.
  Requires the corresponding PennyLane release for the new ``AllocateState`` values.
  [(#3027)](https://github.com/PennyLaneAI/catalyst/pull/3027), [(#3029)](https://github.com/PennyLaneAI/catalyst/pull/3029)

* The `local-random` unitary folding option for :func:`~.mitigate_with_zne` is now implemented,
  reproducing Mitiq's ``fold_gates_at_random``: every gate is folded ``floor((scale_factor-1)/2)``
  times, then a random subset is folded once more (without replacement) to reach ``scale_factor * n``
  gates. Non-integer scale factors are now also accepted for `local-random`. The `mitigation.zne`
  operation's `numFolds` operand is now always a floating-point tensor; the integer folding methods
  require integral values and convert the count internally.
  [(#2956)](https://github.com/PennyLaneAI/catalyst/pull/2956)

<h3>Improvements 🛠</h3>

* A new runtime transport layer for remote/local executors is introduced.
  [(#3043)](https://github.com/PennyLaneAI/catalyst/pull/3043)

* A `BufferizableOpInterface` implementation is now added for `catalyst.launch_kernel` operation and it is now bufferizable.
  [(#3024)](https://github.com/PennyLaneAI/catalyst/pull/3024)

* `quantum.extract` canonicalization now looks through a `quantum.insert` at a distinct
  static index, rewriting the extract to read from the register feeding the insert and
  sinking the bypassed insert below the gates acting on the extracted qubits. This removes
  the false data dependency between wires that act on different qubits of the same register
  and leaves extracts grouped above the gates and inserts below them.
  [(#2965)](https://github.com/PennyLaneAI/catalyst/pull/2965)

* Adds a `catalyst::symbolic_array` operation and integrates it with the new `qp.capture.symbolic_array` function.
  [(#2982)](https://github.com/PennyLaneAI/catalyst/pull/2982)

* The `decompose-lowering` pass now supports applying a selection of the available decomposition rules via the `target_rules` parameter.
  The pass also no longer applies the `inline`, `cse` and `canonicalize` passes to avoid unnecessary IR mutations.
  Instead, decomposition rules are deterministically inlined by a custom function (`inline` is non-deterministic, using an estimated benefit and threshold as criteria for inlining).
  Decomposition rules are no longer removed after the `decompose-lowering` pass, which allows them to be used by subsequent passes, namely `graph-decomposition`.
  Instead, rules are removed by the `symbol-dce` pass at the end of the `QuantumCompilationStage`.
  [(#2973)](https://github.com/PennyLaneAI/catalyst/pull/2973)

* The new `pennylane.core.Operator2` can now be lowered to MLIR with program capture for operators
  without non-lowerable arguments. `Operator2` classes are now lowered to specialized operations
  where applicable, unlocking compilation and execution for these cases. `qp.specs` and the
  `ResourceAnalysis` pass now support the `quantum::OperatorOp` and `qref::OperatorOp` instructions.
  [(#2979)](https://github.com/PennyLaneAI/catalyst/pull/2979)
  [(#2969)](https://github.com/PennyLaneAI/catalyst/pull/2969)
  [(#2980)](https://github.com/PennyLaneAI/catalyst/pull/2980)
  [(#2990)](https://github.com/PennyLaneAI/catalyst/pull/2990)
  [(#2993)](https://github.com/PennyLaneAI/catalyst/pull/2993)
  [(#2998)](https://github.com/PennyLaneAI/catalyst/pull/2998)
  [(#2981)](https://github.com/PennyLaneAI/catalyst/pull/2981)

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

* `ResourceAnalysis` now uses a single JSON serializer owned by `ResourceResult`, removing
  duplicate serialization logic and keeping its output consistent.
  [(#3007)](https://github.com/PennyLaneAI/catalyst/issues/3007)

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

* The `--decompose-lowering` pass now uses the `DecomposableGate` interface, allowing it to support
  many new gate operations, including `quantum.paulirot`.
  [(#2893)](https://github.com/PennyLaneAI/catalyst/pull/2893)
  [(#3040)](https://github.com/PennyLaneAI/catalyst/pull/3040)

* Exclude more packages from AutoGraph conversion, since converting code unintentionally can lead
  to tracing errors.
  [(#2891)](https://github.com/PennyLaneAI/catalyst/pull/2891)

* Dynamically allocated wires can now be used in quantum adjoints.
  [(#2720)](https://github.com/PennyLaneAI/catalyst/pull/2720)

* Dynamic shapes with ``qp.cond`` are now supported with ``qjit(capture=True)``:
  [(#2740)](https://github.com/PennyLaneAI/catalyst/pull/2740)

* The `catalyst.custom_call` operation now accepts an optional `backend_config` attribute,
  which allows backend-specific configuration to be attached to custom calls.
  [(#3037)](https://github.com/PennyLaneAI/catalyst/pull/3037)

* Introduced compile-time python-decompositions, allowing compiler passes to lower decomposition
  rules instantiated with static data (ex. pauli strings). Using this, the `graph-decomposition`
  pass can now decompose `quantum.paulirot` operations using the decomposition rule defined in
  PennyLane.
  [(#2769)](https://github.com/PennyLaneAI/catalyst/pull/2769)

* Added ``CZ`` support to ``to-ppr`` pass.
  [(#3009)](https://github.com/PennyLaneAI/catalyst/pull/3009)

<h3>Breaking changes 💔</h3>

* Python 3.11 is no longer supported. Catalyst now requires Python 3.12 or newer.
  [(#2974)](https://github.com/PennyLaneAI/catalyst/pull/2974)

* Catalyst's xDSL dependencies have been updated to `xdsl` 0.63.0 and `xdsl-jax` 0.5.2.
  [(#2840)](https://github.com/PennyLaneAI/catalyst/pull/2840)

* Removes support for `Transform.plxpr_transform` from the `qp.qjit(capture=True)` capture pipeline.
  All transforms must now have a MLIR or XDSL implementation and a corresponding `pass_name`.

* Support for `qjit` integration with `cudaq` has been removed in order to feasbily drop support
  for Python 3.11.
  [(#2984)](https://github.com/PennyLaneAI/catalyst/pull/2984)

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

<h3>Internal changes ⚙️</h3>

* The `dim` argument of the `quantum.pcphase` operation has been changed to a static integer attribute
  (previously a dynamic float operand). This allows, among other things, the decomposition graph to
  distinguish pcphase gates with different `dim` values, since they need different decomposition rules.
  [(#3034)](https://github.com/PennyLaneAI/catalyst/pull/3034)

* The `cond` PLxPR primitive's lowering rule no longer expects a `True` Literal for the predicate
  of the default else branch.
  [(#3018)](https://github.com/PennyLaneAI/catalyst/pull/3018)

* Add the `DecomposableGate` op interface to allow generic handling of operations in the `graph-decomposition` pass.
  This allows arbitrary operations implementing the interface to be registered to and decomposed by the graph.
  This also allows the use of python-decompositions for any operator pre-registered in the frontend graph.
  The graph solver now supports the new `graphOpId`s provided by the interface, as well as the legacy pathway with `name`, `numWires` etc.
  [(#2983)](https://github.com/PennyLaneAI/catalyst/pull/2983)
  [(#3022)](https://github.com/PennyLaneAI/catalyst/pull/3022)
  [(#3039)](https://github.com/PennyLaneAI/catalyst/pull/3039)

* The `graph-decomposition` pass eliminates three redundant IR manipulations:
  the cloning, removal, and re-insertion of user rules. This optimization is particularly
  beneficial when the pass is executed multiple times within the compilation pipeline.
  [(#2977)](https://github.com/PennyLaneAI/catalyst/pull/2977)

* `from_plxpr` no longer depends on the `Transform.plxpr_transform` property.
  [(#3004)](https://github.com/PennyLaneAI/catalyst/pull/3004)

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

* Rename the pipeline names in the default pipeline specification (e.g. `quantum-compilation-pipeline`) to match the
  `-stage` naming convention used when invoking them from the command line (e.g. `quantum-compilation-stage`).
  [#3002](https://github.com/PennyLaneAI/catalyst/pull/3002)

<h3>Documentation 📝</h3>

* A broken link was removed in the [Compiler Core](https://docs.pennylane.ai/projects/catalyst/en/stable/modules/mlir.html) documentation page. The link referred to where precompiled decomposition rules were implemented, which has since been refactored.
  [(#2913)](https://github.com/PennyLaneAI/catalyst/pull/2913)

* The documentation for `QJIT.mlir` and `QJIT.mlir_opt` was updated with type hints and docstrings that better reflect the compilation-dependent nature of the properties.
  [(#2975)](https://github.com/PennyLaneAI/catalyst/pull/2975)

* The [MLIR Plugins](https://docs.pennylane.ai/projects/catalyst/en/stable/dev/plugins.html)
  documentation has been updated to fix a number of typos and formatting issues, and to improve
  overall readability.
  [(#3005)](https://github.com/PennyLaneAI/catalyst/pull/3005)

<h3>Contributors ✍️</h3>

This release contains contributions from (in alphabetical order):

Ali Asadi,
Joey Carter,
Yushao Chen,
Lillian Frederiksen,
Sengthai Heng,
David Ittah,
JiaRung Jian,
Jacob Kitchen,
Korbinian Kottmann,
Christina Lee,
Joseph Lee,
Rylan Malarchick,
Mehrdad Malekmohammadi,
River McCubbin,
Shuli Shu,
Paul Haochen Wang,
Jake Zaia.
