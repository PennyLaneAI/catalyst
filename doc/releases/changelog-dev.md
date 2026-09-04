# Release 0.16.0 (development release)

<h3>New features since last release</h3>

* A new `quantum.ctrl` region op and a `ctrl-lowering` pass are added to the Quantum Dialect
  for controlled subcircuits in Catalyst.

  Programs can now express an entire controlled quantum region, as opposed to individual
  operations. The `ctrl-lowering` pass distributes the control wires and control values from
  the `quantum.ctrl` operation onto the individual gate operations inside the region.

  The corresponding reference semantics operation `qref.ctrl`, and the bidirectional conversion
  between the two operations, are also added.
  [(#3089)](https://github.com/PennyLaneAI/catalyst/pull/3089)
  [(#3090)](https://github.com/PennyLaneAI/catalyst/pull/3090)
  [(#3096)](https://github.com/PennyLaneAI/catalyst/pull/3096)
  [(#3116)](https://github.com/PennyLaneAI/catalyst/pull/3116)
  [(#3127)](https://github.com/PennyLaneAI/catalyst/pull/3127)
  [(#3131)](https://github.com/PennyLaneAI/catalyst/pull/3131)

* The graph-based decomposition system now supports **adjoint operators** for `Operator2`.
  [(#3120)](https://github.com/PennyLaneAI/catalyst/pull/3120)
  [(#3115)](https://github.com/PennyLaneAI/catalyst/pull/3115)

  For a target gate set, `Adjoint(Op)` is reached through any of three pathways:
    1. Rules registered on the base `Op`,
    2. Rules registered directly for `Adjoint(Op)`, and
    3. Rules *synthesized by distribution* (`decompose(Adjoint(Op)) = adjoint(decompose(Op))`).

* The graph-based decomposition system now supports **controlled operators** for `Operator2`,
  including single control (`C(Op)`), multiple controls (`<n>C(Op)`), and their composition with
  adjoint.
  [(#3129)](https://github.com/PennyLaneAI/catalyst/pull/3129)
  [(#3127)](https://github.com/PennyLaneAI/catalyst/pull/3127)

  Control is folded into the operator identity *control-outermost* (e.g. `C(Adjoint(Op))`), so
  `ctrl(adjoint(Op))` and `adjoint(ctrl(Op))` collapse to a single node, while a distinct control
  count is its own node keyed. For a target gate set, `<n>C(Op)` is reached through:
    1. Rules registered directly for `<n>C(Op)` (e.g. named `CNOT`/`CRX`, `ctrl_decomp_zyz`), and
    2. Rules *synthesized by distribution* (`decompose(C(Op)) = ctrl(decompose(Op))`), controlling
       each produced gate with the same control count.

  A rule may re-emit its base decomposition inside a `quantum.ctrl` region; the apply pipeline
  reduces it to op-level controls with `ctrl-lowering`, iterating
  `(decompose-lowering -> ctrl-lowering -> adjoint-lowering)` to a fixpoint.
  Controlled basis gates are only free when their own `<n>C(...)` id is in the target gate set.

* The `local-random` unitary folding option for :func:`~.mitigate_with_zne` is now implemented,
  reproducing Mitiq's ``fold_gates_at_random``: every gate is folded ``floor((scale_factor-1)/2)``
  times, then a random subset is folded once more (without replacement) to reach ``scale_factor * n``
  gates. Non-integer scale factors are now also accepted for `local-random`. The `mitigation.zne`
  operation's `numFolds` operand is now always a floating-point tensor; the integer folding methods
  require integral values and convert the count internally.
  [(#2956)](https://github.com/PennyLaneAI/catalyst/pull/2956)

<h3>Improvements 🛠</h3>

* `RuleLoweringWarning` is now silenced by default. To display these warnings, set
  `CATALYST_SILENCE_RULE_LOWERING_WARNINGS=0`. This helps debug unexpected decompositions where
  un-lowerable rules are silently dropped from the graph-decomposition system instead of raising
  an error.

* Add the `XMEM_REPLY_BRAM` memory type and use it to allocate reply buffers in dedicated BRAM.
  [(#3148)](https://github.com/PennyLaneAI/catalyst/pull/3148)

* a PennyLane `Backline` is serialized to the `catalyst.backline` module attribute and compiled
  through the transport passes.
  [(#3068)](https://github.com/PennyLaneAI/catalyst/pull/3068)

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
  [(#3109)](https://github.com/PennyLaneAI/catalyst/pull/3109)
  [(#3075)](https://github.com/PennyLaneAI/catalyst/pull/3075)
  [(#3162)](https://github.com/PennyLaneAI/catalyst/pull/3162)

* The graph-based decomposition system has been greatly improved.

  In previous versions, graph-based decomposition was occurring during the conversion from PennyLane
  PLXPR to catalyst JAXPR in a JAX-based interpreter, namely `DecompRuleInterpreter`.

  In Catalyst v0.15, the high-performance `--graph-decomposition` MLIR pass was developed to mirror PennyLane's graph solving in Python.

  This current Catalyst version migrates all graph-based decomposition logic out of `DecompRuleInterpreter` into `--graph-decomposition`:

  - Added the `DecomposableGate` op interface to allow generic handling of operations in the `graph-decomposition` pass.
    [(#2983)](https://github.com/PennyLaneAI/catalyst/pull/2983)
    [(#3022)](https://github.com/PennyLaneAI/catalyst/pull/3022)

    This allows arbitrary operations implementing the interface to be registered to and decomposed by the graph.
    This also allows the use of python-decompositions for any operator pre-registered in the frontend graph.

  - The graph solver now matches operators solely by `graphOpId`; the legacy `name`/`numWires` matching pathway has been removed.
    [(#3039)](https://github.com/PennyLaneAI/catalyst/pull/3039)
    [(#3046)](https://github.com/PennyLaneAI/catalyst/pull/3046)
    [(#3052)](https://github.com/PennyLaneAI/catalyst/pull/3052)
    [(#3053)](https://github.com/PennyLaneAI/catalyst/pull/3053)

    The format of `graphOpID` is as follows:
        op_name{dynamic_shape_dictionary}{wire_lens_dictionary}{static_data_dictionary}[UID]

    The types in the dynamic shape dictionary should be represented as a list of MLIR-style type annotations.
    The UID is a hash computed from the shapes, dtypes and pytree structures of any data on the Python operator that cannot be lowered to MLIR directly.

    For example, an operator with class name `HybridOpArg`, taking in one float param
    argument named `angle`, one wire argument named `cwires`, one static data argument
    `label="hello"`, and a computed UID of 10 would be parsed to the following graph op ID:
        HybridOpArg{angle:[tensor<f64>]}{cwires:1}{label:hello}[10]

    A node in the decomposition graph is completely identified by its `graphOpId`. For example,
        PauliRot{angle:[f64]}{wires:1}{pauli_word:X}
    and
        PauliRot{angle:[f64]}{wires:2}{pauli_word:XX}
    will have different decomposition rules.

  - A decomposition rule function can arrive in a piece of MLIR in one of three ways:
    1. As a precompiled rule shipped with the Catalyst package directly.
    This pathway was implemented in Catalyst v0.15.
    Note that this pathway only includes rules from gates with a fixed number of wires and no static data.

    2. When lowering a gate operation from JAXPR to MLIR, all rules reachable from that gate are injected into the IR.
    [(#3061)](https://github.com/PennyLaneAI/catalyst/pull/3061)
    [(#3160)](https://github.com/PennyLaneAI/catalyst/pull/3160)
    [(#3149)](https://github.com/PennyLaneAI/catalyst/pull/3149)
    [(#3169)](https://github.com/PennyLaneAI/catalyst/pull/3169)

    This pathway of rule injection can be opted-out via a new keyword argument on `qp.qjit` named `collect_decomp_rules`.
    This kwarg controls whether or not to compile the decomposition rules during lower-time. Default value is `True`.
    If ``False``, only the circuit itself will be compiled (aka standard legacy behavior).
    If `True`, all the decomposition rules reachable from all gates in the circuit will be compiled.
    If `capture=False`, or `capture="global"` and `qp.capture.enabled() == False`, this argument will be ignored.

    3. When the `--graph-decomposition` pass encounters a gate operation without an existing rule in the IR, it will compile rules from that gate with a newly launched Python subprocess on-demand.
    [(#2769)](https://github.com/PennyLaneAI/catalyst/pull/2769)
    [(#3110)](https://github.com/PennyLaneAI/catalyst/pull/3110)

    With pathways 2 and 3, gates with static data only known at compile time can now be decomposed using the decomposition rule defined in PennyLane.
    For example, this includes `quantum.paulirot`, with Pauli words being the static data.

  - The `graph-decomposition` pass eliminated three redundant IR manipulations:
    the cloning, removal, and re-insertion of user rules.
    This optimization is particularly beneficial when the pass is executed multiple times within the compilation pipeline.
    [(#2977)](https://github.com/PennyLaneAI/catalyst/pull/2977)

  - A few improvements have been made to the `--decompose-lowering` pass.
    [(#2973)](https://github.com/PennyLaneAI/catalyst/pull/2973)
    [(#2836)](https://github.com/PennyLaneAI/catalyst/pull/2836)
    [(#2855)](https://github.com/PennyLaneAI/catalyst/pull/2855)
    [(#3156)](https://github.com/PennyLaneAI/catalyst/pull/3156)
    [(#3158)](https://github.com/PennyLaneAI/catalyst/pull/3158)

    1. The pass now supports applying a selection of the available decomposition rules via the `target_rules` parameter.

    2. The pass also no longer applies the `inline`, `cse` and `canonicalize` passes to avoid unnecessary IR mutations.
    Instead, decomposition rules are deterministically inlined by a custom function (`inline` is non-deterministic, using an estimated benefit and threshold as criteria for inlining).

    3. Decomposition rules are no longer removed after the `decompose-lowering` pass, which allows them to be used by subsequent passes, namely `graph-decomposition`.
    Instead, rules are removed by the `symbol-dce` pass at the end of the `QuantumCompilationStage`.

    4. The pass can now handle decomposition rule functions whose quantum register argument is at an arbitrary position in the argument list.

    5. The pass can now handle null decomposition rules, which are rule functions that do not have any quantum values as arguments or results.
    Gates with null decomposition rules are simply removed.

    6. The pass can now handle register-mode rules that target gates in control flow regions whose qubits were extracted outside the region.

* A failure during AOT compilation is now downgraded to a warning and logged.
  [(#3100)](https://github.com/PennyLaneAI/catalyst/pull/3100)

* Adds the ability to use `pennylane.typing.AbstractArray` and `pennylane.wires.AbstractWires` as type hints for
  AOT compilation and as arguments to `pennylane.specs` calculations.
  [(#2953)](https://github.com/PennyLaneAI/catalyst/pull/2953)

* The `ResourceAnalysis` pass can now report concrete resource counts for nested loops in cases
  where the bounds of an inner loop are directly dependent on the loop variable of a static outer loop.
  [(#3140)](https://github.com/PennyLaneAI/catalyst/pull/3140)

  For example, this program reports a total of `56` `PauliX` operations, since the number of iterations of the inner loops can be statically determined from the outer loop:

  ```python
  import pennylane as qp

  @qp.qjit(autograph=True)
  @qp.qnode(qp.device("null.qubit", wires=1))
  def circuit():
      for i in range(8):
          for j in range(i):
              for _ in range(j):
                  qp.PauliX(0)

      return qp.expval(qp.X(0))

  resources = qp.specs(circuit, level=0)().resources
  print(resources.quantum_operations["PauliX"])  # 56
  ```

* The `ResourceAnalysis` pass has received a new compiler hint to more accurately estimate quantum
  resources in the presence of conditional operations (`scf.if` and `scf.index_switch`). The
  operations in question can be annotated with either a `catalyst.estimated_probability` or
  `catalyst.estimated_probabilities` attribute, respectively, to indicate the expected probability
  distribution over the branches. The counted resources are then scaled proportionally and summed.
  [(#3059)](https://github.com/PennyLaneAI/catalyst/pull/3059)

* Warnings and diagnostics emitted by successful Catalyst compiler subprocesses are now forwarded to
  Python callers instead of being silently discarded. LLVM diagnostic colors are preserved in
  interactive terminals.
  [(#3080)](https://github.com/PennyLaneAI/catalyst/pull/3080)

* A new runtime transport layer for remote/local executors is introduced.
  [(#3043)](https://github.com/PennyLaneAI/catalyst/pull/3043)
  [(#3045)](https://github.com/PennyLaneAI/catalyst/pull/3045)

* `qp.runtime_call` is supported to lower to an ordinary `catalyst.custom_call`. The shared library
  exporting the symbol is given via `qp.runtime_declare(..., library=...)` (or `runtime_call(..., library=...)`)
  and recorded on the module so the compiler links it. A local call may take a `buf` argument and
  may return nothing (a `void` call is kept for its side effects), neither of which a dispatched
  call allows.
  [(#3101)](https://github.com/PennyLaneAI/catalyst/pull/3101)

* A CPU transport backend built on libibverbs is added, which implements the controller and coprocessor
  session roles over RDMA.
  [(#3062)](https://github.com/PennyLaneAI/catalyst/pull/3062)

* A GPU transport backend is added, which implements the coprocessor session role.
  [(#3069)](https://github.com/PennyLaneAI/catalyst/pull/3069)

* Catalyst can now cross-compile target nested modules to standalone object files and
  either statically link them into the host program or ship them to an executor for dispatch.
 [(#3033)](https://github.com/PennyLaneAI/catalyst/pull/3033)

* A new `Transport` MLIR dialect is added, providing typed ops for driving a transport session's
  lifecycle at the IR level.
  [(#3047)](https://github.com/PennyLaneAI/catalyst/pull/3047)

* A `convert-transport-to-llvm` pass is added, lowering the `Transport` dialect ops to the
  transport runtime CAPI.
  [(#3048)](https://github.com/PennyLaneAI/catalyst/pull/3048)

* An `inject-transport-session` pass is added, which reads the `catalyst.backline` module
  attribute and emits the transport session lifecycle into the host entry function.
  [(#3063)](https://github.com/PennyLaneAI/catalyst/pull/3063)

* A `BufferizableOpInterface` implementation is added for the `Transport` dialect ops.
  [(#3064)](https://github.com/PennyLaneAI/catalyst/pull/3064)

* A `lower-decode-to-transport` pass is added, which replaces each qecp.decode_esm_css with
  a transport kick/collect round over its buffers.
  [(#3066)](https://github.com/PennyLaneAI/catalyst/pull/3066)

* A `remove-global-phases` pass is added, which removes global phases by deleting `quantum.gphase`
  operations without control wires.
  [(#3143)](https://github.com/PennyLaneAI/catalyst/pull/3143)

* An X/Z syndrome decode can now be routed to its own decoder in a backline coprocessor.
  `qecp.decode_esm_css` carries an optional `check_type` attribute recording which check family a
  syndrome came from, which `lower-decode-to-transport` maps to a `decoder_id` on `transport.kick`.
  [(#3092)](https://github.com/PennyLaneAI/catalyst/pull/3092)

* CPU & GPU backline transport backends via `memcpy` are added.
  [(#3113)](https://github.com/PennyLaneAI/catalyst/pull/3113)

* A new remote/local executor infrastructure has been added to Catalyst, enabling qnode kernels to
  be dispatched to a separate executor process.

  - `executor` dialect models the session lifecycle through an `!executor.session` handle that is
    threaded from `executor.open` through `send_binary`, `launch`, `call`, and `close`.
    [(#2909)](https://github.com/PennyLaneAI/catalyst/pull/2909)

  - `--convert-executor-to-llvm` pass that lowers each `executor` op to a call into the
    `__catalyst__executor__*` C-ABI runtime, marshalling string endpoints, memref descriptors, and
    per-argument metadata.
    [(#2910)](https://github.com/PennyLaneAI/catalyst/pull/2910)

  - Host-side runtime (`rt_executor`) that backs those symbols. It opens a TCP connection to the
    executor and uses LLVM's ORC v2 EPC as the wire protocol to ship cross-compiled kernel objects
    into the remote JIT.
    [(#2915)](https://github.com/PennyLaneAI/catalyst/pull/2915)

  - Added `executor.launch_async` and `executor.await` ops, paired by a new `!executor.token` type
    to the executor dialect along with the necessary lowerings. This allows one to start an async
    kernel on a background host thread and join it later.
    [(#3073)](https://github.com/PennyLaneAI/catalyst/pull/3073)
    [(#3031)](https://github.com/PennyLaneAI/catalyst/pull/3031)
    [(#3030)](https://github.com/PennyLaneAI/catalyst/pull/3030)

  - The `catalyst-executor` server side is now added to Catalyst that receives objects, maps them
    and calls them.
    [(#3088)](https://github.com/PennyLaneAI/catalyst/pull/3088)

  - A `catalyst.Executor` is added for deploying and managing the `catalyst-executor` process that
    cross-compiled objects are dispatched to.
    [(#3082)](https://github.com/PennyLaneAI/catalyst/pull/3082)
    [(#3119)](https://github.com/PennyLaneAI/catalyst/pull/3119)


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

* The `ResourceAnalysis` pass now reports each loop body and each subroutine as its own entry
  instead of folding their gate counts into the caller. Loops with constant bounds appear as `for_loop_<N>`
  with their trip count. Loops with dynamic bounds appear as `dyn_for_loop_<N>` with a stable
  identifier, and totals across the call graph are computed on demand.
  [(#2782)](https://github.com/PennyLaneAI/catalyst/pull/2782)
  [(#2900)](https://github.com/PennyLaneAI/catalyst/pull/2900)

* The `ResourceAnalysis` pass now supports IR in reference semantics natively, rather than requiring a conversion step.
  [(#2923)](https://github.com/PennyLaneAI/catalyst/pull/2923)

* The `resource-analysis` pass JSON output now includes `depth` for worst-case PBC layer depth
  (`any_commuting_depth` / `qubit_disjoint_depth`) per function and lifted loop entry, including
  commuting vs disjoint-qubit layer grouping and worst-case depth across ``scf.if`` /
  ``scf.index_switch`` branches and statically-bounded ``scf.for`` loops.
  [(#2863)](https://github.com/PennyLaneAI/catalyst/pull/2863)
  [(#2876)](https://github.com/PennyLaneAI/catalyst/pull/2876)
  [(#2877)](https://github.com/PennyLaneAI/catalyst/pull/2877)
  [(#2879)](https://github.com/PennyLaneAI/catalyst/pull/2879)
  [(#2884)](https://github.com/PennyLaneAI/catalyst/pull/2884)
  [(#2967)](https://github.com/PennyLaneAI/catalyst/pull/2967)

* The `resource-analysis` pass now supports pluggable resource metrics through the
  `ResourceResultExtension`/`ResourceAnalysisExtension` interface and a self-registering global
  registry. Dialects and plugins can contribute additional per-function resource data
  (such as PBC circuit depth) without modifying the core analysis.

  A metric is added by defining a value object, an analysis that fills it, and
  registering that analysis with the global registry:
  ```cpp
  // 1. Per-function value object; shows up in the JSON under name().
  class TCountExtension : public ResourceResultExtension {
   public:
    llvm::StringRef name() const override { return "t_count"; }
    llvm::json::Value toJson() const override { return tCount; }
    // Accumulated in collect(), so define how it combines and scales.
    void mergeWith(const ResourceResultExtension &other, MergeMethod method) override {
      tCount += static_cast<const TCountExtension &>(other).tCount;  // use `method` for max/min
    }
    void multiplyBy(double factor) override { tCount *= factor; }
    double tCount = 0;
  };

  // 2. Analysis that fills it, one operation at a time.
  class TCountAnalysis : public ResourceAnalysisExtensionOf<TCountExtension> {
   public:
    llvm::StringRef name() const override { return "t_count"; }
   protected:
    void collect(mlir::Operation *op, TCountExtension &ext, bool isAdjoint) override {
      if (isTGate(op)) ext.tCount += 1;
    }
  };

  // 3. Self-register from your dialect or plugin (no core changes needed).
  REGISTER_RESOURCE_ANALYSIS_EXTENSION(std::make_unique<TCountAnalysis>());
  ```
  The metric then appears under its `name()` key (`"t_count"`) in each function's JSON
  output. Override `analyze(Region&, Ext&, bool)` instead of, or alongside, `collect` to
  compute a metric per region rather than per operation (as `PBCDepthExtension` does).
  [(#3070)](https://github.com/PennyLaneAI/catalyst/pull/3070)

* The `resource-analysis` pass now uses a single JSON serializer owned by `ResourceResult`, removing
  duplicate serialization logic and keeping its output consistent.
  [(#3007)](https://github.com/PennyLaneAI/catalyst/issues/3007)

* The `ResourceAnalysis` pass now counts quantum, measurement, and allocation
  ops through dialect-agnostic MLIR OpInterfaces instead of hard-coded check.
  New dialects can opt in by implementing these interfaces without changing
  the analysis.
  [(#3025)](https://github.com/PennyLaneAI/catalyst/pull/3025)

* The `resource-analysis` pass JSON output has been standardized into a nested schema.
  Gate counts are grouped by wire count under `quantum_operations`, function metadata
  lives under `metadata`, qubit counts under `num_qubits`, static and dynamic calls
  under `function_calls.static` / `function_calls.dynamic`, measurement processes under
  `measurement_processes`, and pluggable metrics under `extended_fields`.
  [(#3076)](https://github.com/PennyLaneAI/catalyst/pull/3076)

* The `--adjoint-lowering` pass no longer turns statically bounded for loops into
  dynamically bounded ones. In this way they remain analyzable by functionality like `qp.specs`.
  [(#2959)](https://github.com/PennyLaneAI/catalyst/issues/2959)

* PPRs and PPMs can now be lowered properly into MLIR directly in the non-capture workflow.
  [(#2816)](https://github.com/PennyLaneAI/catalyst/pull/2816)

* The ``--partition-layers`` pass now supports a ``disjoint-qubit`` option to group PBC ops
  into the same layer only when they act on disjoint qubits. By default, commuting ops on
  overlapping qubits may still be merged into one layer.
  [(#2858)](https://github.com/PennyLaneAI/catalyst/pull/2858)

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

* Added ``CZ`` support to ``to-ppr`` pass.
  [(#3009)](https://github.com/PennyLaneAI/catalyst/pull/3009)

<h3>Breaking changes 💔</h3>

* Removes :func:`~.passes.ppm_specs` and the ``--ppm-specs`` MLIR pass. Use :func:`~.specs` and
  the ``ResourceAnalysis`` pass instead for PPR/PPM resource counts and PBC layer depth
  (``any_commuting_depth`` / ``qubit_disjoint_depth``).
  [(#3081)](https://github.com/PennyLaneAI/catalyst/pull/3081)

* Removes the non-graph decomposition fallback when `capture=True` is enabled.
  [(#3058)](https://github.com/PennyLaneAI/catalyst/pull/3058/)

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

* Fixed a bug where an executor's SSH connection multiplexing was silently disabled on macOS,
  making every remote operation pay a fresh authentication handshake. The control socket went in
  the system temp dir, which macOS puts under a per-user `/var/folders/...` path long enough to
  overrun the 104-byte `sun_path` limit on its own. Sockets now live in `~/.catalyst/cm`, created
  `0700`, which is both short enough and reachable only by its owner. Where no such directory can
  be made, multiplexing is skipped rather than falling back to a world-writable one.
  [(#3110)](https://github.com/PennyLaneAI/catalyst/pull/3110)

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

* Fixed the assembly format for `quantum.adjoint` when it has no quantum operands/results.
  [(#2938)](https://github.com/PennyLaneAI/catalyst/pull/2938)

<h3>Internal changes ⚙️</h3>

* Update calls to `GlobalPhase` to no longer use the `wires` argument.
  [(#3108)](https://github.com/PennyLaneAI/catalyst/pull/3108)
  
* A GPU CI workflow runs the runtime transport tests on the `single-gpu-x64` runner, gated by
  the `gpu` label.
  [(#3113)](https://github.com/PennyLaneAI/catalyst/pull/3113)

* The `--to-ppr` and `--ppm-compilation` passes now run `--symbol-dce` at the beginning,
  to eliminate unnecessary decomposition rules that might contain gates that cannot be converted to PPRs.
  [(#3125)](https://github.com/PennyLaneAI/catalyst/pull/3125)
  [(#3135)](https://github.com/PennyLaneAI/catalyst/pull/3135)

* The `dim` argument of the `quantum.pcphase` operation has been changed to a static integer attribute
  (previously a dynamic float operand). This allows, among other things, the decomposition graph to
  distinguish pcphase gates with different `dim` values, since they need different decomposition rules.
  [(#3034)](https://github.com/PennyLaneAI/catalyst/pull/3034)

* The `cond` PLxPR primitive's lowering rule no longer expects a `True` Literal for the predicate
  of the default else branch.
  [(#3018)](https://github.com/PennyLaneAI/catalyst/pull/3018)

* `ResourceAnalysis` can now optionally count `DecomposableGate` operations by their full graph
  operation ID. Decomposition-rule resource generation enables this detailed mode so that resource
  annotations preserve parameter types, wire counts, static data, and operator UIDs used for graph
  matching.
  [(#3102)](https://github.com/PennyLaneAI/catalyst/pull/3102)

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
Nikhil Sreekumar,
Paul Haochen Wang,
Jake Zaia,
Hongsheng Zheng.
