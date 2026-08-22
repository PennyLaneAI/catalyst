# Phase Folding — Starter Guide

A working document for anyone picking up this project.

This is an MLIR compiler pass inside Catalyst. It reduces T-count (and other Z-rotations) on hybrid quantum–classical programs by grouping phase gates that rotate the same parity, then merging them.

The classical circuit case is the standard phase-polynomial / T-count optimization. The hybrid case is the interesting part: `if` and `for` change which parities are still identifiable, so the pass tracks both a phase map and an affine description of the computational-basis state.

---

## 1. What to read first

In this order:

1. This guide.
2. `mlir/lib/QRef/Transforms/phase_folding.cpp` — the MLIR pass: walk the IR, analyze, then rewrite.
3. `mlir/lib/QRef/Transforms/PhaseFolding/ProgramAbstraction.cpp` — how a gate or a control-flow region updates the abstraction.
4. `mlir/lib/QRef/Transforms/PhaseFolding/RegionSummary.cpp` — how `if` / `for` / procedure bodies are summarized.
5. `frontend/catalyst/passes/builtin_passes.py` — the Python decorator that schedules the pass.

Classic background (straight-line Clifford+T circuits):

- Amy, Maslov, Mosca, Roetteler — phase polynomials / T-count reduction. The pass file cites https://arxiv.org/pdf/1303.2042.
- Matthew Amy’s work on CNOT-dihedral circuits: a Z-rotation on a wire is a rotation of a linear function (parity) of the computational-basis bits. Two rotations of the same parity can be added.

This project’s extra idea: when the program has branches and loops, the state is no longer a single affine transform. It becomes an affine relation between pre-state and post-state. Phases that are no longer uniquely determined get “orphaned” and can only be folded among themselves, not with later gates.

You do not need to understand every GF(2) schema class on day one. Start from “parity of a phase gate” and “what H and if do to that parity.”

---

## 2. Mental model in one page

Every computational-basis bit of qubit `i` is tracked as an affine function over GF(2):

    x'_i  =  (linear combination of input bits X and auxiliaries Y)  ⊕  c

A Z-rotation (T, S, Z, RZ, or the Z part of Y) on qubit `i` does not change that bit. It contributes a phase `e^{iθ}` whenever the current parity of that bit is 1.

So the pass keeps two things together, called a ProgramAbstraction:

- Affine transform: the current X / Y / c matrix (one row per qubit).
- Phase map: parity → list of phase-gate IDs that rotate that parity.

Folding is then: for each parity, add the angles (with a sign coming from the affine constant c), keep one gate, delete the rest.

Control flow is where this stops being a textbook circuit pass:

- `scf.if`: analyze then-branch and else-branch separately, join their affine relations, keep both phase maps.
- `scf.for`: analyze the body once, then take the Kleene star of the body’s relation (identity ∪ R ∪ R² ∪ … until a fixpoint). That over-approximates “the loop ran any number of times.”
- Merging a region back into the parent: use the parent’s current state as a precondition, drop phases that that precondition cannot justify, push the parent state through the region relation, then add the surviving region phases into the parent.

Soundness is the priority. When the analysis is unsure (Hadamard, unknown gate, loop, both sides of a branch), it allocates fresh auxiliary variables or orphans a parity rather than incorrectly merging two different rotations.

---

## 3. Where the code lives

All of the C++ lives under Catalyst’s QRef transforms:

    mlir/lib/QRef/Transforms/phase_folding.cpp
    mlir/lib/QRef/Transforms/PhaseFolding/
    mlir/include/QRef/Transforms/Passes.td          # pass registration + options
    mlir/lib/QRef/Transforms/CMakeLists.txt         # build list
    frontend/catalyst/passes/builtin_passes.py      # @phase_folding decorator

PhaseFolding/ is a small linear-algebra library plus the program abstraction. phase_folding.cpp is the only file that talks to MLIR ops.

    PhaseFolding/
      Gate.hpp                 Gate enum, angles, arity
      GateBundle.hpp/.cpp      Phase-gate IDs grouped by affine constant 0 vs 1
      Parity.hpp/.cpp          Packed GF(2) bitvectors (64-bit blocks)
      BinaryMatrix.hpp/.cpp    Rows of parities; REF / RREF
      AffineSchema.hpp/.cpp    Column layouts: pre / post / aux / constant
      AffineBase.hpp           Matrix + schema; project-out, equality
      AffineTransform.hpp/.cpp State as x' = A x ⊕ B y ⊕ c
      AffineRelation.hpp/.cpp  Constraints on (x', x, y, c); join / meet / compose / star
      PhaseAbstraction.hpp/.cpp   parity → GateBundle, plus orphan list
      ProgramAbstraction.hpp/.cpp Transform + phases; applyGate / applySummary
      RegionSummary.hpp/.cpp   Summarize a nested region

Namespace: `catalyst::phase_folding`.

The pass itself is `catalyst::qref::PhaseFoldingPass`, MLIR name `phase-folding`.

---

## 4. How it is invoked

The Python decorator does not run the pass in isolation. It inserts a four-pass pipeline:

1. `convert-to-reference-semantics` — value-semantics quantum IR → QRef (reference qubits).
2. `cse`
3. `phase-folding`
4. `convert-to-value-semantics` — back to value semantics.

That conversion is required. The analyzer walks `qref.custom`, `qref.get`, `qref.alloc_qubit`, `scf.if`, `scf.for`. It does not run on the value-semantics `quantum` dialect.

Typical usage:

    from catalyst import qjit
    from catalyst.passes import phase_folding
    import pennylane as qp

    @qjit(keep_intermediate=2)
    @phase_folding(report_stats=True, trace_abstraction=True)
    @qp.qnode(qp.device("lightning.qubit", wires=2))
    def circuit(x: float):
        qp.T(0)
        if x > 1.4:
            qp.CNOT([0, 1])
        qp.adjoint(qp.T(0))
        return qp.probs()

`keep_intermediate=2` dumps MLIR after each compilation stage into a directory named after the function, e.g. `circuit/1_QuantumCompilationStage/`. The file whose name contains `phase-folding` is the IR after this pass.

Pass options (also in Passes.td):

- `report_stats` / `--report-stats`
  Writes `phase_folding_report_<module>.txt` in the current working directory: gate counts before/after, plus the final program abstraction.
- `trace_abstraction` / `--trace-abstraction`
  Writes `phase_folding_trace_<module>.txt`: the abstraction after every analyzed gate and after every region merge. This is the main debugging tool.

The pass currently also prints `Hello phase-folding world!` to stdout. That is leftover debug output.

Scratch / example programs used during development (not a formal test suite):

- `test_phase_folding.py` — many commented circuits (if, loop, SWAP, reset, RUS, nested). Uncomment one, run it.
- `grover_5.py` — larger example; a saved report is `phase_folding_report_module_grover_5.txt`.
- `PhaseFoldingTest.mlir`, `PhaseFoldingQRefTest.mlir` — hand-written MLIR snippets.

There is not yet a FileCheck / pytest suite dedicated to this pass. Treat the Python files as the current way to try ideas.

---

## 5. How a run is structured

`PhaseFoldingPass::runOnOperation` is short:

1. Optionally open a trace file.
2. For every non-external `func.func` in the module, run `PhaseAnalyzer::analyzeFuncOp`.
3. The function marked `quantum.node` is treated as the main program; its abstraction is stored on the plan.
4. `PhaseFolder::foldPhases` rewrites the IR using that main abstraction.
5. Optionally write the stats report.

So analysis is a forward walk that only builds data structures. Rewriting happens afterwards, in one shot, by GateID.

### 5.1 WireTable

QRef qubits are SSA values (`qref.get` from a register, or `qref.alloc_qubit`). Internally the math wants dense wire indices `0 .. n-1`.

`WireTable` assigns a stable integer to each distinct `(register, index)` get, and to each allocated qubit. Those integers are the row indices of the affine transform.

If a gate mentions a wire the matrix does not yet have, `ProgramAbstraction::areWiresInBound` grows the matrix.

### 5.2 PhaseAnalyzer

Walks a block in order. Relevant ops:

    qref.custom            applyGate
    qref.set_basis_state   prepareQubit (constant |0>/|1| only)
    scf.if                 analyze both regions, summarize as Conditional, merge
    scf.for                analyze body, summarize as Loop, merge
    func.call              currently a no-op (stub exists)
    measure / alloc / dealloc / yield / global_phase   ignored

`extractCliffTGate` maps an op name to the internal `Gate` enum. Controlled gates are special:

- Controlled phase (CRZ, CT, …) is treated as `I` for the state (the bits do not change) but the extra phase is not tracked. Comment in the code: this is nonlinear (`xy = x + y − (x ⊕ y)`) and is future work.
- Other controls become `U` (uninterpreted): those qubits’ rows are replaced by fresh auxiliaries.

Phase gates get a `GateID` (index into `plan.phaseOps`) so folding can find the original `qref.custom` later.

After each function, a `RegionSummary(Procedure, …)` is stored in `procedureSummaries`. Call-sites do not use it yet. `applySummary` consumes the summary, so a later call-site integration must copy before merging.

### 5.3 PhaseFolder

For every bundle in the main phase map:

- If the parity is the zero (trivial) vector, those rotations are global phases on `|0…0>`-style constants and the corresponding gates are deleted.
- If a bundle has more than one gate, sum the angles, rewrite the first “merge target” gate, delete the others.
- Y is not a pure phase: removing its phase leaves an X.

Angle rules:

    Z → π,  S → π/2,  T → π/4,  RZ → the constant float if present else 0
    adjoint flips the sign
    net angle is reduced into [−π, π]
    0 → erase,  π → Z,  π/2 → S,  π/4 → T,  otherwise RZ

The merge target is the first ID in the affine-0 list if nonempty, otherwise the first affine-1 ID. If the target came from the affine-1 list, the summed angle is negated so the kept gate has the right sign.

---

## 6. Core types

### 6.1 Gate

    I, H, X, Y, Z, S, T, RZ, CNOT, SWAP, U, GP

Phase gates: Z, S, T, RZ, Y.

`U` means “this op destroys affine knowledge of these wires.” `I` means “state unchanged, and we are not tracking a phase either.”

### 6.2 Parity

A bitvector over GF(2), stored as `SmallVector<uint64_t>` blocks of 64 bits.

Addition is XOR. Used as:

- one row of a binary matrix (one qubit’s affine expression, or one constraint)
- the key in the phase map

`BitLocation` is `(block, bit)`. Location `(0,0)` is reserved for the affine constant `c`. New variables are allocated from the next free location.

A trivial parity is the all-zero vector. An unsat parity is “0 = 1”, represented as only the constant bit set.

`Parity` can be a DenseMap key (empty/tombstone states are tagged separately from the bits).

### 6.3 GateBundle

The value stored at a parity:

    zeroAffineGates   IDs whose rotation applies when c = 0 for that row
    oneAffineGates    IDs whose rotation applies when c = 1

Why two lists: a phase on `x ⊕ 1` is the same as a negated phase on `x`. When a later constraint flips the constant bit, the bundle swaps the two lists instead of rewriting every ID.

`[0: (3, 7) __ 1: (12)]` in a dump means gates 3 and 7 on the even side, gate 12 on the odd side.

### 6.4 BinaryMatrix

A list of `Parity` rows. CNOT is `addRowToRow`. SWAP is `swapRows`. X flips the constant bit of a row. H / U replace a row with a fresh basis vector on a new auxiliary.

`normalize` computes RREF over GF(2) and drops zero rows. `toREF` is used when reducing a single parity against a constraint system (the “tracking row” is the parity being reduced).

Column order is not “left to right in memory.” It is whatever the schema’s `getOrder()` says. That is why schemas exist.

### 6.5 Schemas

A schema is a named layout of bit locations:

    TransformSchema    columns:  preVars (inputs X) , auxVars (Y) , affVal (c)
    RelationSchema     columns:  postVars (X') , preVars (X) , auxVars (Y) , affVal

Temporary layouts for algebra:

    MeetSchema, JoinSchema, CompositionSchema, PropagateSchema

They exist so join/compose can put two relations’ variables side by side, row-reduce, and project the intermediate columns. You almost never construct these outside AffineRelation.cpp.

Printing:

    Affine transform dump:   X Y | c     then one row per qubit
    Affine relation dump:    X' X Y | c  then one row per constraint

### 6.6 AffineTransform vs AffineRelation

AffineTransform: a function. Row `i` is qubit `i`’s current bit as a formula in the original inputs and auxiliaries.

How gates update it:

    X on i        flip c on row i
    CNOT c→t      row t  ⊕=  row c
    SWAP i,j      swap rows
    H on i        row i becomes a fresh auxiliary  (Z-parity is no longer affine in X)
    U on wires    each of those rows becomes a fresh auxiliary
    prepare |b⟩   row becomes the constant b

AffineRelation: a set of affine constraints relating pre-state X and post-state X'. Any transform can be viewed as the relation `X' = T(X)`. Relations are the right object for control flow because a branch is “T_then or T_else”, which is a join of relations, not a join of functions.

Factories:

    Identity(n)   X' = X
    Trivial(n)    no constraints (top / anything goes) — used as Kleene-star accumulator start
    Unsat(n)      0 = 1

Operations (all over GF(2) affine relations):

    meet       intersection of constraint sets
    join       union, implemented by embedding both into a combined schema and projecting
    compose    R ; S   (R then S)
    kleeneStar 1 ∨ R ∨ R² ∨ … until the join stops changing
    propagateThrough   push a known pre-state through a region relation
    solveRelation      turn a (hopefully functional) relation back into a transform
    reduce             rewrite one phase-parity into a canonical form under the constraints

If you change join, star, or reduce, you are changing the meaning of hybrid folding. Test `if` and `for` examples when you touch these.

### 6.7 PhaseAbstraction

    activeBundles   DenseMap<Parity, GateBundle>   still identified with a linear form
    orphanBundles   vector<GateBundle>             no longer identified; fold only inside the bundle

`insertContributor`: add a gate to the bundle for that parity (merge if the parity already exists).

`normalizeByCond`: reduce every active parity against a relation; if the constant bit flips, swap 0/1 lists.

`nullifyByPrecond`: reduce against a precondition, then orphan every bundle whose remaining parity is non-trivial. Meaning: “this region phase only fires for some incoming states, not all, so it must not be merged with parent phases.”

`projectOutAuxVars`: used when summarizing a region — drop dependence on auxiliaries that will not be in the relation.

### 6.8 ProgramAbstraction

    phases            PhaseAbstraction
    stateTransform    AffineTransform

`applyGate` is the straight-line case. For RZ/T/S/Z it takes the current row of the target wire, strips `c`, and inserts the gate ID under that parity.

Y is expanded as X then RZ (or RZ then X for adjoint), plus an ignored global phase of ±i.

`applySummary` is the hybrid case (see next section).

### 6.9 RegionSummary

Built from one or two ProgramAbstractions:

    Conditional   then + optional else; affineRel = join(then, else)
    Loop          body; affineRel = kleeneStar(body)
    Procedure     body; project aux, orphan non-trivial phases (conservative)

It also stores the then-phases (always) and else-phases (conditionals). After construction, the original transforms have been moved into the relation.

---

## 7. Merging a nested region (`applySummary`)

This is the heart of the hybrid algorithm. When the analyzer finishes an `if` or `for`, it does:

    AffineRelation precondition = parent.stateTransform;   // “what we know on entry”
    summary.nullifyPhasesUnder(precondition);
    precondition.propagateThrough(summary.affineRel);
    parent.normalizePhasesUnder(precondition);
    parent.stateTransform = precondition.solveRelation();
    summary.accumulatePhasesInto(parent.phases);

In words:

1. Parent state becomes the precondition relation `X' = T_parent(X)`.
2. Region phases that are not forced by that precondition are orphaned (they are not safe to combine with anything outside).
3. Push the parent state through the region’s (joined / starred) relation. The result is a relation describing possible states after the region.
4. Rewrite the parent’s existing phases into that post-state basis.
5. Solve back to a transform (best affine approximation of the post-state).
6. Add the (already nullified) region phases into the parent map. Same parities merge; that is the actual fold across a branch/loop boundary when it is safe.

If a phase inside a loop depends on “how many times we iterated,” it should end up orphaned or on an auxiliary, not merged with a T after the loop.

---

## 8. A small straight-line example

Circuit:

    T(q0)
    CNOT(q0, q1)
    T(q1)
    CNOT(q0, q1)
    T†(q0)

Start: identity, two qubits.

    T(q0)          parity of q0 is `x0`. Bundle {T0} at `x0`.
    CNOT           q1 := q0 ⊕ q1
    T(q1)          parity of q1 is `x0 ⊕ x1`. Bundle {T1} at `x0 ⊕ x1`.
    CNOT           q1 restored
    T†(q0)         parity of q0 is `x0` again. Bundle {T0, T†} at `x0`.

Net angle at `x0` is π/4 − π/4 = 0 → both T and T† can be deleted. The middle T stays, on a different parity.

That is all phase folding is, until control flow or H appears.

Hadamard on a wire replaces that row with a new Y variable. Later T on that wire is a rotation of an auxiliary, which generally cannot cancel with a T from before the H.

---

## 9. A small hybrid example

    T(q0)
    if cond:
        CNOT(q0, q1)
    T†(q0)

CNOT does not change the control bit, so both branches leave `q0` as `x0`. The join of the two relations still says `q0' = q0`. The T and T† share parity `x0` and cancel, whether or not the CNOT ran.

If the then-branch did `H(q0)` instead, the join would no longer give a unique affine expression for `q0` after the `if`. The T inside/after would not be identified with the T before, which is the conservative (correct) outcome.

---

## 10. How to read dumps

Enable `trace_abstraction=True` and open `phase_folding_trace_<module>.txt`.

You will see nested steps:

    --- step 1 [enter scf.if] ---
    --- step 1.1 [gate T on wire 0] ---
    .Phase abstraction:
    10  -> [0: (0) __ 1: ()]
    .Affine transformation:
    X Y | c
    10  | 0
    01  | 0
    --- step 1.2 [then branch-exit] ---
    --- step 1.3 [scf.if region-summary] ---
    --- step 1.4 [scf.if parent-after-merge] ---
    --- step 1.5 [exit scf.if] ---

Phase lines look like:

    <pre bits> <aux bits> -> [0: (ids) __ 1: (ids)]

Left of the arrow: the parity in schema order (inputs, then auxiliaries). Right: which GateIDs sit on that parity.

    Unsat -> [0: (180) __ 1: ()], ...

means orphaned bundles. They will still be folded internally if an orphan bundle contains more than one ID.

The stats report (`report_stats=True`) is the same abstraction after analysis, plus counts:

    T: initial-> 1722,  final-> 1402. difference-> -320
    S: initial-> 0,  final-> 130. difference-> 130

Negative T and positive S is common: eight T’s on one parity become an S, etc.

GateIDs are indices into the analyzer’s `phaseOps` vector, in the order phase ops were first seen. They are not MLIR result numbers.

---

## 11. What is implemented vs not

Works today (with the usual over-approximation):

- Straight-line Clifford+T+RZ, including CNOT and SWAP
- Y (as X plus a phase)
- `scf.if` / `scf.for`, including nesting
- Mid-circuit `set_basis_state` with a constant tensor
- Growing the qubit set when new wires appear
- Folding that rewrites T/S/Z/RZ and turns Y into X when the phase is removed

Known gaps / sharp edges (from comments and stubs in the code):

- `func.call` is not applied. Procedure summaries are computed and stored, then ignored.
- Multiple `quantum.node` functions: the last one analyzed overwrites `mainProgramAbst`.
- Controlled phases are not tracked.
- Controlled-X is not recognized as CNOT (would need to pass the control wire into `applyGate`).
- Measurements do not update the abstraction (no collapse / no classical feedback into the affine state).
- Dynamic `set_basis_state` asserts.
- Dynamic RZ parameters are treated as angle 0 when folding, so they can be merged incorrectly relative to a true symbolic angle. Constant RZ is fine.
- GlobalPhase is ignored.
- Uninterpreted ops (`U`) forget those qubits’ parities; they do not try a more precise model.
- Loop analysis does not use trip counts even when they are constant; it always uses Kleene star.
- Debug `Hello phase-folding world!` is still in `runOnOperation`.
- No dedicated lit/pytest suite.

Soundness-related caution: the Kleene-star / join over-approximation is meant to never fold two phases that might not share a parity at runtime. If you ever see a fold that changes program semantics, look at `joinWith`, `kleeneStar`, `nullifyByPrecond`, and `reduce` first.

---

## 12. Suggested onboarding path

Spend a day in this order. Do not start by refactoring schemas.

1. Run the decorator on a 3-line circuit (`T`, `CNOT`, `T†`) with both flags on. Read the trace. Confirm the two T’s share a parity and the stats show T decreasing.

2. Uncomment `circ_if_simple` in `test_phase_folding.py`. Compare the trace with and without the `CNOT` in the then-branch.

3. Uncomment `circ_loop_simple` and `circ_loop_swap`. SWAP in a loop is a good test that join/star is actually mixing wires.

4. Read `applyGate` and `applyGateRZ` until you can explain why the constant bit is stripped before the parity is used as a map key.

5. Read `applySummary` and `RegionSummary::summarizeCond` / `summarizeLoop` with the trace for an `if` open beside the code.

6. Only then open `AffineRelation.cpp`. Follow one `joinWith` on a 1- or 2-qubit example on paper (columns X', X, c).

After that you can pick a real task. Natural next pieces, if nobody has claimed them:

- Wire `handleCallOp` using copies of `procedureSummaries`.
- Treat single-control X as CNOT.
- Track measurements / resets more precisely.
- Replace the stdout hello-world; maybe log under `LLVM_DEBUG`.
- Add FileCheck tests from the small Python examples (dump QRef MLIR before/after).
- Controlled-phase via the `xy = x + y − (x ⊕ y)` expansion (called out in `extractCliffTGate`).

---

## 13. Build / debug notes

The library is `qref-transforms` (`CMakeLists.txt` in the same folder). After a C++ change, rebuild Catalyst as you usually do, then re-run the Python `qjit` script. The pass is linked into the compiler, not loaded as a Python plugin.

If the abstraction looks wrong:

1. Turn on `trace_abstraction` and find the first step where a parity is not what you expect.
2. For gates, the bug is usually in `applyGate*` or `extractCliffTGate`.
3. For `if`/`for`, dump the RegionSummary (the trace already does) and check whether the relation still implies the parity you wanted. If it does not, folding is supposed to refuse.
4. `LLVM_DEBUG` type is `phase_folding` (`#define DEBUG_TYPE "phase_folding"`), but most useful output is currently the custom trace file, not LLVM debug.

If folding looks wrong but the abstraction looks right: the bug is in `PhaseFolder` (angle sum, adjoint sign, Y → X, or merge-target choice).

If the compiler crashes on `Untracked operand wire`, the WireTable missed a qubit definition (block argument, function argument, or an op other than `get` / `alloc_qubit`).

---

## 14. Glossary

Parity
Linear boolean function of computational-basis bits, plus a constant. The thing a Z-rotation actually rotates.

Phase polynomial
The collection of (parity → angle) terms. This pass stores contributors rather than a numeric polynomial, then sums at fold time.

Auxiliary variable (Y)
A bit the analysis introduced because the true bit is no longer an affine function of the inputs (Hadamard, unknown gate, projected-out join intermediates).

Orphan bundle
Phase gates whose parity could not be kept as a linear form on the live variables. They may still cancel with each other.

Precondition / postcondition
Affine relation describing possible states before / after a region.

Kleene star
Over-approximation of “this loop body ran 0, 1, 2, … times.”

QRef
Catalyst’s reference-semantics quantum dialect: qubits are mutated in place, which makes SSA-insensitive analysis on a block straightforward.

GateID
Integer index of a phase `qref.custom` in `PhaseFoldingPlan::phaseOps`. Same role as location `ℓ` in the thesis / Feynman-style notes.

---

## 15. File-to-question cheat sheet

“How does a T get recorded?”
`PhaseAnalyzer::handleCustomOp` → `ProgramAbstraction::applyGate` → `applyGateRZ` → `PhaseAbstraction::insertContributor`

“How does an if get merged?”
`handleIfOp` → `RegionSummary(Conditional, then, else)` → `joinWith` → `applySummary`

“How does a for get merged?”
`handleForOp` → `RegionSummary(Loop, body)` → `applyKleeneStar` → `applySummary`

“Where do gates disappear?”
`PhaseFolder::foldPhases` → `foldBundle` / `removePhaseOp`

“Where is GF(2) row reduction?”
`BinaryMatrix::computeEchelonForm` (REF and RREF)

“Where are column layouts defined?”
`AffineSchema.hpp`

“How do I dump IR after the pass?”
`@qjit(keep_intermediate=2)` and look under `<fn>/1_QuantumCompilationStage/`
