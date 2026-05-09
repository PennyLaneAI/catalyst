// Copyright 2023 Xanadu Quantum Technologies Inc.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef QUANTUM_PASSES
#define QUANTUM_PASSES

include "mlir/Pass/PassBase.td"

def QuantumConversionPass : Pass<"convert-quantum-to-llvm"> {
    let summary = "Perform a dialect conversion from Quantum to LLVM (QIR).";

    let options = [
        Option<
            "useArrayBackedRegisters",
            "use-array-backed-registers",
            "bool",
            default="false",
            desc="Use the array-backed-registers conversion pattern for quantum.insert ops."
        >
    ];

    let dependentDialects = ["LLVM::LLVMDialect"];
}

def EmitCatalystPyInterfacePass : Pass<"emit-catalyst-py-interface"> {
    let summary = "Emit catalyst python's default interface.";

    let dependentDialects = ["LLVM::LLVMDialect"];
}

def CopyGlobalMemRefPass : Pass<"cp-global-memref"> {
    let summary = "Copy global memrefs before returning from C interface.";

    let dependentDialects = [
        "memref::MemRefDialect",
        "func::FuncDialect",
        "scf::SCFDialect",
        "LLVM::LLVMDialect"
    ];
}

def AdjointLoweringPass : Pass<"adjoint-lowering"> {
    let summary = "Lower adjoint regions containing a single quantum operations.";

    let dependentDialects = [
        "catalyst::CatalystDialect",
        "index::IndexDialect",
        "tensor::TensorDialect",
        "complex::ComplexDialect",
        "scf::SCFDialect"
    ];
}

def SplitMultipleTapesPass : Pass<"split-multiple-tapes"> {
    let summary = "Given a qnode containing multiple tapes, split each tape into its own function.";
}

def SplitNonCommutingPass : Pass<"split-non-commuting", "mlir::ModuleOp"> {
    let summary = "Split quantum functions non-commuting observables into multiple executions.";

    let options = [
        Option<
            "groupingStrategy",
            "grouping-strategy",
            "std::string",
            /*default=*/"\"\"",
            "Grouping strategy for observables. "
            "\"\" (default) assigns each observable its own group. "
            "\"wires\" groups observables on non-overlapping wires."
        >
    ];

    let dependentDialects = ["arith::ArithDialect"];
}

def SplitToSingleTermsPass : Pass<"split-to-single-terms", "mlir::ModuleOp"> {
    let summary = "Split quantum functions into single-term observables.";
}

// ----- Quantum circuit transformation passes begin ----- //
// For example, automatic compiler peephole opts, etc.

def IonsDecompositionPass : Pass<"ions-decomposition"> {
    let summary = "Decompose the gates to the set {RX, RY, MS}";
}

def CancelInversesPass : Pass<"cancel-inverses"> {
    let summary = "Perform removal of chained operations that are inverses.";
}

def GridsynthPass : Pass<"gridsynth"> {
    let summary = "Perform Ross-Selinger/Gridsynth decomposition.";

    let options = [Option<
        /*C++ var name=*/"epsilon",
        /*CLI name=*/"epsilon",
        /*Type=*/"double",
        /*Default=*/"1e-4",
        /*Description=*/"Maximum error tolerance for per-gate discretization.">,
        Option<
        /*C++ var name=*/"pprBasis",
        /*CLI name=*/"ppr-basis",
        /*Type=*/"bool",
        /*Default=*/"false",
        /*Description=*/desc="Whether to decompose into the PPR basis set instead of the standard Clifford+T basis set.">,];

        let dependentDialects = [
        "memref::MemRefDialect",
        "scf::SCFDialect",
        "catalyst::pbc::PBCDialect",
    ];
}

def MergeRotationsPass : Pass<"merge-rotations"> {
    let summary = "Perform merging of chained rotation gates about the same axis.";

    let dependentDialects = ["math::MathDialect"];
}

def GraphDecompositionPass : Pass<"graph-decomposition", "mlir::ModuleOp"> {
    let summary = "Decompose gates using an MLIR-native graph-based framework.";

    let options = [
        ListOption<
            /*C++ name*/"targetGateSetOption",
            /*CLI name*/"gate-set",
            /*Type*/"std::string",
            /*Description*/"The accepted gates to decompose to.">,
        ListOption<
            /*C++ name*/"fixedDecompsOption",
            /*CLI name*/"fixed-decomps",
            /*Type*/"std::string",
            /*Description*/"Maps an operator to a decomposition rule that will be applied.">,
        ListOption<
            /*C++ name*/"altDecompsOption",
            /*CLI name*/"alt-decomps",
            /*Type*/"std::string",
            /*Description*/
            "Maps an operator to a `|` separated list of alternative decomposition "
            "rules that will be considered alongside any built-in rules for the operator.">,
        Option<
            /*C++ name*/"bytecodeRulesFile",
            /*CLI name*/"bytecode-rules",
            /*Type*/"std::string",
            /*Default*/[{ "./frontend/catalyst/resources/decomposition_rules.mlirbc" }],
            /*Description*/"A path to a bytecode file of compiled decomposition rules.">
    ];

    let dependentDialects = [
        "memref::MemRefDialect",
        "func::FuncDialect",
        "scf::SCFDialect",
        "LLVM::LLVMDialect",
        "stablehlo::StablehloDialect",
        "tensor::TensorDialect",
    ];
}

def DecomposeLoweringPass : Pass<"decompose-lowering"> {
    let summary = "Replace quantum operations with compiled decomposition rules.";
}

def DisentangleCNOTPass : Pass<"disentangle-cnot"> {
    let summary = "Replace a CNOT gate with two single qubit gates whenever possible.";

    let options = [
    Option<"EmitFSMStateRemark", "emit-FSM-state-remark",
           "bool", /*default=*/"false",
           "Whether to emit the state analysis result from the simple states propagation FSM onto the gate operations.">,
    ];
}

def DisentangleSWAPPass : Pass<"disentangle-swap"> {
    let summary = "Replace a SWAP gate with single qubit gates and a shorter SWAPZ gates whenever possible.";

    let options = [
    Option<"EmitFSMStateRemark", "emit-FSM-state-remark",
           "bool", /*default=*/"false",
           "Whether to emit the state analysis result from the simple states propagation FSM onto the gate operations.">,
    ];
}

def LoopBoundaryOptimizationPass : Pass<"loop-boundary"> {
    let summary = "Perform loop boundary optimization to eliminate the redundancy of operations on loop boundary.";
}


def MergeUnitaryGatesPass : Pass<"merge-unitary-gates"> {
    let summary = "Perform merging of chained unitary gates.";

    // let constructor = "catalyst::createMergeUnitaryGatesPass()";
}

def DynamicOneShotPass : Pass<"dynamic-one-shot", "mlir::ModuleOp"> {
    let summary = "Apply the dynamic one-shot transform.";

    let dependentDialects = [
        "index::IndexDialect",
        "tensor::TensorDialect",
        "scf::SCFDialect",
        "stablehlo::StablehloDialect"
    ];
}

def CombineGlobalPhasesPass : Pass<"combine-global-phases", "mlir::ModuleOp"> {
    let summary = "Scope-wise merging of global phase operators (safe).";
}


// ----- Quantum circuit transformation passes end ----- //

// ----- DO-QAOA analysis passes ----- //

def LandscapeOverlapAnalysisPass : Pass<"doqaoa-landscape-overlap", "mlir::func::FuncOp"> {
    let summary = "Evaluate energy landscapes, compute cosine similarity, and detect phase-transition regime for DO-QAOA.";
    let description = [{
        For each quantum.freeze_partition op in the function:

        1. LANDSCAPE OVERLAP (Eq. 2.6, arXiv:2602.21689v1)
           Evaluates the QAOA p=1 expectation value E(gamma,beta) on a
           gridSize x gridSize coarse grid for each of the 2^m frozen
           sub-problem replicas. Computes pairwise cosine similarity S_kl
           and annotates the op with the mean overlap q and recommended K.

        2. PHASE TRANSITION DETECTOR
           Estimates the effective s parameter from the graph diameter via
           the heuristic  s_eff = 2 / (1 + diameter).
           Graphs with small diameter (well-connected) have s_eff near 1
           (concentrated regime, DO-QAOA works well).  Graphs with large
           diameter (sparse/path-like) have s_eff near 0 (fragmented
           regime, landscape similarity may be low).
           Emits a warning if s_eff < scThreshold (default 0.6).

        Attributes written onto freeze_partition:
          landscape_overlap_q : F64Attr   -- mean pairwise cosine similarity
          recommended_k       : I32Attr   -- 1 if q >= overlapThreshold else 2^m
          s_eff               : F64Attr   -- estimated phase-transition parameter
    }];
    let options = [
        Option<"gridSize", "grid-size", "unsigned", /*default=*/"16",
               "Number of grid points per axis (gridSize x gridSize total).">,
        Option<"overlapThreshold", "overlap-threshold", "double", /*default=*/"0.9",
               "Cosine similarity threshold above which K=1 is used.">,
        Option<"scThreshold", "sc-threshold", "double", /*default=*/"0.6",
               "Phase-transition threshold sc. Warning emitted when s_eff < sc.">,
    ];
    let dependentDialects = ["mlir::func::FuncDialect"];
}

def BiasShiftAnalysisPass : Pass<"doqaoa-bias-shift", "mlir::func::FuncOp"> {
    let summary = "Compute sub-problem bias metrics and shortcut init values for DO-QAOA.";
    let description = [{
        Operates on freeze_partition ops already annotated by
        doqaoa-landscape-overlap (requires cluster_k, cluster_assignments).

        For each op:
        1. BIAS METRICS (Task 5)
           Computes B_k = (1/N_free) Σ|h_eff[i]| for every sub-problem k,
           where h_eff[i] = h[i] + Σ_{frozen j} J[i,j]·s_j.
           For each cluster, picks the representative as the member with
           minimum B_k (most generic sub-problem).
           Writes ΔB_k = |B_k − B_rep| for every k (0 for representatives).

        2. SHORTCUT INITIALISATION (Task 7)
           Sets physics-informed starting angles for p=1 QAOA:
             init_gamma = −π/6  (≈ −0.5236 rad)
             init_beta  = −π/8  (≈ −0.3927 rad)

        3. BASIN ANALYSIS (Task 8)
           Finds the argmin of the representative sub-problem's landscape
           (cache-hit from doqaoa-landscape-overlap) and records the actual
           basin centre (basin_gamma, basin_beta).
           Emits a warning when the basin deviates > basinTol from the
           shortcut values, indicating the shortcut may be suboptimal.

        Attributes written onto freeze_partition:
          b_values        : tensor<2^m × f64>  -- B_k for every sub-problem
          bias_shifts     : tensor<2^m × f64>  -- ΔB_k for every sub-problem
          representatives : array<i32, K>      -- representative index per cluster
          init_gamma      : f64
          init_beta       : f64
          basin_gamma     : f64
          basin_beta      : f64
    }];
    let options = [
        Option<"biasThreshold", "bias-threshold", "double", /*default=*/"0.3",
               "ΔB threshold: below → direct copy; above → warm-start.">,
        Option<"gridSize", "grid-size", "unsigned", /*default=*/"16",
               "Grid size for basin analysis (must match doqaoa-landscape-overlap).">,
        Option<"basinTol", "basin-tol", "double", /*default=*/"0.2",
               "Tolerance (rad) for basin vs shortcut proximity check.">,
    ];
    let dependentDialects = ["mlir::func::FuncDialect"];
}

def RepresentativeSelectionPass : Pass<"doqaoa-representative-selection", "mlir::func::FuncOp"> {
    let summary = "Mark cluster representatives and annotate parameter transfer modes for DO-QAOA.";
    let description = [{
        Operates on freeze_partition ops already annotated by doqaoa-bias-shift
        (requires cluster_k, cluster_assignments, representatives, bias_shifts, b_values).

        For each sub-problem k in every freeze_partition op:
          - If k is the cluster representative → is_representative[k] = 1, transfer_mode[k] = 0
          - Else if bias_shifts[k] < biasThreshold → transfer_mode[k] = 1  (direct copy)
          - Else → transfer_mode[k] = 2  (warm start)

        Also records aggregate counts: direct_copy_count, warm_start_count.

        Transfer mode encoding:
          0 = representative  (needs full variational optimisation)
          1 = direct copy     (zero training cost, parameters inherited from rep)
          2 = warm start      (short fine-tuning from rep parameters)

        Attributes written onto freeze_partition:
          is_representative  : array<i32, 2^m>  -- 1 for cluster reps, 0 otherwise
          transfer_modes     : array<i32, 2^m>  -- per-sub-problem mode code
          direct_copy_count  : i32              -- total sub-problems eligible for copy
          warm_start_count   : i32              -- total sub-problems needing warm start
    }];
    let options = [
        Option<"biasThreshold", "bias-threshold", "double", /*default=*/"0.3",
               "ΔB threshold: below → direct copy; at/above → warm start.">,
    ];
    let dependentDialects = ["mlir::func::FuncDialect"];
}

def SharedParameterBufferPass : Pass<"doqaoa-shared-buffer", "mlir::func::FuncOp"> {
    let summary = "Compute shared parameter buffer layout for DO-QAOA representative parameters.";
    let description = [{
        Operates on freeze_partition ops already annotated by doqaoa-training-schedule
        (requires training_schedule, schedule_phase_ends, schedule_sources, init_gamma,
        init_beta, basin_gamma, basin_beta).

        Produces a compile-time parameter buffer layout for θ*_rep:

          param_buffer_size  — number of f64 parameters per sub-problem (2 for p=1: [γ, β])
          buffer_slot_map    — array<i32, 2^m>: which slot each sub-problem reads from.
                               Representatives get their own slot (slot = cluster index c).
                               Non-representatives share their cluster rep's slot.
          init_params        — tensor<K × 2 × f64>: initial [γ, β] for each cluster
                               representative.  Uses basin_{gamma,beta} when available,
                               falls back to {init_gamma, init_beta}.
          use_atomic_guards  — i32, always 1: signals the LLVM lowering pass to emit
                               atomic load/store guards on the global buffer.

        The LLVM lowering pass (quantum_to_llvm) reads use_atomic_guards and emits:
          @doqaoa_param_buffer = global [K × 2 × double] zeroinitializer
          @doqaoa_buffer_lock  = global i64 0          ; spin-lock word
        providing thread-safe access when sub-problems are dispatched in parallel.
    }];
    let dependentDialects = ["mlir::func::FuncDialect"];
}

def TrainingScheduleRewriterPass : Pass<"doqaoa-training-schedule", "mlir::func::FuncOp"> {
    let summary = "Lower 2^m sub-problems into a three-phase training schedule for DO-QAOA.";
    let description = [{
        Operates on freeze_partition ops already annotated by
        doqaoa-representative-selection (requires transfer_modes, representatives,
        cluster_assignments, cluster_k).

        Produces a MLIR schedule attribute encoding the execution plan in
        three ordered phases:

          Phase 1 — Full optimisation (mode 0, K sub-problems):
            One full optimisation loop per cluster representative.
            Each representative runs for fullEpochs Adam steps.

          Phase 2 — Warm-start fine-tuning (mode 2):
            For each non-representative sub-problem with bias_shifts >= threshold,
            initialise from the cluster representative's parameters and run
            warmstartEpochs fine-tuning steps.

          Phase 3 — Direct transfer (mode 1):
            For each non-representative sub-problem with bias_shifts < threshold,
            directly copy the cluster representative's optimised parameters.
            Zero training cost.

        The complete schedule is encoded as four parallel arrays (all indexed
        over sub-problems in their original k=0..2^m-1 order):

          training_schedule  : array<i32, 2^m>  -- k indices in execution order
                               [rep_0, ..., rep_{K-1},
                                warm_k0, ...,
                                copy_k0, ...]
          schedule_phase_ends: array<i32: 3>     -- [K, K+warm_count, 2^m]
          schedule_epochs    : array<i32, 2^m>   -- per-k epoch budget
                               (fullEpochs for reps, warmstartEpochs for warm,
                                0 for direct copy)
          schedule_sources   : array<i32, 2^m>   -- per-k param source sub-problem
                               (self for reps, representative index for others)

        Example for 4-cycle m=1, K=1:
          training_schedule   = array<i32: 0, 1>   -- sp0 first (full opt), sp1 second (copy)
          schedule_phase_ends = array<i32: 1, 1, 2> -- phase1=[0], phase2=[], phase3=[1]
          schedule_epochs     = array<i32: 100, 0>  -- sp0 gets 100 epochs, sp1 gets 0
          schedule_sources    = array<i32: 0, 0>    -- sp1 copies from sp0
    }];
    let options = [
        Option<"fullEpochs", "full-epochs", "unsigned", /*default=*/"100",
               "Number of Adam steps for full optimisation of representatives.">,
        Option<"warmstartEpochs", "warmstart-epochs", "unsigned", /*default=*/"20",
               "Number of Adam fine-tuning steps for warm-start sub-problems.">,
    ];
    let dependentDialects = ["mlir::func::FuncDialect"];
}

def WarmStartSchedulerPass : Pass<"doqaoa-warmstart-scheduler", "mlir::func::FuncOp"> {
    let summary = "Run compile-time Adam warm-start for mode-2 DO-QAOA sub-problems.";
    let description = [{
        Operates on freeze_partition ops annotated by doqaoa-shared-buffer
        (requires param_buffer_size, buffer_slot_map, init_params,
        training_schedule, schedule_phase_ends, schedule_sources,
        transfer_modes, cluster_assignments).

        For every mode-2 (warm-start) sub-problem k:
          1. Load init angles [γ, β] from init_params[slot_map[k]]
             (the basin / shortcut parameters of the cluster representative).
          2. Run up to warmstartEpochs steps of Adam (lr=0.01, β1=0.9,
             β2=0.999, ε=1e-8) using EnergyEval finite-difference gradients
             (step h=1e-4 rad).
          3. Declare convergence when ||∇θ||₂ < gradNormTol.
          4. Record final angles, convergence flag, and epoch count.

        For mode-0 (representative) sub-problems the basin init values are
        stored as-is (they will be further refined at runtime in phase 1).
        For mode-1 (direct copy) sub-problems the representative's final
        angles are copied verbatim (zero additional training).

        Attributes written onto freeze_partition:
          warmstart_params       : tensor<2^m × 2 × f64>
              Final [γ, β] for every sub-problem in its original k order.
              mode-0 rows = init (runtime will refine).
              mode-1 rows = representative's init (direct copy).
              mode-2 rows = post-Adam angles.

          warmstart_converged    : array<i32, 2^m>
              1=converged (||∇θ||₂ < tol), 0=hit epoch limit, -1=N/A (mode 0/1).

          warmstart_epochs_used  : array<i32, 2^m>
              Actual epochs run per sub-problem (0 for modes 0 and 1).

          warmstart_final_energy : tensor<2^m × f64>
              ⟨H_k⟩ at the final parameters (NaN for modes 0 and 1).
    }];
    let options = [
        Option<"warmstartEpochs", "warmstart-epochs", "unsigned", /*default=*/"10",
               "Maximum Adam steps per warm-start sub-problem.">,
        Option<"learningRate", "learning-rate", "double", /*default=*/"0.01",
               "Adam learning rate for warm-start fine-tuning.">,
        Option<"gradNormTol", "grad-norm-tol", "double", /*default=*/"1e-4",
               "Convergence threshold on ||∇θ||₂.">,
        Option<"finiteDiffStep", "fd-step", "double", /*default=*/"1e-4",
               "Finite-difference step size h for gradient estimation (rad).">,
    ];
    let dependentDialects = ["mlir::func::FuncDialect"];
}

def DirectTransferPass : Pass<"doqaoa-direct-transfer", "mlir::func::FuncOp"> {
    let summary = "Annotate quantum.bias_transfer ops for direct-copy vs warm-start lowering.";
    let description = [{
        Walks every quantum.bias_transfer op in the function. For each op:

          delta_B = |B_target - B_rep|

          delta_B <= threshold  →  is_direct_copy = 1  (zero training cost)
                                   memcpy param_byte_count bytes src → dst
          delta_B  > threshold  →  is_direct_copy = 0  (warm-start)

        Shape verification: looks up the enclosing function's freeze_partition
        param_buffer_size attribute and asserts param_byte_count = param_buffer_size × 8.
        Fails the pass if param_buffer_size is missing or inconsistent.

        Attributes written onto each bias_transfer op:
          is_direct_copy  : i32  -- 1 for direct copy, 0 for warm-start
          param_byte_count: i32  -- byte count for memcpy (= param_buffer_size × 8)

        Attributes written onto enclosing freeze_partition:
          dt_direct_count : i32  -- number of direct-copy sub-problems
          dt_warmstart_count : i32  -- number of warm-start sub-problems
    }];
    let dependentDialects = ["mlir::func::FuncDialect"];
}

def AggregateMinLoweringPass : Pass<"doqaoa-aggregate-min", "mlir::func::FuncOp"> {
    let summary = "Compute compile-time argmin over DO-QAOA sub-problem energies.";
    let description = [{
        Walks every quantum.aggregate_min op in the function and resolves the
        reduction using compile-time energy data produced by prior passes:

          - warmstart_final_energy (from doqaoa-warmstart-scheduler) for mode-1 and mode-2 sub-problems.
          - Mode-0 sub-problems have NaN energy (runtime optimised); they are given
            the conservative value of 0.0 for ranking only when all others are also NaN.

        Picks best_k = argmin(finite energy values) and encodes it as a
        binary bitstring: bit i of best_k gives the spin of hotspot qubit i
        (0 = +1 spin, 1 = -1 spin).

        Attributes written onto freeze_partition:
          agg_best_k         : i32
          agg_min_energy     : f64            (NaN if all energies are NaN)
          agg_best_bitstring : array<i32, m>  -- spin assignment for hotspot qubits
          agg_candidates_evaluated : i32      -- how many sub-problems had finite energy
    }];
    let dependentDialects = ["mlir::func::FuncDialect"];
}

def NoiseModelPreservationPass : Pass<"doqaoa-noise-preserve", "mlir::func::FuncOp"> {
    let summary = "Validate IBM FakeBrisbane noise parameters are preserved across sub-circuit splits.";
    let description = [{
        Records gate infidelity, T1/T2 decoherence times, and expected CNOT counts as
        MLIR attributes on freeze_partition ops. Emits a warning if any sub-circuit's
        expected CNOT count is high enough that decoherence error > infidelity budget.

        CNOT count per sub-circuit: 2 × |E_free(k)| for p=1 QAOA (each ZZ term
        requires a pair of CNOT gates when compiled to native gate set).

        FakeBrisbane defaults (Table IV of arXiv:2602.21689v1):
          T1  = 127 000 ns
          T2  = 218 000 ns
          CX gate fidelity = 0.9918
          CX gate time     =   533 ns

        Attributes written onto freeze_partition:
          noise_t1_ns         : f64
          noise_t2_ns         : f64
          noise_cx_fidelity   : f64
          noise_cx_time_ns    : f64
          noise_cnot_counts   : array<i32, 2^m>  -- per-sub-problem CNOT budget
          noise_max_cnots     : i32               -- maximum across sub-problems
          noise_depth_ok      : i32               -- 1 if T1-budget not exceeded
    }];
    let options = [
        Option<"noiseT1Ns",       "noise-t1-ns",       "double", /*default=*/"127000.0",
               "T1 relaxation time in nanoseconds.">,
        Option<"noiseT2Ns",       "noise-t2-ns",       "double", /*default=*/"218000.0",
               "T2 dephasing time in nanoseconds.">,
        Option<"noiseCxFidelity", "noise-cx-fidelity", "double", /*default=*/"0.9918",
               "Per-CNOT gate fidelity (FakeBrisbane default).">,
        Option<"noiseCxTimeNs",   "noise-cx-time-ns",  "double", /*default=*/"533.0",
               "CX gate duration in nanoseconds.">,
        Option<"expectedMaxCnots","expected-max-cnots", "unsigned", /*default=*/"0",
               "If > 0, fail pass when any sub-circuit exceeds this CNOT count.">,
    ];
    let dependentDialects = ["mlir::func::FuncDialect"];
}

def DepthCheckPass : Pass<"doqaoa-depth-check", "mlir::func::FuncOp"> {
    let summary = "Assert DO-QAOA CNOT counts match Table IV of arXiv:2602.21689v1.";
    let description = [{
        Computes per-sub-problem CNOT counts from the graph topology and asserts
        the maximum does not exceed expectedMaxCnots.  This serves as a regression
        test ensuring circuit depth is correct after sub-circuit partitioning.

        CNOT count per sub-problem k:
            2 × |{(u,v) : u,v free, |J[u,v]| > 0}|    (p=1 ZZ → 2 CNOTs each edge)

        Table IV reference values (arXiv:2602.21689v1):
          Power-Law m=3, representative sub-circuit: 3 CNOTs
          10-node MaxCut m=2: depends on graph realisation

        If the regression fails, the pass signals failure (like a compile error).

        Attributes written onto freeze_partition:
          depth_cnot_counts   : array<i32, 2^m>   -- per-sub-problem CNOT count
          depth_max_cnots     : i32
          depth_regression_ok : i32               -- 1 if max <= expectedMaxCnots
    }];
    let options = [
        Option<"expectedMaxCnots", "expected-max-cnots", "unsigned", /*default=*/"0",
               "If > 0, assert max CNOT count <= this value (regression mode).">,
    ];
    let dependentDialects = ["mlir::func::FuncDialect"];
}

#endif // QUANTUM_PASSES
