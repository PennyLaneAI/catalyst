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
include "mlir/IR/EnumAttr.td"

def GridsynthMethod : I32EnumAttr<"GridsynthMethod", "Synthesis method of the gridsynth pass",
    [
        I32EnumAttrCase<"Deterministic", 0, "deterministic">,
        I32EnumAttrCase<"Mixed", 1, "mixed">,
        I32EnumAttrCase<"MixedFallback", 2, "mixed-fallback">,
    ]> {
    let cppNamespace = "catalyst::quantum";
    let genSpecializedAttr = 0;
}

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

def ReferenceSemanticsConversionPass : Pass<"convert-to-reference-semantics"> {
    let summary = "Convert to reference semantics quantum dialect";

    let dependentDialects = [
       "catalyst::qref::QRefDialect"
    ];
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

def ResolveGateLevelAdjoint : Pass<"resolve-gate-level-adjoint"> {
    let summary = "Applies canonicalization patterns for adjoint quantum operations.";
}

def CtrlLoweringPass : Pass<"ctrl-lowering"> {
    let summary = "Lower ctrl regions by distributing the controls over the enclosed operations.";

    let dependentDialects = ["scf::SCFDialect"];
}

def ModifiersLoweringPass : Pass<"modifiers-lowering"> {
    let summary = "Lower quantum.ctrl and quantum.adjoint regions to op-level modifiers.";

    let description = [{
        Runs the ctrl-lowering and adjoint-lowering rewrite patterns together under a single greedy
        driver until a fixed point. This allows arbitrarily nested control and adjoint regions to be
        resolved. Each pattern defers (returns a match failure) while its region still contains the
        other modifier, so the greedy worklist naturally interleaves them: an inner adjoint region is
        reduced, then the enclosing control lowers on a re-try, and vice versa. This reaches the same
        fixed point as manually alternating the standalone ctrl-lowering and adjoint-lowering passes,
        in a single pass and regardless of nesting depth or order.
    }];

    let dependentDialects = [
        "catalyst::CatalystDialect",
        "index::IndexDialect",
        "tensor::TensorDialect",
        "complex::ComplexDialect",
        "scf::SCFDialect"
    ];
}

def RemoveGlobalPhasesPass : Pass<"remove-global-phases"> {
    let summary = "Remove global phases by deleting phase operations without control wires.";
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
    let description = [{
        Discretizes single-qubit RZ and PhaseShift gates, and PPR-Z rotations with an arbitrary
        angle, at execution time. The pass fails on controlled RZ and PhaseShift gates, which need
        to be decomposed into single-qubit rotations first.

        With `method=mixed-fallback`, each rotation allocates an ancilla qubit, measures it and
        releases it, and applies the fallback sequence inside an `scf.if` on the measurement
        result.
    }];

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
        /*Description=*/desc="Whether to decompose into the PPR basis set instead of the standard Clifford+T basis set.">,
        Option<
        /*C++ var name=*/"method",
        /*CLI name=*/"method",
        /*Type=*/"::catalyst::quantum::GridsynthMethod",
        /*Default=*/"::catalyst::quantum::GridsynthMethod::Deterministic",
        /*Description=*/desc="Synthesis method. 'mixed' and 'mixed-fallback' are cheaper but match the accuracy of 'deterministic' only on average over samples.",
        /*Additional flags=*/additionalFlags=[{::llvm::cl::values(
            clEnumValN(::catalyst::quantum::GridsynthMethod::Deterministic, "deterministic",
                       "A single Ross-Selinger sequence per rotation."),
            clEnumValN(::catalyst::quantum::GridsynthMethod::Mixed, "mixed",
                       "A probabilistic mixture of Ross-Selinger sequences for under- and over-rotations."),
            clEnumValN(::catalyst::quantum::GridsynthMethod::MixedFallback, "mixed-fallback",
                       "A probabilistic mixture of projective rotations on an ancilla, with a mixed fallback on failure.")
        )}]>,];

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
            "Maps an operator to a list of alternative decomposition rules that will be considered"
            " alongside any built-in rules for the operator.">,
        Option<
            /*C++ name*/"bytecodeRulesFile",
            /*CLI name*/"bytecode-rules",
            /*Type*/"std::string",
            /*Default*/[{ "" }],
            /*Description*/"A path to a bytecode file of compiled decomposition rules.">,
        Option<
            /*C++ name*/"libQPDPath",
            /*CLI name*/"libQPD-path",
            /*Type*/"std::string",
            /*Default*/[{ "" }],
            /*Description*/"A path to the QuantumPythonDecompositions dynamic library.">,
        Option<
            /*C++ name*/"libpythonPath",
            /*CLI name*/"libpython-path",
            /*Type*/"std::string",
            /*Default*/[{ "" }],
            /*Description*/"A path to the python shared library.">,
        Option<
            /*C++ name*/"verboseOption",
            /*CLI name*/"verbose",
            /*Type*/"bool",
            /*Default*/[{ false }],
            /*Description*/"Print the decomposition rule chosen for each operator to stderr.">,
        Option<
            /*C++ name*/"inlineRuleBody",
            /*CLI name*/"inline-rule-body",
            /*Type*/"bool",
            /*Default*/[{ false }],
            /*Description*/"Whether to inline the rule body. Defaults to false.">
    ];

    let dependentDialects = [
        "memref::MemRefDialect",
        "func::FuncDialect",
        "scf::SCFDialect",
        "LLVM::LLVMDialect",
        "stablehlo::StablehloDialect",
        "tensor::TensorDialect",
        "catalyst::CatalystDialect",
        "catalyst::qref::QRefDialect",
        "catalyst::pbc::PBCDialect",
        "catalyst::mbqc::MBQCDialect",
    ];
}

def ResolveStatePrepOperatorPass : Pass<"resolve-state-prep-operator"> {
    let summary = "Replace all quantum.operator ops of StatePrep/BasisState to equivalent quantum.set_state/set_basis_state operations ";
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

def InstrumentClassicalControlPass : Pass<"instrument-classical-control", "mlir::ModuleOp"> {
    let summary = "Inform the runtime which operations are classically controlled by measurements.";
    let description = [{
        Instruments the program with runtime calls that tell the device which operations are
        classically controlled by mid-circuit measurement results. This is used by resource
        tracking on `null.qubit` to place classically controlled operations after the
        measurements they depend on when computing the circuit depth.

        A region is classically controlled if the value deciding whether it executes depends on
        a measurement result: the branches of `scf.if` and `scf.index_switch`, the body of
        `scf.while`, and the body of `scf.for` when its bounds depend on a measurement. A
        `pbc.select.ppm` whose basis selection depends on a measurement is also classically
        controlled. Each such region is wrapped in
        `__catalyst__rt__begin_classical_control(token)` and
        `__catalyst__rt__end_classical_control()`, where `token` is the maximum of the
        tokens returned by `__catalyst__rt__measurement_token()` right after each measurement
        the region depends on.

        Dependencies are traced through the classical operations that compute the controlling
        value, and through the values carried by `scf` control flow. A measurement whose
        token does not dominate the controlled region, e.g. one carried by a loop from a
        previous iteration, is represented by the token of the most recent measurement at the
        start of the region. Dependencies through function arguments, the results of function
        calls (other than through the call operands), and memory are not traced.

        This pass must run before bufferization, while the controlling values are still SSA
        values, and before control flow is lowered to the `cf` dialect.
    }];

    let dependentDialects = [
        "func::FuncDialect",
        "arith::ArithDialect"
    ];
}

// ----- Quantum circuit transformation passes end ----- //

#endif // QUANTUM_PASSES
