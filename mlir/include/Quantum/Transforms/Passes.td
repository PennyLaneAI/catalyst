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

def InstantiatePythonDecompRulePass : Pass<"instantiate-decomp-rules"> {
    let summary = "Lowers instantiated decomposition rules using static op data.";
}

// ----- Quantum circuit transformation passes end ----- //

#endif // QUANTUM_PASSES
