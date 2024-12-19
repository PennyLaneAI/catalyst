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

def QuantumBufferizationPass : Pass<"quantum-bufferize"> {
    let summary = "Bufferize tensors in quantum operations.";

    let dependentDialects = [
        "bufferization::BufferizationDialect",
        "memref::MemRefDialect"
    ];

    let constructor = "catalyst::createQuantumBufferizationPass()";
}

def QuantumConversionPass : Pass<"convert-quantum-to-llvm"> {
    let summary = "Perform a dialect conversion from Quantum to LLVM (QIR).";

    let dependentDialects = ["LLVM::LLVMDialect"];

    let constructor = "catalyst::createQuantumConversionPass()";
}

def EmitCatalystPyInterfacePass : Pass<"emit-catalyst-py-interface"> {
    let summary = "Emit catalyst python's default interface.";

    let dependentDialects = ["LLVM::LLVMDialect"];

    let constructor = "catalyst::createEmitCatalystPyInterfacePass()";
}

def CopyGlobalMemRefPass : Pass<"cp-global-memref"> {
    let summary = "Copy global memrefs before returning from C interface.";

    let dependentDialects = [
        "memref::MemRefDialect",
        "func::FuncDialect",
        "scf::SCFDialect",
        "LLVM::LLVMDialect"
    ];

    let constructor = "catalyst::createCopyGlobalMemRefPass()";
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

    let constructor = "catalyst::createAdjointLoweringPass()";
}

def AnnotateFunctionPass : Pass<"annotate-function"> {
    let summary = "Annotate functions that contain a measurement operation.";

    let dependentDialects = [
        "catalyst::CatalystDialect",
        "func::FuncDialect"
    ];

    let constructor = "catalyst::createAnnotateFunctionPass()";
}

def SplitMultipleTapesPass : Pass<"split-multiple-tapes"> {
    let summary = "Given a qnode containing multiple tapes, split each tape into its own function.";

    let constructor = "catalyst::createSplitMultipleTapesPass()";
}

def StaticCustomLoweringPass : Pass<"static-custom-lowering"> {
    let summary = "Lower static custom ops to regular custom op with dynamic parameters.";

    let constructor = "catalyst::createStaticCustomLoweringPass()";
}


// ----- Quantum circuit transformation passes begin ----- //
// For example, automatic compiler peephole opts, etc.

def IonsDecompositionPass : Pass<"ions-decomposition"> {
    let summary = "Decompose the gates to the set {RX, RY, MS}";

    let constructor = "catalyst::createIonsDecompositionPass()";
}

def RemoveChainedSelfInversePass : Pass<"remove-chained-self-inverse"> {
    let summary = "Perform removal of chained operations that are self inverses.";

    let constructor = "catalyst::createRemoveChainedSelfInversePass()";
}

def MergeRotationsPass : Pass<"merge-rotations"> {
    let summary = "Perform merging of chained rotation gates about the same axis.";

    let constructor = "catalyst::createMergeRotationsPass()";
}

def DisentangleCNOTPass : Pass<"disentangle-CNOT"> {
    let summary = "Replace a CNOT gate with two single qubit gates whenever possible.";

    let constructor = "catalyst::createDisentangleCNOTPass()";
    let options = [
    Option<"EmitFSMStateRemark", "emit-FSM-state-remark",
           "bool", /*default=*/"false",
           "Whether to emit the state analysis result from the simple states propagation FSM onto the gate operations.">,
    ];
}

// ----- Quantum circuit transformation passes end ----- //

#endif // QUANTUM_PASSES
