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

// ----- Quantum circuit transformation passes begin ----- //
// For example, automatic compiler peephole opts, etc.

class QuantumCircuitTransformationPassBase {
    list<Option> options = [];
}

def QuantumCircuitTransformationPass: QuantumCircuitTransformationPassBase {
    let options = [
    Option<"FuncNameOpt", "func-name",
           "std::string", /*default=*/"\"\"",
           "The name of the function (or qnode) to run the quantum circuit transformation pass on.">,
    ];
}

def RemoveChainedSelfInversePass : Pass<"remove-chained-self-inverse"> {
    let summary = "Perform removal of chained operations that are self inverses.";

    let constructor = "catalyst::createRemoveChainedSelfInversePass()";
    let options = QuantumCircuitTransformationPass.options;
}

def MergeRotationPass : Pass<"merge-rotation"> {
    let summary = "merge rotation boilerplate words";

    let constructor = "catalyst::createMergeRotationPass()";
    let options = !listconcat(
    QuantumCircuitTransformationPass.options,
    [
    Option<"MyOption", "my-option",
            "std::string", /*default=*/"\"\"",
           "Boilerplate option. Delete if unnecessary.">,
    ]
    );
}

def PropagateSimpleStatesTesterPass : Pass<"propagate-simple-states-tester"> {
    let summary = "Propagate simple states through their corresponding gates down the circuit. The simple states are the six Pauli basis states.";

    let constructor = "catalyst::createPropagateSimpleStatesTesterPass()";
    let options = QuantumCircuitTransformationPass.options;
}

def DisentangleCNOTPass : Pass<"disentangle-CNOT"> {
    let summary = "Replace a CNOT gate with two single qubit gates whenever possible.";

    let constructor = "catalyst::createDisentangleCNOTPass()";
    let options = QuantumCircuitTransformationPass.options;
}

// ----- Quantum circuit transformation passes end ----- //

#endif // QUANTUM_PASSES
