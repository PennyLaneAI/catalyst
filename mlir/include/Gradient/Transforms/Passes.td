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

#ifndef GRADIENT_PASSES
#define GRADIENT_PASSES

include "mlir/Pass/PassBase.td"

def GradientBufferizationPass : Pass<"gradient-bufferize"> {
    let summary = "Bufferize tensors in quantum operations.";

    let dependentDialects = [
        "bufferization::BufferizationDialect",
        "memref::MemRefDialect",
        "index::IndexDialect"
    ];

    let constructor = "catalyst::createGradientBufferizationPass()";
}

def GradientPreprocessingPass : Pass<"gradient-preprocess"> {
    let summary = "Insert Func.CallOp for ForwardOp and ReverseOp.";

    let dependentDialects = [
        "index::IndexDialect"
    ];

    let constructor = "catalyst::createGradientPreprocessingPass()";
}

def GradientPostprocessingPass : Pass<"gradient-postprocess"> {
    let summary = "Make the signatures of ForwardOp and ReverseOp compatible"
        "to Enzyme.";

    let dependentDialects = [
        "memref::MemRefDialect",
        "index::IndexDialect"
    ];

    let constructor = "catalyst::createGradientPostprocessingPass()";
}

def GradientLoweringPass : Pass<"lower-gradients"> {
    let summary = "Lower operations in the gradient dialect to core MLIR dialects.";

    let dependentDialects = [
        "arith::ArithDialect",
        "linalg::LinalgDialect",
        "index::IndexDialect",
        "tensor::TensorDialect",
        "memref::MemRefDialect",
        "bufferization::BufferizationDialect",
        "catalyst::CatalystDialect"
    ];

    let constructor = "catalyst::createGradientLoweringPass()";
}

def GradientConversionPass : Pass<"convert-gradient-to-llvm"> {
    let summary = "Perform a dialect conversion from Gradient to LLVM";

    let dependentDialects = [
        "index::IndexDialect",
        "linalg::LinalgDialect",
        "LLVM::LLVMDialect",
        "func::FuncDialect",
        "catalyst::quantum::QuantumDialect"
    ];

    let constructor = "catalyst::createGradientConversionPass()";

    let options = [
        Option<
            /*C++ var name=*/"useGenericFunctions",
            /*CLI arg name=*/"use-generic-functions",
            /*type=*/"bool",
            /*default=*/"true",
            /*description=*/
            "Use generic allocation and deallocation functions instead of the "
            "classic 'malloc', 'aligned_alloc' and 'free' functions"
        >
    ];
}

def GradientPreprocessingPass : Pass<"gradient-preprocess"> {
    let summary = "Bufferize tensors in quantum operations.";

    let dependentDialects = [
        "bufferization::BufferizationDialect",
        "memref::MemRefDialect",
        "index::IndexDialect"
    ];

    let constructor = "catalyst::createGradientPreprocessingPass()";
}

#endif // GRADIENT_PASSES
