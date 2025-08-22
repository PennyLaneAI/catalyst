// Copyright 2025 Xanadu Quantum Technologies Inc.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// This file registers a couple mlir-hlo passes that have been removed upstream.
// The passes have been modified from its original form in the
//   tensorflow/mlir-hlo
// repository at
//   https://github.com/tensorflow/mlir-hlo/blob/a5529d99fc4d1132b0c282a053d26c11e6636b3a/mhlo/transforms/mhlo_passes.td
// released under the Apache License, Version 2.0, with the following copyright notice:
//
//    /* Copyright 2019 The OpenXLA Authors.
//    Licensed under the Apache License, Version 2.0 (the "License");
//    you may not use this file except in compliance with the License.
//    You may obtain a copy of the License at
//        http://www.apache.org/licenses/LICENSE-2.0
//    Unless required by applicable law or agreed to in writing, software
//    distributed under the License is distributed on an "AS IS" BASIS,
//    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//    See the License for the specific language governing permissions and
//    limitations under the License.
//    ==============================================================================*/

#ifndef CATALYST_STABLEHLO_PASSES
#define CATALYST_STABLEHLO_PASSES

include "mlir/Pass/PassBase.td"

// -------------------- Catalyst's own hlo-related passes ------------------------ //

def ScatterLoweringPass : Pass<"scatter-lowering"> {
    let summary = "Lower scatter op from Stable HLO to loops.";

    let dependentDialects = [
        "mlir::func::FuncDialect",
        "index::IndexDialect",
        "stablehlo::StablehloDialect",
        "tensor::TensorDialect",
        "scf::SCFDialect"
    ];
}

def HloCustomCallLoweringPass : Pass<"hlo-custom-call-lowering"> {
    let summary = "Lower custom calls op from Stable HLO to CallOp.";

    let dependentDialects = [
        "index::IndexDialect",
        "mlir::func::FuncDialect",
        "catalyst::CatalystDialect",
    ];
}

// -------------------- upstream mhlo passes removed in stablehlo ------------------------ //
// stablehlo legalize sort pass.
def StablehloLegalizeSortPass : Pass<"stablehlo-legalize-sort", "func::FuncOp"> {
  let summary = "Legalize from Stablehlo sort to SCF control flow.";

  let dependentDialects = ["arith::ArithDialect",
                           "bufferization::BufferizationDialect",
                           "scf::SCFDialect", "tensor::TensorDialect"];
}

// stablehlo legalize to std pass.
def StablehloLegalizeToStandardPass : Pass<"stablehlo-legalize-to-std", "func::FuncOp"> {
  let summary = "Legalize from Stablehlo dialect to standard dialect.";
}

// stablehlo legalize to control flow pass.
def StablehloLegalizeControlFlowPass : Pass<"stablehlo-legalize-control-flow", "func::FuncOp"> {
  let summary = "Legalize from Stablehlo control flow to SCF control flow.";
  let dependentDialects = ["scf::SCFDialect", "tensor::TensorDialect"];
}

#endif // CATALYST_STABLEHLO_PASSES
