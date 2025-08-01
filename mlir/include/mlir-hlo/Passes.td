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

#ifndef CATALYST_MLIRHLO_PASSES
#define CATALYST_MLIRHLO_PASSES

include "mlir/Pass/PassBase.td"

// mhlo legalize sort pass.
def MhloLegalizeSortPass : Pass<"mhlo-legalize-sort", "func::FuncOp"> {
  let summary = "Legalize from Mhlo sort to SCF control flow.";
  let constructor = "createMhloLegalizeSortPass()";
  let dependentDialects = ["arith::ArithDialect",
                           "bufferization::BufferizationDialect",
                           "scf::SCFDialect", "tensor::TensorDialect"];
}

// mhlo legalize to std pass.
def MhloLegalizeToStandardPass : Pass<"mhlo-legalize-to-std", "func::FuncOp"> {
  let summary = "Legalize from MHLO dialect to standard dialect.";
  let constructor = "createMhloLegalizeToStdPass()";
}

// mhlo legalize to control flow pass.
def MhloLegalizeControlFlowPass : Pass<"mhlo-legalize-control-flow", "func::FuncOp"> {
  let summary = "Legalize from MHLO control flow to SCF control flow.";
  let constructor = "createMhloLegalizeControlFlowPass()";
  let dependentDialects = ["scf::SCFDialect", "tensor::TensorDialect"];
}

#endif // CATALYST_MLIRHLO_PASSES
