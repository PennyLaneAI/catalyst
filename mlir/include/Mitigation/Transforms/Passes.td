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

#ifndef MITIGATION_PASSES
#define MITIGATION_PASSES

include "mlir/Pass/PassBase.td"

def MitigationLoweringPass : Pass<"lower-mitigation"> {
    let summary = "Lower the ZNE operation in the mitigation dialect to core MLIR dialects.";

    let dependentDialects = [
        "arith::ArithDialect",
        "index::IndexDialect",
        "scf::SCFDialect",
        "catalyst::quantum::QuantumDialect"
    ];

    let constructor = "catalyst::createMitigationLoweringPass()";
}

#endif // MITIGATION_PASSES
