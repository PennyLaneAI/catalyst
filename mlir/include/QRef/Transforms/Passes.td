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

#ifndef REF_QUANTUM_PASSES
#define REF_QUANTUM_PASSES

include "mlir/Pass/PassBase.td"

def RQHelloWorldPass : Pass<"rq-hw"> {
    let summary = "QRef Hello world!";

    // let dependentDialects = [
    //    "scf::SCFDialect",
    //    "catalyst::quantum::QuantumDialect"
    // ];
}

#endif // REF_QUANTUM_PASSES
