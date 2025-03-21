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

#ifndef QEC_PASSES
#define QEC_PASSES

include "mlir/Pass/PassBase.td"

def CliffordTToPPRPass : Pass<"convert-clifford-t-to-ppr"> {
    let summary = "Convert quantum dialects to the QEC dialect.";

    let dependentDialects = [
        "catalyst::qec::QECDialect",
    ];

    let constructor = "catalyst::createLowerToQECPass()";
}

def CommuteCliffordTPPRPass : Pass<"commute-clifford-t-ppr"> {
    let summary = "Commutate CliffordT operations to PPRotation operations.";
    
    let dependentDialects = [
        "catalyst::qec::QECDialect",
    ];
    
    let constructor = "catalyst::createCommuteCliffordTPPRPass()";
}

def AbsorbCliffordToPPMPass : Pass<"ppr_to_ppm"> {
    let summary = "Absorb Clifford operations into Pauli Product Measurement operations.";
    
    let dependentDialects = [
        "catalyst::qec::QECDialect",
    ];
    
    let constructor = "catalyst::createAbsorbCliffordToPPMPass()";
}

#endif // QEC_PASSES
