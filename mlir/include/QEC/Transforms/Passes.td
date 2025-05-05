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

def MaxPauliSizeOption : Option<
    /*C++ var name=*/"max_pauli_size",
    /*CLI name=*/"max-pauli-size",
    /*Type=*/"uint",
    /*Default=*/"0", // 0 means no limit
    /*Description=*/"The maximum size of the Pauli product. 0 means no limit.">;

def CliffordTToPPRPass : Pass<"to_ppr"> {
    let summary = "Convert quantum dialects to the QEC dialect.";

    let dependentDialects = [
        "catalyst::qec::QECDialect",
    ];

    let constructor = "catalyst::createLowerToQECPass()";
}

def CommuteCliffordTPPRPass : Pass<"commute_ppr"> {
    let summary = "Commutate CliffordT operations to PPRotation operations.";
    
    let dependentDialects = [
        "catalyst::qec::QECDialect",
    ];
    let options = [MaxPauliSizeOption];
    let constructor = "catalyst::createCommuteCliffordTPPRPass()";
}

def CommuteCliffordPastPPMPass : Pass<"merge_ppr_ppm"> {
    let summary = "Absorb Clifford operations into Pauli Product Measurement operations.";
    
    let dependentDialects = [
        "catalyst::qec::QECDialect",
    ];
    let options = [MaxPauliSizeOption];
    let constructor = "catalyst::createCommuteCliffordPastPPMPass()";
}

#endif // QEC_PASSES
