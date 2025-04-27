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
include "mlir/IR/EnumAttr.td"

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
    
    let constructor = "catalyst::createCommuteCliffordTPPRPass()";
}

def DecompositionMethod : I32EnumAttr<"DecompositionMethod", "Decomposition method", 
    [
    I32EnumAttrCase<"AutoCorrected", 0, "auto-corrected">,
    I32EnumAttrCase<"InjectMagicState", 1, "inject-magic-state">
    ]> { 
    let cppNamespace = "catalyst::qec";
    let genSpecializedAttr = 0; 
}

def DecomposeMethodOption : Option<
    /*C++ var name=*/"decomposeMethod",
    /*CLI arg name=*/"decompose-method",
    /*type=*/"::catalyst::qec::DecompositionMethod",
    /*default=*/"::catalyst::qec::DecompositionMethod::AutoCorrected",
    "Decomposition method to use",
    [{::llvm::cl::values(
        clEnumValN(::catalyst::qec::DecompositionMethod::AutoCorrected,
                    "auto-corrected", "Decompose using auto-corrected method."),
        clEnumValN(::catalyst::qec::DecompositionMethod::InjectMagicState,
                    "inject-magic-state",
                    "Decompose by injecting magic states method.")
    )}]>;

def CommuteCliffordPastPPMPass : Pass<"merge_ppr_ppm"> {
    let summary = "Absorb Clifford operations into Pauli Product Measurement operations.";
    
    let dependentDialects = [
        "catalyst::qec::QECDialect",
    ];
    
    let constructor = "catalyst::createCommuteCliffordPastPPMPass()";
}

def DecomposeNonCliffordPPRToPPMPass : Pass<"decompose_non_clifford_ppr_to_ppm"> {
    let summary = "Decompose non-Clifford PPRotation operations to set of Pauli Product Measurement operations.";
    
    let dependentDialects = [
        "catalyst::qec::QECDialect",
    ];

    let constructor = "catalyst::createDecomposeNonCliffordPPRToPPMPass()";

    let options = [DecomposeMethodOption];
}

#endif // QEC_PASSES
