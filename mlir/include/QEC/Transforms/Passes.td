// Copyright 2025-2026 Xanadu Quantum Technologies Inc.

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

//===----------------------------------------------------------------------===//
//                               Enums
//===----------------------------------------------------------------------===//

def DecomposeMethod : I32EnumAttr<"DecomposeMethod", "Decomposition method", 
    [
        I32EnumAttrCase<"AutoCorrected", 0, "auto-corrected">,
        I32EnumAttrCase<"CliffordCorrected", 1, "clifford-corrected">,
        I32EnumAttrCase<"PauliCorrected", 2, "pauli-corrected">,
    ]> { 
    let cppNamespace = "catalyst::qec";
    let genSpecializedAttr = 0; 
}

//===----------------------------------------------------------------------===//
//                               Options
//===----------------------------------------------------------------------===//

def MaxPauliSizeOption : Option<
    /*C++ var name=*/"maxPauliSize",
    /*CLI name=*/"max-pauli-size",
    /*Type=*/"size_t",
    /*Default=*/"0", // 0 means no limit
    /*Description=*/"The maximum size of the Pauli product. 0 means no limit.">;

def DecomposeMethodOption : Option<
    /*C++ var name=*/"decomposeMethod",
    /*CLI arg name=*/"decompose-method",
    /*type=*/"::catalyst::qec::DecomposeMethod",
    /*default=*/"::catalyst::qec::DecomposeMethod::PauliCorrected",
    "Decomposition method to use",
    [{::llvm::cl::values(
        clEnumValN(::catalyst::qec::DecomposeMethod::AutoCorrected,
                    "auto-corrected", "Decompose using auto-corrected method."),
        clEnumValN(::catalyst::qec::DecomposeMethod::CliffordCorrected,
                    "clifford-corrected",
                    "Decompose by injecting magic states method."),
        clEnumValN(::catalyst::qec::DecomposeMethod::PauliCorrected,
                    "pauli-corrected",
                    "Decompose by using the pauli-corrected method.")
    )}]>;

def AvoidYMeasureOption : Option<
    /*C++ var name=*/"avoidYMeasure",
    /*CLI arg name=*/"avoid-y-measure",
    /*type=*/"bool",
    /*default=*/"false",
    "Avoid Pauli-Y measurements for Clifford rotations. Rather than performing a Pauli-Y measurement for Clifford rotations (sometimes more costly), a Y state is used instead (requires Y state preparation).">;

//===----------------------------------------------------------------------===//
//                               Passes
//===----------------------------------------------------------------------===//

def ToPPRPass : Pass<"to-ppr"> {
    let summary = "Convert quantum dialects to the QEC dialect.";

    let dependentDialects = [ "catalyst::qec::QECDialect" ];
}

def CommutePPRPass : Pass<"commute-ppr"> {
    let summary = "Commutate CliffordT operations to PPRotation operations.";
    
    let dependentDialects = [ "catalyst::qec::QECDialect" ];

    let options = [MaxPauliSizeOption];
}

def MergePPRIntoPPMPass : Pass<"merge-ppr-ppm"> {
    let summary = "Absorb Clifford operations into Pauli Product Measurement operations.";
    
    let dependentDialects = [ "catalyst::qec::QECDialect" ];

    let options = [MaxPauliSizeOption];
}

def DecomposeNonCliffordPPRPass : Pass<"decompose-non-clifford-ppr"> {
    let summary = "Decompose non-Clifford PPRotation operations to set of Pauli Product Measurement operations.";
    
    let dependentDialects = [ "catalyst::qec::QECDialect", "scf::SCFDialect" ];

    let options = [DecomposeMethodOption, AvoidYMeasureOption];
}

def DecomposeCliffordPPRPass : Pass<"decompose-clifford-ppr"> {
    let summary = "Decompose Clifford PPRotation operations to set of Pauli Product Measurement operations.";
    
    let dependentDialects = [ "catalyst::qec::QECDialect" ];

    let options = [AvoidYMeasureOption];
}

def PPRToPPMPass : Pass<"ppr-to-ppm"> {
    let summary = "Convert PPRotation operations to Pauli Product Measurement operations.";

    let dependentDialects = [ "catalyst::qec::QECDialect" ];

    let options = [DecomposeMethodOption, AvoidYMeasureOption];
}

def PPMCompilationPass : Pass<"ppm-compilation"> {
    let summary = "Convert CliffordT operations to Pauli Product Measurement operations.";
    
    let dependentDialects = [ "catalyst::qec::QECDialect" ];

    let options = [MaxPauliSizeOption, DecomposeMethodOption, AvoidYMeasureOption];
}

def CountPPMSpecsPass : Pass<"ppm-specs"> {
    let summary = "Count specs in Pauli Product Measurement operations.";
}

def PartitionLayersPass : Pass<"partition-layers"> {
    let summary = "Partition layers of Pauli Product Measurement operations.";
}

def TLayerReductionPass : Pass<"reduce-t-depth"> {
    let summary = "Reduce T layers to PPR operations.";
}

def PPRToMBQCPass : Pass<"ppr-to-mbqc"> {
    let summary = "Convert PPR/PPM operations to MBQC operations.";
}

def DecomposeArbitraryPPRPass : Pass<"decompose-arbitrary-ppr"> {
    let summary = "Decompose an arbitrary angle PPR into a collection of PPRs, PPMs and a single-qubit arbitrary PPR in the Z basis.";
}

def UnrollConditionalPPRPPMPass : Pass<"unroll-conditional-ppr-ppm"> {
    let summary = "Lower conditional PPR/PPM operations into normal PPR/PPM using scf conditional dialect.";
    
    let dependentDialects = [ "catalyst::qec::QECDialect", "scf::SCFDialect" ];
}

#endif // QEC_PASSES
