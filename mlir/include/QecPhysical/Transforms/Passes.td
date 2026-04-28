// Copyright 2026 Xanadu Quantum Technologies Inc.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef QECPHYSICAL_PASSES
#define QECPHYSICAL_PASSES

include "mlir/Pass/PassBase.td"

def QecPhyiscalLLVMConversionPass : Pass<"convert-qecp-to-llvm"> {
    let summary = "Perform a dialect conversion from QECPhyiscal to LLVM.";

    let options = [
        Option<
            "useArrayBackedRegisters",
            "use-array-backed-registers",
            "bool",
            default="false",
            desc="Use the array-backed-registers conversion pattern for quantum.insert ops."
        >
    ];

    let dependentDialects = ["LLVM::LLVMDialect"];
}


def DecoderPass : Pass<"decoder-lowering"> {
    let summary = "Lower qecp.decode_esm_css opertions to llvm func calls.";

    let dependentDialects = [
        "memref::MemRefDialect",
    ];
}