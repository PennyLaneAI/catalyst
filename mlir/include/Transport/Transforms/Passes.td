// Copyright 2026 Xanadu Quantum Technologies Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef TRANSPORT_PASSES
#define TRANSPORT_PASSES

include "mlir/Pass/PassBase.td"

def ConvertTransportToLLVMPass : Pass<"convert-transport-to-llvm", "mlir::ModuleOp"> {
    let summary = "Lower the transport dialect to LLVM dialect with runtime calls.";
    let description = [{
        Lowers each `transport` op to an `llvm.call` on the matching
        `__catalyst__transport__*` symbol.
    }];

    let dependentDialects = [
        "mlir::LLVM::LLVMDialect"
    ];
}

#endif // TRANSPORT_PASSES
