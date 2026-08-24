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

def InjectTransportSessionPass : Pass<"inject-transport-session", "mlir::ModuleOp"> {
    let summary = "Emit transport session bring-up/teardown from a catalyst.backline attribute.";
    let description = [{
        Reads the `catalyst.backline` module attribute and injects the session
        lifecycle into the host entry function.
    }];

    let dependentDialects = [
        "::catalyst::transport::TransportDialect",
        "mlir::func::FuncDialect"
    ];
}

def LowerDecodeToTransportPass : Pass<"lower-decode-to-transport", "mlir::ModuleOp"> {
    let summary = "Lower qecp decode ops to a transport post/collect round over their buffers.";
    let description = [{
        Replaces each bufferized `qecp.decode_esm_css` op with transport.get_session +
        transport.post + transport.collect, offloading the decode to a backline coprocessor.
        The pass is a no-op unless the module carries a `catalyst.backline` attribute declaring at
        least one coprocessor
    }];

    let dependentDialects = [
        "::catalyst::transport::TransportDialect"
    ];
}

#endif // TRANSPORT_PASSES
