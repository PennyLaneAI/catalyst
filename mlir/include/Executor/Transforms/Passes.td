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

#ifndef EXECUTOR_PASSES
#define EXECUTOR_PASSES

include "mlir/Pass/PassBase.td"

def ConvertExecutorToLLVMPass : Pass<"convert-executor-to-llvm", "mlir::ModuleOp"> {
    let summary = "Lower the `executor` dialect to direct calls into the Catalyst remote runtime.";
    let description = [{
        Replaces each `executor.*` op with the corresponding `__catalyst__remote__*`
        runtime entry point:

        * `executor.open`        -> `__catalyst__remote__open(addr)`
        * `executor.send_binary` -> `__catalyst__remote__send_binary(addr, path, format)`
        * `executor.launch`      -> `__catalyst__remote__launch(addr, "_catalyst_pyface_<callee>", ...)`
        * `executor.call`        -> `__catalyst__remote__call_wrapper(addr, sym, ...)` + `__catalyst__remote__free_result`
    }];

    let dependentDialects = [
        "mlir::LLVM::LLVMDialect",
        "catalyst::executor::ExecutorDialect",
    ];
}

#endif // EXECUTOR_PASSES
