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
    let summary = "Lower the `executor` dialect to direct calls into the Catalyst executor runtime.";
    let description = [{
        Replaces each `executor.*` op with the corresponding `__catalyst__executor__*`
        runtime entry point:

        * `executor.open`        -> `session = __catalyst__executor__open(addr)`
        * `executor.send_binary` -> `__catalyst__executor__send_binary(session, path, format)`
        * `executor.launch`      -> `__catalyst__executor__launch(session, "_catalyst_pyface_<callee>", ...)`
        * `executor.call`        -> `__catalyst__executor__call_wrapper(session, sym, ...)` + `__catalyst__executor__free_result`
        * `executor.close`       -> `__catalyst__executor__close(session)`
    }];

    let dependentDialects = [
        "mlir::LLVM::LLVMDialect",
        "catalyst::executor::ExecutorDialect",
    ];
}

#endif // EXECUTOR_PASSES
