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

        * `executor.open`        -> `__catalyst__executor__open(addr)`
        * `executor.send_binary` -> `__catalyst__executor__send_binary(addr, path, format)`
        * `executor.launch`      -> `__catalyst__executor__launch(addr, "_catalyst_pyface_<callee>", ...)`
        * `executor.call`        -> `__catalyst__executor__call_wrapper(addr, sym, ...)` + `__catalyst__executor__free_result`
        * `executor.close`       -> `__catalyst__executor__close(addr)`
    }];

    let dependentDialects = [
        "mlir::LLVM::LLVMDialect",
        "catalyst::executor::ExecutorDialect",
    ];
}

def DispatchExecutorTargetsPass : Pass<"dispatch-executor-targets", "mlir::ModuleOp"> {
    let summary = "Ship cross-compiled catalyst.target modules to a executor executor.";
    let description = [{
        For every `builtin.module` carrying a `catalyst.dispatch` attribute and the
        `catalyst.object_file` path, this pass will perform the following:

        1. Injects `executor.open` into `setup()` (once per unique address) and `executor.send_binary`
           into `setup()` (once per module).
        2. Replaces every host-side `func.call` to the module's entry functions with a
           `executor.launch` op carrying the executor address and the entry callee.
        3. Erases the bodyless external declarations and the nested module from the host.

        Session teardown is handled by the runtime, which closes every open session when the
        process exits, so no explicit close op is emitted.

        The pass is a no-op when no `catalyst.dispatch` modules are present.
    }];

    let dependentDialects = [
        "catalyst::executor::ExecutorDialect",
        "catalyst::CatalystDialect",
        "mlir::func::FuncDialect"
    ];
}

#endif // EXECUTOR_PASSES
