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
    let summary = "Ship cross-compiled `catalyst.target` modules to an executor.";
    let description = [{
        For every nested `builtin.module` carrying a `catalyst.dispatch` attribute and a
        `catalyst.object_file` path, the pass:

        1. Injects `executor.open` into `setup()` (once per unique address) and `executor.send_binary`
           into `setup()` (once per module).
        2. Rewrites every host-side `catalyst.launch_kernel` targeting that module into an
           `executor.launch` carrying the executor address, the entry callee, and the object-file
           path.
        3. Erases the nested module from the host after its `catalyst.launch_kernel`s are rewritten.

        A `catalyst.custom_call` carrying a `dispatch` entry in its `backend_config` is rewritten into
        an `executor.call` on the resolved executor address.

        Session teardown is handled by the runtime, which closes every open session at process exit,
        so no explicit close op is emitted.

        The pass is a no-op when no `catalyst.dispatch` modules and no dispatch `catalyst.custom_call`
        ops are present.
    }];

    let dependentDialects = [
        "catalyst::executor::ExecutorDialect",
        "catalyst::CatalystDialect",
        "mlir::func::FuncDialect"
    ];
}

#endif // EXECUTOR_PASSES
