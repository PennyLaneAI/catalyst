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

#ifndef REMOTE_PASSES
#define REMOTE_PASSES

include "mlir/Pass/PassBase.td"

def CrossCompileRemoteKernelsPass : Pass<"cross-compile-remote-kernels", "mlir::ModuleOp"> {
    let summary = "Cross-compile each `qnode` to a `.o` and emit `remote.*` ops at the host call sites.";
    let description = [{
        For every `func.func` in the host module carrying the `qnode` attribute, this pass:
        1. Clones the func into a standalone module.
        2. Sanitizes it for lowering to LLVM IR and emits the `.o` to `<workspace>/<name>.o`.
        3. Marks every `func.call` targeting the qnode with `catalyst.remote_kernel_path`
           holding the absolute path of the produced `.o`.
    }];

    let dependentDialects = [
        "mlir::LLVM::LLVMDialect",
        "mlir::func::FuncDialect",
        "catalyst::CatalystDialect",
        "catalyst::remote::RemoteDialect",
    ];

    let options = [
        Option<
            /*C++ var name=*/"workspace",
            /*CLI arg name=*/"workspace",
            /*type=*/"std::string",
            /*default=*/"\"\"",
            /*description=*/
            "Filesystem directory to write cross-compiled `.o` files into."
        >,
        Option<
            /*C++ var name=*/"target",
            /*CLI arg name=*/"target",
            /*type=*/"std::string",
            /*default=*/"\"x86_64\"",
            /*description=*/
            "LLVM target triple used for object emission."
        >,
        Option<
            /*C++ var name=*/"cpu",
            /*CLI arg name=*/"cpu",
            /*type=*/"std::string",
            /*default=*/"\"generic\"",
            /*description=*/
            "LLVM CPU model fed to `createTargetMachine`. Defaults to "
            "`generic`, which emits baseline code for the triple."
        >,
        Option<
            /*C++ var name=*/"features",
            /*CLI arg name=*/"features",
            /*type=*/"std::string",
            /*default=*/"\"\"",
            /*description=*/
            "Comma-separated subtarget features fed to `createTargetMachine` "
            "(each token must be `+feat` or `-feat`, e.g. `+crc,+aes,+sha2`). "
            "Empty (default) lets the CPU choose its own feature set."
        >,
        Option<
            /*C++ var name=*/"address",
            /*CLI arg name=*/"address",
            /*type=*/"std::string",
            /*default=*/"\"\"",
            /*description=*/
            "Executor's TCP `host:port` address for remote dispatch."
        >
    ];
}

def ConvertRemoteToLLVMPass : Pass<"convert-remote-to-llvm", "mlir::ModuleOp"> {
    let summary = "Lower the `remote` dialect to direct calls into the Catalyst remote runtime.";
    let description = [{
        Replaces each `remote.*` op with the corresponding `__catalyst__remote__*`
        runtime entry point:

        * `remote.open`        -> `__catalyst__remote__open(addr)`
        * `remote.send_binary` -> `__catalyst__remote__send_binary(addr, path, format)`
        * `remote.launch`      -> `__catalyst__remote__launch(addr, "_catalyst_pyface_<callee>", ...)`
        * `remote.call`        -> `__catalyst__remote__call_wrapper(addr, sym, ...)` + `__catalyst__remote__free_result`
    }];

    let dependentDialects = [
        "mlir::LLVM::LLVMDialect",
        "catalyst::remote::RemoteDialect",
    ];
}

#endif // REMOTE_PASSES
