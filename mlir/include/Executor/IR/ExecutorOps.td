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

#ifndef EXECUTOR_OPS
#define EXECUTOR_OPS

include "mlir/IR/OpBase.td"
include "mlir/IR/BuiltinAttributes.td"
include "Executor/IR/ExecutorDialect.td"
include "Executor/IR/ExecutorTypes.td"

//===----------------------------------------------------------------------===//
// executor.open
//===----------------------------------------------------------------------===//

def Executor_OpenOp : Executor_Op<"open"> {
    let summary = "Open a session with an executor.";
    let description = [{
        Establish a connection to the executor reachable at `address` and return an
        `!executor.session` handle identifying it. Subsequent executor operations consume the handle
        to target this session.
    }];

    let arguments = (ins StrAttr:$address);

    let results = (outs Executor_SessionType:$session);

    let assemblyFormat = [{
        `(` $address `)` attr-dict `:` type($session)
    }];
}

//===----------------------------------------------------------------------===//
// executor.send_binary
//===----------------------------------------------------------------------===//

def Executor_SendBinaryOp : Executor_Op<"send_binary"> {
    let summary = "Ship a compiled kernel object file to the executor.";
    let description = [{
        Transfer the compiled binary at `binary_path` over the `session` to the executor.
        The executor loads the file and exposes all of its symbols.
        `format` selects the wire encoding which is handled by the runtime.
    }];

    let arguments = (ins
        Executor_SessionType:$session,
        StrAttr:$binary_path,
        DefaultValuedAttr<I32Attr, "0">:$format
    );

    let assemblyFormat = [{
        $session `(` $binary_path `)` attr-dict `:` type($session)
    }];
}

//===----------------------------------------------------------------------===//
// executor.launch
//===----------------------------------------------------------------------===//

def Executor_LaunchOp : Executor_Op<"launch"> {
    let summary = "Dispatch a cross-compiled kernel on the executor and receive its results.";
    let description = [{
        Invoke, on the given `session`, the kernel previously registered via
        `executor.send_binary`. The `inputs` operand list mirrors the original host-side
        `func.call` to the kernel; `results` mirrors the kernel's return values.
    }];

    let arguments = (ins
        Executor_SessionType:$session,
        Variadic<AnyMemRef>:$inputs,
        StrAttr:$kernel_callee
    );

    let results = (outs Variadic<AnyMemRef>:$results);

    let assemblyFormat = [{
        $session `(` $kernel_callee `)` `(` $inputs `)` attr-dict
            `:` type($session) `,` functional-type($inputs, $results)
    }];
}

//===----------------------------------------------------------------------===//
// executor.call
//===----------------------------------------------------------------------===//

def Executor_CallOp : Executor_Op<"call"> {
    let summary = "Invoke a symbol in a previously-loaded shared library.";
    let description = [{
        Invoke `symbol` in a shared library already loaded on the executor `session`.

        The operand list packs input buffers followed by output buffers.
        `num_input_args`, when present, marks the boundary: operands `[0, num_input_args)`
        are inputs and the remainder are output buffers to be filled by the callee.
        Without `num_input_args` all operands are treated as inputs.
    }];

    let arguments = (ins
        Executor_SessionType:$session,
        Variadic<AnyMemRef>:$inputs,
        StrAttr:$symbol,
        OptionalAttr<I32Attr>:$num_input_args
    );

    let results = (outs Variadic<AnyMemRef>:$results);

    let assemblyFormat = [{
        $session `(` $symbol `)` `(` $inputs `)` attr-dict
            `:` type($session) `,` functional-type($inputs, $results)
    }];
}

//===----------------------------------------------------------------------===//
// executor.close
//===----------------------------------------------------------------------===//

def Executor_CloseOp : Executor_Op<"close"> {
    let summary = "Release a session handle with an executor.";
    let description = [{
        Release the `session` previously established via `executor.open`, marking the end of its use
        in the IR.
    }];

    let arguments = (ins Executor_SessionType:$session);

    let assemblyFormat = [{
        $session attr-dict `:` type($session)
    }];
}

#endif // EXECUTOR_OPS
