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

#ifndef REMOTE_OPS
#define REMOTE_OPS

include "mlir/IR/OpBase.td"
include "mlir/IR/BuiltinAttributes.td"
include "Remote/IR/RemoteDialect.td"

//===----------------------------------------------------------------------===//
// remote.open
//===----------------------------------------------------------------------===//

def Remote_OpenOp : Remote_Op<"open"> {
    let summary = "Open a session with a remote executor.";
    let description = [{
        Establish a connection to the remote executor reachable at `address`
        Subsequent remote operations targeting the same `address` reuse this session.
    }];

    let arguments = (ins StrAttr:$address);

    let assemblyFormat = [{
        `(` $address `)` attr-dict
    }];
}

//===----------------------------------------------------------------------===//
// remote.send_binary
//===----------------------------------------------------------------------===//

def Remote_SendBinaryOp : Remote_Op<"send_binary"> {
    let summary = "Ship a compiled kernel object file to the remote executor.";
    let description = [{
        Transfer the compiled binary at `binary_path` to the remote executor at `address`.
        The remote loads the file and exposes all of its symbols.
        `format` selects the wire encoding which is handled by the runtime.
    }];

    let arguments = (ins
        StrAttr:$address,
        StrAttr:$binary_path,
        DefaultValuedAttr<I32Attr, "0">:$format
    );

    let assemblyFormat = [{
        `(` $address `,` $binary_path `)` attr-dict
    }];
}

//===----------------------------------------------------------------------===//
// remote.launch
//===----------------------------------------------------------------------===//

def Remote_LaunchOp : Remote_Op<"launch"> {
    let summary = "Dispatch a remote-compiled kernel and receive its results.";
    let description = [{
        Invoke the kernel previously registered via `remote.send_binary`.
        The `inputs` operand list mirrors the original host-side `func.call`
        to the kernel; `results` mirrors the kernel's return values.
    }];

    let arguments = (ins
        Variadic<AnyMemRef>:$inputs,
        StrAttr:$address,
        StrAttr:$kernel_callee
    );

    let results = (outs Variadic<AnyMemRef>:$results);

    let assemblyFormat = [{
        `(` $kernel_callee `,` $address `)` `(` $inputs `)` attr-dict
            `:` functional-type($inputs, $results)
    }];
}

//===----------------------------------------------------------------------===//
// remote.call
//===----------------------------------------------------------------------===//

def Remote_CallOp : Remote_Op<"call"> {
    let summary = "Invoke a symbol in a previously-loaded shared library.";
    let description = [{
        Invoke `symbol` in a shared library already loaded on the remote executor.

        The operand list packs input buffers followed by output buffers.
        `num_input_args`, when present, marks the boundary: operands `[0, num_input_args)`
        are inputs and the remainder are output buffers to be filled by the callee.
        Without `num_input_args` all operands are treated as inputs.
    }];

    let arguments = (ins
        Variadic<AnyMemRef>:$inputs,
        StrAttr:$address,
        StrAttr:$symbol,
        OptionalAttr<I32Attr>:$num_input_args
    );

    let results = (outs Variadic<AnyMemRef>:$results);

    let assemblyFormat = [{
        `(` $symbol `,` $address `)` `(` $inputs `)` attr-dict
            `:` functional-type($inputs, $results)
    }];
}

#endif // REMOTE_OPS
