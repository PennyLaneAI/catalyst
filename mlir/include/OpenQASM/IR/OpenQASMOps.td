// Copyright 2024 Xanadu Quantum Technologies Inc.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef OPENQASM_OPS
#define OPENQASM_OPS

include "mlir/Interfaces/CallInterfaces.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/IR/BuiltinAttributes.td"
include "mlir/IR/OpBase.td"

include "OpenQASM/IR/OpenQASMDialect.td"


def OpenQASM_AllocateOp : OpenQASM_Op<"allocate"> {
    let summary = "Allocate qubit register";
    let description = [{
        Allocates a global qubit register of the specified size.
    }];

    let arguments = (ins I32Attr:$size);

    let assemblyFormat = [{
        attr-dict `(` $size `)`
    }];
}

def OpenQASM_RYOp : OpenQASM_Op<"ry"> {
    let summary = "RY gate";
    let description = [{
        Applies an RY gate with the given angle to the specified qubit register index.
    }];

    let arguments = (ins F64Attr:$angle, I32Attr:$qubit);

    let assemblyFormat = [{
        attr-dict `(` $angle `)` $qubit 
    }];
}

def OpenQASM_CNOTOp : OpenQASM_Op<"cnot"> {
    let summary = "CNOT gate";
    let description = [{
        Applies a CNOT gate to the specified control and target qubit register indices.
    }];

    let arguments = (ins I32Attr:$control, I32Attr:$target);

    let assemblyFormat = [{
        attr-dict $control `,` $target
    }];
}

#endif // OPENQASM_OPS
