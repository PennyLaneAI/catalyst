// Copyright 2023 Xanadu Quantum Technologies Inc.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef ION_INTERFACES
#define ION_INTERFACES

include "mlir/IR/OpBase.td"

def IonRegion : OpInterface<"IonRegion"> {
    let description = [{
        This interface provides a generic way to interact with instructions that are
        considered quantum regions. These are characterized by operating on a single
        qubit register, and returning a new register value.
    }];

    let cppNamespace = "::catalyst::ion";

    let methods = [
        InterfaceMethod<
            "Return the quantum register operand.",
            "mlir::Value", "getRegisterOperand", (ins), /*methodBody=*/[{
                return $_op->getOperand(0);
            }]
        >,
        InterfaceMethod<
            "Return the quantum register result.",
            "mlir::Value", "getRegisterResult", (ins), /*methodBody=*/[{
                return $_op->getResult(0);
            }]
        >,
    ];
}

#endif // ION_INTERFACES
