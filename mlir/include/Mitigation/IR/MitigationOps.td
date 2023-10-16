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

#ifndef MITIGATION_OPS
#define MITIGATION_OPS

include "mlir/Interfaces/CallInterfaces.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/IR/BuiltinAttributes.td"
include "mlir/IR/OpBase.td"

include "Mitigation/IR/MitigationDialect.td"

def ZneOp : Mitigation_Op<"zne", [DeclareOpInterfaceMethods<CallOpInterface>,
        DeclareOpInterfaceMethods<SymbolUserOpInterface>]> {
    let summary = "Compute a quantum function with ZNE error mitigation.";
    let description = [{
        The `mitigation.zne` operation computes the quantum function with ZNE mitigation.
    }];

    let arguments = (ins
        StrAttr:$method,
        FlatSymbolRefAttr:$callee,
        Variadic<AnyType>:$params,
        RankedTensorOf<[AnyFloat]>:$scalarFactors
    );
    let results = (outs Variadic<AnyTypeOf<[AnyFloat, RankedTensorOf<[AnyFloat]>]>>);

    let hasVerifier = 1;

    let assemblyFormat = [{
        $method $callee `(` $params `)` `(` $scalarFactors `,` type($scalarFactors) `)` attr-dict `:` functional-type($params, results)
    }];
}

#endif // MITIGATION_OPS
