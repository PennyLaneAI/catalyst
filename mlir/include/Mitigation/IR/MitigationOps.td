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

include "mlir/IR/BuiltinAttributes.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/CallInterfaces.td"

include "Mitigation/IR/MitigationDialect.td"

def Folding : I32EnumAttr<"Folding",
    "Folding types",
    [
        I32EnumAttrCase<"GLOBAL", 1>,
        I32EnumAttrCase<"ALL",    2>,
        I32EnumAttrCase<"RANDOM", 3>,
    ]> {
    let cppNamespace = "catalyst::mitigation";
    let genSpecializedAttr = 0;
}

def FoldingAttr : EnumAttr<Mitigation_Dialect, Folding, "folding">;


def ZneOp : Mitigation_Op<"zne", [DeclareOpInterfaceMethods<CallOpInterface>,
    DeclareOpInterfaceMethods<SymbolUserOpInterface>]> {
    let summary = "Compute a quantum function with ZNE (Zero Noise Extrapolation) error mitigation.";
    let description = [{
        The `mitigation.zne` operation computes the results of the quantum function with ZNE mitigation.
    }];

    let arguments = (ins
        FlatSymbolRefAttr:$callee,
        Variadic<AnyType>:$args,
        FoldingAttr:$folding,
        RankedTensorOf<[AnySignlessIntegerOrIndex]>:$scaleFactors
    );
    let results = (outs Variadic<AnyTypeOf<[AnyFloat, RankedTensorOf<[AnyFloat]>]>>);

    let assemblyFormat = [{
        $callee `(` $args `)` `folding` `(` $folding `)` `scaleFactors` `(` $scaleFactors `:` type($scaleFactors) `)` attr-dict `:` functional-type($args, results)
    }];
}

#endif // MITIGATION_OPS
