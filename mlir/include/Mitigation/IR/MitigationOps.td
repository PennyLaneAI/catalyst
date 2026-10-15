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
        I32EnumAttrCase<"global", 1>,
        I32EnumAttrCase<"all",    2>,
        I32EnumAttrCase<"random", 3>,
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

        The `numFolds` operand holds one floating-point fold count
        `(scale_factor - 1) / 2` per scale factor. The `global` and `all` folding
        methods scale the circuit by an exact odd-integer factor and thus require
        integral values. `random` folding also accepts fractional values, realized
        as base folds plus one extra fold sampled with probability equal to the
        fractional remainder.
    }];

    let arguments = (ins
        SymbolRefAttr:$callee,
        Variadic<AnyType>:$args,
        FoldingAttr:$folding,
        RankedTensorOf<[AnyFloat]>:$numFolds,
        OptionalAttr<DictArrayAttr>:$arg_attrs,
        OptionalAttr<DictArrayAttr>:$res_attrs
    );
    let results = (outs Variadic<AnyTypeOf<[AnyFloat, RankedTensorOf<[AnyFloat]>]>>);

    let assemblyFormat = [{
        $callee `(` $args `)` `folding` `(` $folding `)` `numFolds` `(` $numFolds `:` type($numFolds) `)` attr-dict `:` functional-type($args, results)
    }];
}

def RemOp : Mitigation_Op<"rem", [DeclareOpInterfaceMethods<CallOpInterface>,
    DeclareOpInterfaceMethods<SymbolUserOpInterface>,
    AttrSizedResultSegments]> {
    let summary = "Compute a quantum function with Readout Error Mitigation (REM).";
    let description = [{
        The `mitigation.rem` operation wraps a quantum function call so the
        mitigation-lowering pass can insert calibration circuits next to it.
        When `runCalibration` is true the pass emits an all-zeros and an
        all-ones calibration circuit alongside the original callee; their
        sample/probs results are returned as the `zeros` and `ones` SSA values
        and consumed by the classical REM post-processing on the JAX side.
        When false the calibration is ommited, which lets the caller supply
        cached confusion matrices and skip the calibration circuits without
        changing the op's result signature.
    }];

    let arguments = (ins
        SymbolRefAttr:$callee,
        Variadic<AnyType>:$args,
        BoolAttr:$runCalibration,
        OptionalAttr<DictArrayAttr>:$arg_attrs,
        OptionalAttr<DictArrayAttr>:$res_attrs
    );
    let results = (outs
        Variadic<AnyTypeOf<[AnyFloat, RankedTensorOf<[AnyFloat, AnyInteger]>]>>:$callee_result,
        Variadic<RankedTensorOf<[AnyFloat, AnyInteger]>>:$zeros,
        Variadic<RankedTensorOf<[AnyFloat, AnyInteger]>>:$ones
    );
    let assemblyFormat = [{
        $callee `(` $args `)` `runCalibration` `(` $runCalibration `)` attr-dict `:` functional-type($args, results)
    }];
}

#endif // MITIGATION_OPS
