// Copyright 2022-2023 Xanadu Quantum Technologies Inc.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef GRADIENT_OPS
#define GRADIENT_OPS

include "mlir/Interfaces/CallInterfaces.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/IR/BuiltinAttributes.td"
include "mlir/IR/OpBase.td"

include "Gradient/IR/GradientDialect.td"

def GradOp : Gradient_Op<"grad", [DeclareOpInterfaceMethods<CallOpInterface>,
        DeclareOpInterfaceMethods<SymbolUserOpInterface>]> {
    let summary = "Compute the gradient of a function.";
    let description = [{
        The `gradient.grad` operation computes the gradient of a function
        using the finite difference method.

        This operation acts much like the `func.call` operation, taking a
        symbol reference and arguments to the original functionan as input.
        However, instead of the function result, the gradient of the function
        is returned.

        Example:

        ```mlir
        func.func @foo(%arg0: f64) -> f64 {
            %res = arith.mulf %arg0, %arg0 : f64
            func.return %res : f64
        }

        %0 = arith.constant 2.0 : f64
        %1 = gradient.grad @foo(%0) : (f64) -> f64
        ```
    }];

    let arguments = (ins
        StrAttr:$method,
        FlatSymbolRefAttr:$callee,
        Variadic<AnyType>:$operands,
        OptionalAttr<AnyIntElementsAttr>:$diffArgIndices,
        OptionalAttr<Builtin_FloatAttr>:$finiteDiffParam
    );
    let results = (outs Variadic<AnyTypeOf<[AnyFloat, RankedTensorOf<[AnyFloat]>]>>);

    let hasVerifier = 1;

    let assemblyFormat = [{
        $method $callee `(` $operands `)` attr-dict `:` functional-type($operands, results)
    }];
}

def ValueAndGradOp : Gradient_Op<"value_and_grad", [
        SameVariadicResultSize,
        DeclareOpInterfaceMethods<CallOpInterface>,
        DeclareOpInterfaceMethods<SymbolUserOpInterface>
        ]> {
    let summary = "Compute the value and gradient of a function.";

    let arguments = (ins
        StrAttr:$method,
        FlatSymbolRefAttr:$callee,
        Variadic<AnyType>:$operands,
        OptionalAttr<AnyIntElementsAttr>:$diffArgIndices,
        OptionalAttr<Builtin_FloatAttr>:$finiteDiffParam
    );

    let results = (outs
        Variadic<AnyTypeOf<[AnyFloat, RankedTensorOf<[AnyFloat]>]>>:$vals,
        Variadic<AnyTypeOf<[AnyFloat, RankedTensorOf<[AnyFloat]>]>>:$gradients
    );

    let assemblyFormat = [{
        $method $callee `(` $operands `)`
        attr-dict `:` functional-type(operands, results)
    }];


    let hasVerifier = 1;
}

def AdjointOp : Gradient_Op<"adjoint", [AttrSizedOperandSegments]> {
    let summary = "Perform quantum AD using the adjoint method on a device.";

    let arguments = (ins
        FlatSymbolRefAttr:$callee,
        Index:$gradSize,
        Variadic<AnyType>:$args,
        Variadic<MemRefOf<[AnyFloat]>>:$data_in
    );

    let results = (outs
        Variadic<AnyTypeOf<[
            AnyFloat,
            RankedTensorOf<[AnyFloat]>,
        ]>>
    );

    let assemblyFormat = [{
        $callee `(` $args `)`
        `size` `(` $gradSize `)`
        ( `in` `(` $data_in^ `:` type($data_in) `)` )?
        attr-dict `:` functional-type($args, results)
    }];
}

def BackpropOp : Gradient_Op<"backprop", [AttrSizedOperandSegments,
        DeclareOpInterfaceMethods<SymbolUserOpInterface>]> {
    let summary = "Perform classic automatic differentiation using Enzyme AD.";

    let arguments = (ins
        FlatSymbolRefAttr:$callee,
        Variadic<AnyType>:$args,
        Variadic<MemRefOf<[AnyFloat]>>:$diffArgShadows,
        Variadic<MemRefOf<[AnyFloat]>>:$calleeResults,
        Variadic<AnyTypeOf<[
            RankedTensorOf<[AnyFloat]>,
            MemRefOf<[AnyFloat]>
        ]>>:$cotangents,
        OptionalAttr<AnyIntElementsAttr>:$diffArgIndices
    );

    let results = (outs
        Variadic<AnyTypeOf<[
            AnyFloat,
            RankedTensorOf<[AnyFloat]>
        ]>>:$gradients
    );

    let hasVerifier = 1;

    let assemblyFormat = [{
        $callee `(` $args `)`
        ( `grad_out` `(` $diffArgShadows^ `:` type($diffArgShadows) `)` )?
        ( `callee_out` `(` $calleeResults^ `:` type($calleeResults) `)` )?
        `cotangents` `(` $cotangents `:` type($cotangents) `)`
        attr-dict `:` functional-type($args, results)
    }];
}

def BackpropWithValueOp : Gradient_Op<"backprop_with_value", [
        AttrSizedOperandSegments,
        SameVariadicResultSize,
        DeclareOpInterfaceMethods<SymbolUserOpInterface>
        ]> {
    let summary = "Perform classic automatic differentiation using Enzyme AD.";

    let arguments = (ins
        FlatSymbolRefAttr:$callee,
        Variadic<AnyType>:$args,
        Variadic<MemRefOf<[AnyFloat]>>:$diffArgShadows,
        Variadic<MemRefOf<[AnyFloat]>>:$calleeResults,
        Variadic<AnyTypeOf<[
            RankedTensorOf<[AnyFloat]>,
            MemRefOf<[AnyFloat]>
        ]>>:$cotangents,
        OptionalAttr<AnyIntElementsAttr>:$diffArgIndices
    );

    let results = (outs
        Variadic<AnyTypeOf<[
            AnyFloat,
            RankedTensorOf<[AnyFloat]>
        ]>>:$vals,
        Variadic<AnyTypeOf<[
            AnyFloat,
            RankedTensorOf<[AnyFloat]>
        ]>>:$gradients
    );

    let hasVerifier = 1;

    let assemblyFormat = [{
        $callee `(` $args `)`
        ( `grad_out` `(` $diffArgShadows^ `:` type($diffArgShadows) `)` )?
        ( `callee_out` `(` $calleeResults^ `:` type($calleeResults) `)` )?
        `cotangents` `(` $cotangents `:` type($cotangents) `)`
        attr-dict `:` functional-type($args, results)
    }];
}


def JVPOp : Gradient_Op<"jvp", [
        AttrSizedOperandSegments,
        SameVariadicResultSize,
        DeclareOpInterfaceMethods<CallOpInterface>,
        DeclareOpInterfaceMethods<SymbolUserOpInterface>
        ]> {
    let summary = "Compute the jvp of a function.";

    let arguments = (ins
        StrAttr:$method,
        FlatSymbolRefAttr:$callee,
        Variadic<AnyType>:$params,
        Variadic<AnyType>:$tangents,
        OptionalAttr<AnyIntElementsAttr>:$diffArgIndices,
        OptionalAttr<Builtin_FloatAttr>:$finiteDiffParam
    );

    let results = (outs
        Variadic<AnyTypeOf<[AnyFloat, RankedTensorOf<[AnyFloat]>]>>:$calleeResults,
        Variadic<AnyTypeOf<[AnyFloat, RankedTensorOf<[AnyFloat]>]>>:$jvps
    );

    let assemblyFormat = [{
        $method $callee `(` $params `)` `tangents` `(` $tangents `)`
        attr-dict `:` functional-type(operands, results)
    }];


    let hasVerifier = 1;
}


def VJPOp : Gradient_Op<"vjp", [
        AttrSizedOperandSegments,
        AttrSizedResultSegments,
        DeclareOpInterfaceMethods<CallOpInterface>,
        DeclareOpInterfaceMethods<SymbolUserOpInterface>]> {
    let summary = "Compute the vjp of a function.";

    let arguments = (ins
        StrAttr:$method,
        FlatSymbolRefAttr:$callee,
        Variadic<AnyType>:$params,
        Variadic<AnyType>:$cotangents,
        OptionalAttr<AnyIntElementsAttr>:$diffArgIndices,
        OptionalAttr<Builtin_FloatAttr>:$finiteDiffParam
    );

    let results = (outs
        Variadic<AnyTypeOf<[AnyFloat, RankedTensorOf<[AnyFloat]>]>>:$calleeResults,
        Variadic<AnyTypeOf<[AnyFloat, RankedTensorOf<[AnyFloat]>]>>:$vjps
    );

    let assemblyFormat = [{
        $method $callee `(` $params `)` `cotangents` `(` $cotangents `)`
            attr-dict `:` functional-type(operands, results)
    }];

    let hasVerifier = 1;
}

#endif // GRADIENT_OPS
