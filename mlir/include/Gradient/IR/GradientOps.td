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

include "mlir/Interfaces/FunctionInterfaces.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/IR/BuiltinAttributes.td"
include "mlir/IR/OpBase.td"

include "Gradient/IR/GradientDialect.td"
include "Gradient/IR/GradientInterfaces.td"

def GradOp : Gradient_Op<"grad", [
        DeclareOpInterfaceMethods<CallOpInterface>,
        DeclareOpInterfaceMethods<SymbolUserOpInterface>, 
        GradientOpInterface
        ]> {
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
        AttrSizedResultSegments,
        DeclareOpInterfaceMethods<CallOpInterface>,
        DeclareOpInterfaceMethods<SymbolUserOpInterface>,
        GradientOpInterface
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


def BackpropOp : Gradient_Op<"backprop", [
        AttrSizedOperandSegments,
        AttrSizedResultSegments,
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
        OptionalAttr<AnyIntElementsAttr>:$diffArgIndices,
        OptionalAttr<BoolAttr>:$keepValueResults
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
        DeclareOpInterfaceMethods<SymbolUserOpInterface>,
        GradientOpInterface
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
        DeclareOpInterfaceMethods<SymbolUserOpInterface>,
        GradientOpInterface
        ]> {
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

def ForwardOp : Gradient_Op<"forward",
    [FunctionOpInterface, IsolatedFromAbove]> {

  let summary = "Operation denoting the forwrad pass that is registered with Enzyme.";

  let description = [{
    Wrapper around the concrete function. This wrapper ensures calling convention.

    This function matches the expected calling convention from Enzyme.
    Enzyme's calling convention expects a shadow argument for every pointer.
    Since the callbacks all expect tensors, all of them are pointers.
    Also, since the callbacks passes out parameters, then these are also duplicated.

    After lowered to LLVM, this function will have the following parameters:

    @foo(%inp0: !llvm.ptr, %diff0: !llvm.ptr,
         ...
         %inpArgc-1: !llvm.ptr, %diffArgc-1: !llvm.ptr,
         %out0: !llvm.ptr, %cotangent0: !llvm.ptr,
         ...
         %outputResc-1: !llvm.ptr, %cotangentResc-1: !llvm.ptr)

    The return value of enzyme is expected to be the tape.
    Enzyme's documentation has the following to say:

        The return type of the augmented forward pass is a struct type containing first the tape type,
        followed by the original return type, if any.
        If the return type is a duplicated type,
        then there is a third argument which contains the shadow of the return.

    Let's just break this down a bit:

        The return type of the augmented forward pass is a struct type containing first the tape type,

    This means that the return type for function foo will be the following in pseudocode

        %tape0Type = { memref elements }
        ...
        %tapeTapec-1Type = { memref elements }
        %tape = { %tape0Type, ... %tapeTapec-1Type }
        %returnTy = { %tape, ... }

    Then:

        followed by the original return type, if any.
    
    since there is none, then:

        %returnTy = { %tape }

    Then:

        If the return type is a duplicated type,
        then there is a third argument which contains the shadow of the return.

     this one is also nothing to worry for the current implementation because there are no returns.

     One thing that was found experimentally and through tests in Enzyme is that the tape can also be a pointer.
     We use this in the case when there is no tape to return. Instead of returning an empty struct, we return a null
     pointer that is just never dereferenced.
    
  }];

  let arguments = (ins
     SymbolNameAttr: $sym_name,
     TypeAttrOf<FunctionType>: $function_type,
     FlatSymbolRefAttr: $implementation,
     I64Attr: $argc,
     I64Attr: $resc,
     I64Attr: $tape,
     OptionalAttr<DictArrayAttr>: $arg_attrs,
     OptionalAttr<DictArrayAttr>: $res_attrs
  );

  let regions = (region AnyRegion: $body);

  let builders = [OpBuilder<(ins
    "mlir::StringRef":$name, "mlir::FunctionType":$type,
    CArg<"mlir::ArrayRef<mlir::NamedAttribute>", "{}">:$attrs)
  >];

  let extraClassDeclaration = [{
    //===------------------------------------------------------------------===//
    // FunctionOpInterface Methods
    //===------------------------------------------------------------------===//

    /// Returns the argument types of this function.
    mlir::ArrayRef<mlir::Type> getArgumentTypes() { return getFunctionType().getInputs(); }

    /// Returns the result types of this function.
    mlir::ArrayRef<mlir::Type> getResultTypes() { return getFunctionType().getResults(); }

    mlir::Region *getCallableRegion() { return &getBody(); }
  }];

  let hasCustomAssemblyFormat = 1;
  let skipDefaultBuilders = 1;
}

def ReverseOp : Gradient_Op<"reverse",
    [FunctionOpInterface, IsolatedFromAbove]> {

  let summary = "Operation denoting the reverse pass that is registered with Enzyme.";

  let description = [{
    Wrapper around the concrete function. This wrapper ensures calling convention.

    This matches Enzyme's calling convention. From the documentation:

        The final argument is a custom “tape” type that can be used to pass information from the forward to the reverse pass.

    Experimentally, it looks like whenever there are no return values, the type passed to this function is the following type
    which matches the return type of the forward op, but it is somewhat ambiguous with what it says in the documentation.

        %returnTy = { %tape }

    
  }];

  let arguments = (ins
     SymbolNameAttr: $sym_name,
     TypeAttrOf<FunctionType>: $function_type,
     FlatSymbolRefAttr: $implementation,
     I64Attr: $argc,
     I64Attr: $resc,
     I64Attr: $tape,
     OptionalAttr<DictArrayAttr>: $arg_attrs,
     OptionalAttr<DictArrayAttr>: $res_attrs
  );

  let regions = (region AnyRegion: $body);

  let builders = [OpBuilder<(ins
    "mlir::StringRef":$name, "mlir::FunctionType":$type,
    CArg<"mlir::ArrayRef<mlir::NamedAttribute>", "{}">:$attrs)
  >];

  let extraClassDeclaration = [{
    //===------------------------------------------------------------------===//
    // FunctionOpInterface Methods
    //===------------------------------------------------------------------===//

    /// Returns the argument types of this function.
    mlir::ArrayRef<mlir::Type> getArgumentTypes() { return getFunctionType().getInputs(); }

    /// Returns the result types of this function.
    mlir::ArrayRef<mlir::Type> getResultTypes() { return getFunctionType().getResults(); }

    mlir::Region *getCallableRegion() { return &getBody(); }
  }];

  let hasCustomAssemblyFormat = 1;
  let skipDefaultBuilders = 1;
}

def ReturnOp : Gradient_Op<"return",
    [Terminator, ParentOneOf<["ForwardOp", "ReverseOp"]>]> {

  let summary = "Return tapes or nothing";

  let arguments = (ins
    Variadic<AnyTypeOf<[MemRefOf<[AnyType]>]>>: $tape,
    I1Attr: $empty
  );

  let assemblyFormat = [{ attr-dict ($tape^ `:` type($tape))? }];
}

def CustomGradOp : Gradient_Op<"custom_grad",
        [DeclareOpInterfaceMethods<SymbolUserOpInterface>]> {

  let summary = "Operation denoting the registration of the custom gradient with Enzyme.";

  let description = [{
    A triple of three functions. The function itself, the forward and reverse pass.
  }];

  let arguments = (ins
    FlatSymbolRefAttr: $callee,
    FlatSymbolRefAttr: $forward,
    FlatSymbolRefAttr: $reverse
  );

  let assemblyFormat = [{
    $callee $forward $reverse attr-dict
  }];

}

#endif // GRADIENT_OPS

