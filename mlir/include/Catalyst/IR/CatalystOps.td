
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

#ifndef CATALYST_OPS
#define CATALYST_OPS

include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/FunctionInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/IR/BuiltinAttributes.td"
include "mlir/IR/OpBase.td"

include "Catalyst/IR/CatalystDialect.td"

def ListInitOp : Catalyst_Op<"list_init", [MemoryEffects<[MemAlloc]>]> {
    let summary = "Initialize a dynamically resizable arraylist.";
    let results = (outs ArrayListType:$list);
    let assemblyFormat = [{ attr-dict `:` type($list) }];
}

def ListDeallocOp : Catalyst_Op<"list_dealloc", [MemoryEffects<[MemFree]>]> {
    let summary = "Deallocate the underlying memory of an arraylist.";
    let arguments = (ins ArrayListType:$list);
    let assemblyFormat = [{ $list attr-dict `:` type($list) }];
}

def ListPushOp : Catalyst_Op<"list_push",
        [TypesMatchWith<"type of 'value' matches element type of 'list'",
                        "list", "value",
                        "mlir::cast<ArrayListType>($_self).getElementType()">,
        MemoryEffects<[MemAlloc, MemWrite]>]> {
    let summary = "Append an element to the end of an array list.";
    let arguments = (ins AnyType:$value, ArrayListType:$list);
    let assemblyFormat = [{ $value `,` $list attr-dict `:` type($list) }];
}

def ListPopOp : Catalyst_Op<"list_pop",
        [TypesMatchWith<"type of 'result' matches element type of 'list'",
                        "list", "result",
                        "mlir::cast<ArrayListType>($_self).getElementType()">,
        MemoryEffects<[MemRead, MemFree]>]> {
    let summary = "Remove an element from the end of an array list and return it.";
    let arguments = (ins ArrayListType:$list);
    let results = (outs AnyType:$result);
    let assemblyFormat = [{ $list attr-dict `:` type($list) }];
}

def ListLoadDataOp : Catalyst_Op<"list_load_data"> {
    let summary = "Get the underlying memref storing the data of an array list.";
    let arguments = (ins ArrayListType:$list);
    let results = (outs AnyMemRef:$data);
    let builders = [
        OpBuilder<(ins "mlir::Value":$list), [{
            ArrayListType listType = cast<ArrayListType>(list.getType());
            MemRefType memrefType = MemRefType::get({ShapedType::kDynamic}, listType.getElementType());
            return build($_builder, $_state, memrefType, list);
        }]>
    ];
    let assemblyFormat = [{ $list attr-dict `:` type($list) `->` type($data) }];
}

def PrintOp : Catalyst_Op<"print", [MemoryEffects<[MemWrite]>]> {
    let summary = "Prints numeric values or constant strings at runtime.";

    let arguments = (ins
        Optional<AnyType>:$val,
        OptionalAttr<StrAttr>:$const_val,
        UnitAttr:$print_descriptor
    );
}

def AssertionOp : Catalyst_Op<"assert", [MemoryEffects<[MemWrite]>]> {
    let summary = "Asserts condition at runtime.";

    let arguments = (ins
      I1:$assertion,
      StrAttr:$error
    );
}

def CustomCallOp: Catalyst_Op<"custom_call",
    [MemoryEffects<[MemAlloc, MemFree, MemWrite, MemRead]>]> {
  let summary = "CustomCall operation";
  let description = [{
    Encapsulates an implementation-defined operation `call_target_name` that
    takes `inputs` and and produces `results`.

    A custom call invokes code external to Catalyst. The `inputs` are passed to the
    external code, and the external code is expected to produce a result of the
    given type.
    ```
  }];
  let arguments = (ins
    Variadic<AnyType>:$inputs,
    StrAttr:$call_target_name,
    OptionalAttr<DenseI32ArrayAttr>: $number_original_arg
  );
  let results = (outs Variadic<AnyType>);

  let assemblyFormat = [{
    `fn` `(`$call_target_name`)` `(` $inputs `)`
      attr-dict `:` functional-type(operands, results)
  }];
}

def CallbackOp : Catalyst_Op<"callback",
    [FunctionOpInterface, IsolatedFromAbove]> {

  let summary = "Operation denoting a symbol to refer to user callbacks.";

  let description = [{
     This is an operation that is intended to be placed at the module level.
     It corresponds to function bodies that are not yet constructed.
  }];

  let arguments = (ins
     SymbolNameAttr: $sym_name,
     TypeAttrOf<FunctionType>: $function_type,
     I64Attr: $id,
     I64Attr: $argc,
     I64Attr: $resc,
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

def CallbackCallOp : Catalyst_Op<"callback_call",
    [MemoryEffects<[MemAlloc, MemFree, MemWrite, MemRead]>,
     DeclareOpInterfaceMethods<CallOpInterface>,
     DeclareOpInterfaceMethods<SymbolUserOpInterface>
    ]> {

  let arguments = (ins
        FlatSymbolRefAttr:$callee,
        Variadic<AnyTypeOf<[AnyRankedTensor, MemRefOf<[AnyType]>]>>:$inputs
  );

  let results = (outs Variadic<AnyType>);

  let assemblyFormat = [{
    $callee `(` $inputs `)` attr-dict `:` functional-type($inputs, results)
  }];

}

#endif // GRADIENT_OPS
