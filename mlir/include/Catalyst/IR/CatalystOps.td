
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

include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/IR/BuiltinAttributes.td"
include "mlir/IR/OpBase.td"

include "Catalyst/IR/CatalystDialect.td"

def ListInitOp : Catalyst_Op<"list_init"> {
    let summary = "Initialize a dynamically resizable arraylist.";
    let results = (outs ArrayListType:$list);
    let assemblyFormat = [{ attr-dict `:` type($list) }];
}

def ListDeallocOp : Catalyst_Op<"list_dealloc"> {
    let summary = "Deallocate the underlying memory of an arraylist.";
    let arguments = (ins ArrayListType:$list);
    let assemblyFormat = [{ $list attr-dict `:` type($list) }];
}

def ListPushOp : Catalyst_Op<"list_push",
        [TypesMatchWith<"type of 'value' matches element type of 'list'",
                        "list", "value",
                        "$_self.cast<ArrayListType>().getElementType()">]> {
    let summary = "Append an element to the end of an array list.";
    let arguments = (ins AnyType:$value, ArrayListType:$list);
    let assemblyFormat = [{ $value `,` $list attr-dict `:` type($list) }];
}

def ListPopOp : Catalyst_Op<"list_pop",
        [TypesMatchWith<"type of 'result' matches element type of 'list'",
                        "list", "result",
                        "$_self.cast<ArrayListType>().getElementType()">]> {
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

def PrintOp : Catalyst_Op<"print"> {
    let summary = "Prints numeric values or constant strings at runtime.";

    let arguments = (ins
        Optional<AnyType>:$val,
        OptionalAttr<StrAttr>:$const_val
    );

    let assemblyFormat = [{
        ($const_val^)? ($val^ `:` type($val))? attr-dict
    }];
}

#endif // GRADIENT_OPS
