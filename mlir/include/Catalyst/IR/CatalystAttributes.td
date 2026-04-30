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

#ifndef CATALYST_ATTRIBUTES
#define CATALYST_ATTRIBUTES

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinAttributes.td"
include "mlir/IR/CommonAttrConstraints.td"

include "Catalyst/IR/CatalystDialect.td"

//===----------------------------------------------------------------------===//
// Catalyst memory space attribute.
//===----------------------------------------------------------------------===//

class Catalyst_Attr<string name, string attrMnemonic, list<Trait> traits = [],
                    string baseCppClass = "::mlir::Attribute">
    : AttrDef<Catalyst_Dialect, name, traits, baseCppClass> {
  let mnemonic = attrMnemonic;
}

def MemSpaceAttr : Catalyst_Attr<"MemSpace", "memspace"> {
  let summary = "Catalyst memory-space attribute: memory domain plus optional address space.";
  let description = [{
    Generic memory-space attribute for use in the `memory_space` slot of
    `memref` types (or as a discardable attribute in general) denoting a memory domain with an 
    optional address space within that domain.

    Syntax:

    ```mlir
    #catalyst.memspace<"<domain>">
    #catalyst.memspace<"<domain>", <addr_space> : i64>
    ```

    Examples:

    ```mlir
    memref<4x64xi8, #catalyst.memspace<"d0">>
    memref<1xi32,   #catalyst.memspace<"d1", 1 : i64>>
    ```
  }];

  let parameters = (ins
    StringRefParameter<"Symbolic memory domain name">:$domain,
    OptionalParameter<"::mlir::IntegerAttr", "Optional explicit LLVM address space">:$addrSpace
  );

  let assemblyFormat =
      "`<` $domain (`,` $addrSpace^)? `>`";

  let genVerifyDecl = 1;
}

#endif // CATALYST_ATTRIBUTES
