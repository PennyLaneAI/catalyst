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

#ifndef QEC_LOGICAL_ATTR_DEFS
#define QEC_LOGICAL_ATTR_DEFS

include "mlir/IR/EnumAttr.td"

include "QecLogical/IR/QecLogicalDialect.td"

//===----------------------------------------------------------------------===//
// QecLogical dialect enums and attributes.
//===----------------------------------------------------------------------===//

def LogicalCodeblockInitState : I32EnumAttr<"LogicalCodeblockInitState",
    "logical codeblock initial state",
    [
        I32EnumAttrCase<"Zero", 0, "zero">  // |0⟩
        // Add other supported codeblock initial states here
    ]> {
    let cppNamespace = "catalyst::qecl";
    let genSpecializedAttr = 0;
}

def LogicalCodeblockInitStateAttr : EnumAttr<
    QecLogical_Dialect, LogicalCodeblockInitState, "codeblock_init_state">;

#endif // QEC_LOGICAL_ATTR_DEFS
