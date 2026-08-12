// Copyright @@@year@@@ Xanadu Quantum Technologies Inc.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h" // needed for generated type parser
#include "llvm/ADT/TypeSwitch.h"           // needed for generated type parser

#include "@@@NewDialect@@@/IR/@@@NewDialect@@@Dialect.h"
#include "@@@NewDialect@@@/IR/@@@NewDialect@@@Ops.h"

using namespace mlir;
using namespace catalyst::@@@new_dialect@@@;

//===----------------------------------------------------------------------===//
// @@@NewDialect@@@ dialect definitions.
//===----------------------------------------------------------------------===//

#include "@@@NewDialect@@@/IR/@@@NewDialect@@@OpsDialect.cpp.inc"

void @@@NewDialect@@@Dialect::initialize()
{
    /// Uncomment the lines below if defining types for the @@@NewDialect@@@ dialect
    //     addTypes<
    // #define GET_TYPEDEF_LIST
    // #include "@@@NewDialect@@@/IR/@@@NewDialect@@@OpsTypes.cpp.inc"
    //         >();

    addAttributes<
#define GET_ATTRDEF_LIST
#include "@@@NewDialect@@@/IR/@@@NewDialect@@@Attributes.cpp.inc"
        >();

    addOperations<
#define GET_OP_LIST
#include "@@@NewDialect@@@/IR/@@@NewDialect@@@Ops.cpp.inc"
        >();
}

//===----------------------------------------------------------------------===//
// @@@NewDialect@@@ type definitions.
//===----------------------------------------------------------------------===//

/// Uncomment the lines below if defining types for the @@@NewDialect@@@ dialect
// #define GET_TYPEDEF_CLASSES
// #include "@@@NewDialect@@@/IR/@@@NewDialect@@@OpsTypes.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "@@@NewDialect@@@/IR/@@@NewDialect@@@Attributes.cpp.inc"
