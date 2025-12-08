// Copyright 2025 Xanadu Quantum Technologies Inc.

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

#include "PauliFrame/IR/PauliFrameDialect.h"
#include "PauliFrame/IR/PauliFrameOps.h"

using namespace mlir;
using namespace catalyst::pauli_frame;

//===----------------------------------------------------------------------===//
// PauliFrame dialect definitions.
//===----------------------------------------------------------------------===//

#include "PauliFrame/IR/PauliFrameOpsDialect.cpp.inc"

void PauliFrameDialect::initialize()
{
    /// Uncomment the lines below if defining types for the PauliFrame dialect
    //     addTypes<
    // #define GET_TYPEDEF_LIST
    // #include "PauliFrame/IR/PauliFrameOpsTypes.cpp.inc"
    //         >();

    addAttributes<
#define GET_ATTRDEF_LIST
#include "PauliFrame/IR/PauliFrameAttributes.cpp.inc"
        >();

    addOperations<
#define GET_OP_LIST
#include "PauliFrame/IR/PauliFrameOps.cpp.inc"
        >();
}

//===----------------------------------------------------------------------===//
// PauliFrame type definitions.
//===----------------------------------------------------------------------===//

/// Uncomment the lines below if defining types for the PauliFrame dialect
// #define GET_TYPEDEF_CLASSES
// #include "PauliFrame/IR/PauliFrameOpsTypes.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "PauliFrame/IR/PauliFrameAttributes.cpp.inc"
