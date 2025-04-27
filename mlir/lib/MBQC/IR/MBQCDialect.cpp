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

#include "MBQC/IR/MBQCDialect.h"
#include "MBQC/IR/MBQCOps.h"

using namespace mlir;
using namespace catalyst::mbqc;

//===----------------------------------------------------------------------===//
// MBQC dialect definitions.
//===----------------------------------------------------------------------===//

#include "MBQC/IR/MBQCOpsDialect.cpp.inc"

void MBQCDialect::initialize()
{
    /// Uncomment the lines below if defining types for the MBQC dialect
    //     addTypes<
    // #define GET_TYPEDEF_LIST
    // #include "MBQC/IR/MBQCOpsTypes.cpp.inc"
    //         >();

    addAttributes<
#define GET_ATTRDEF_LIST
#include "MBQC/IR/MBQCAttributes.cpp.inc"
        >();

    addOperations<
#define GET_OP_LIST
#include "MBQC/IR/MBQCOps.cpp.inc"
        >();
}

//===----------------------------------------------------------------------===//
// MBQC type definitions.
//===----------------------------------------------------------------------===//

/// Uncomment the lines below if defining types for the MBQC dialect
// #define GET_TYPEDEF_CLASSES
// #include "MBQC/IR/MBQCOpsTypes.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "MBQC/IR/MBQCAttributes.cpp.inc"
