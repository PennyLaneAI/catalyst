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

#include "llvm/ADT/TypeSwitch.h" // needed for enums

#include "mlir/IR/DialectImplementation.h"

#include "PBC/IR/PBCDialect.h"
#include "PBC/IR/PBCOps.h"
#include "Quantum/IR/QuantumDialect.h"

using namespace mlir;
using namespace catalyst::pbc;

//===----------------------------------------------------------------------===//
// PBC dialect definitions.
//===----------------------------------------------------------------------===//

#include "PBC/IR/PBCOpsDialect.cpp.inc"

void PBCDialect::initialize()
{
    addTypes<
#define GET_TYPEDEF_LIST
#include "PBC/IR/PBCOpsTypes.cpp.inc"
        >();
    addAttributes<
#define GET_ATTRDEF_LIST
#include "PBC/IR/PBCAttributes.cpp.inc"
        >();
    addOperations<
#define GET_OP_LIST
#include "PBC/IR/PBCOps.cpp.inc"
        >();
}

//===----------------------------------------------------------------------===//
// PBC type definitions.
//===----------------------------------------------------------------------===//

#define GET_TYPEDEF_CLASSES
#include "PBC/IR/PBCOpsTypes.cpp.inc"

//===----------------------------------------------------------------------===//
// PBC enum definitions.
//===----------------------------------------------------------------------===//

#include "PBC/IR/PBCEnums.cpp.inc"

//===----------------------------------------------------------------------===//
// PBC attribute definitions.
//===----------------------------------------------------------------------===//

#define GET_ATTRDEF_CLASSES
#include "PBC/IR/PBCAttributes.cpp.inc"
