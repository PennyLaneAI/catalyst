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

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h" // needed for generated type parser
#include "llvm/ADT/TypeSwitch.h"           // needed for generated type parser

#include "QecPhysical/IR/QecPhysicalDialect.h"
#include "QecPhysical/IR/QecPhysicalOps.h"

using namespace mlir;
using namespace catalyst::qecp;

//===----------------------------------------------------------------------===//
// QecPhysical dialect definitions.
//===----------------------------------------------------------------------===//

#include "QecPhysical/IR/QecPhysicalOpsDialect.cpp.inc"

void QecPhysicalDialect::initialize()
{
    addTypes<
#define GET_TYPEDEF_LIST
#include "QecPhysical/IR/QecPhysicalOpsTypes.cpp.inc"
        >();

    addOperations<
#define GET_OP_LIST
#include "QecPhysical/IR/QecPhysicalOps.cpp.inc"
        >();
}

//===----------------------------------------------------------------------===//
// QecPhysical type definitions.
//===----------------------------------------------------------------------===//

#define GET_TYPEDEF_CLASSES
#include "QecPhysical/IR/QecPhysicalOpsTypes.cpp.inc"
