// Copyright 2024 Xanadu Quantum Technologies Inc.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "mlir/IR/DialectImplementation.h" // needed for generated type parser
#include "llvm/ADT/TypeSwitch.h"           // needed for generated type parser

#include "Ion/IR/IonDialect.h"
#include "Ion/IR/IonOps.h"

using namespace mlir;
using namespace catalyst::ion;

//===----------------------------------------------------------------------===//
// Ion dialect definitions.
//===----------------------------------------------------------------------===//

#include "Ion/IR/IonOpsDialect.cpp.inc"

void IonDialect::initialize()
{
    addTypes<
#define GET_TYPEDEF_LIST
#include "Ion/IR/IonOpsTypes.cpp.inc"
        >();

    addAttributes<
#define GET_ATTRDEF_LIST
#include "Ion/IR/IonAttributes.cpp.inc"
        >();

    addOperations<
#define GET_OP_LIST
#include "Ion/IR/IonOps.cpp.inc"
        >();
}

//===----------------------------------------------------------------------===//
// Ion type definitions.
//===----------------------------------------------------------------------===//

#define GET_TYPEDEF_CLASSES
#include "Ion/IR/IonOpsTypes.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "Ion/IR/IonAttributes.cpp.inc"
