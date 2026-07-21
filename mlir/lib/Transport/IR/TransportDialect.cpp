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

#include "Transport/IR/TransportDialect.h"

#include "llvm/ADT/TypeSwitch.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"

#include "Transport/IR/TransportOps.h"

using namespace mlir;
using namespace catalyst::transport;

//===----------------------------------------------------------------------===//
// Transport dialect definitions.
//===----------------------------------------------------------------------===//

#include "Transport/IR/TransportOpsDialect.cpp.inc"

//===----------------------------------------------------------------------===//
// Transport type definitions.
//===----------------------------------------------------------------------===//

#define GET_TYPEDEF_CLASSES
#include "Transport/IR/TransportOpsTypes.cpp.inc"

void TransportDialect::initialize()
{
    addTypes<
#define GET_TYPEDEF_LIST
#include "Transport/IR/TransportOpsTypes.cpp.inc"
        >();

    addOperations<
#define GET_OP_LIST
#include "Transport/IR/TransportOps.cpp.inc"
        >();
}
