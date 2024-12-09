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

#pragma once

#include <optional>

#include "mlir/Dialect/Bufferization/IR/AllocationOpInterface.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/ErrorHandling.h"

#include "Ion/IR/IonInterfaces.h"

//===----------------------------------------------------------------------===//
// Ion trait declarations.
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Ion ops declarations.
//===----------------------------------------------------------------------===//

#include "Ion/IR/IonDialect.h"
// #include "Ion/IR/IonEnums.h.inc"
#define GET_ATTRDEF_CLASSES
#include "Ion/IR/IonAttributes.h.inc"
#define GET_OP_CLASSES
#include "Ion/IR/IonOps.h.inc"
