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

#pragma once

#include <optional>

#include "llvm/ADT/StringRef.h"

// #include "mlir/Dialect/Bufferization/IR/AllocationOpInterface.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/PatternMatch.h"
// #include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Support/LogicalResult.h"

// TODO: is it dependent on the regular quantum dialect?
// #include "Quantum/IR/QuantumDialect.h"

#include "RefQuantum/IR/RefQuantumDialect.h"
#include "RefQuantum/IR/RefQuantumInterfaces.h"

//===----------------------------------------------------------------------===//
// RefQuantum trait declarations.
//===----------------------------------------------------------------------===//

namespace mlir {
namespace OpTrait {

template <typename ConcreteType>
class UnitaryTrait : public TraitBase<ConcreteType, UnitaryTrait> {};

template <typename ConcreteType>
class HermitianTrait : public TraitBase<ConcreteType, HermitianTrait> {};

} // namespace OpTrait
} // namespace mlir

//===----------------------------------------------------------------------===//
// RefQuantum ops declarations.
//===----------------------------------------------------------------------===//

#include "RefQuantum/IR/RefQuantumEnums.h.inc"

#define GET_ATTRDEF_CLASSES
#include "RefQuantum/IR/RefQuantumAttributes.h.inc"

#define GET_OP_CLASSES
#include "RefQuantum/IR/RefQuantumOps.h.inc"
