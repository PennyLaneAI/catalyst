// Copyright 2022-2023 Xanadu Quantum Technologies Inc.

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
#include <optional>

#include "Quantum/IR/QuantumInterfaces.h"

//===----------------------------------------------------------------------===//
// Quantum trait declarations.
//===----------------------------------------------------------------------===//

namespace mlir {
namespace OpTrait {

template <typename ConcreteType>
class UnitaryTrait : public TraitBase<ConcreteType, UnitaryTrait> {};

template <typename ConcreteType>
class HermitianTrait : public TraitBase<ConcreteType, HermitianTrait> {};

} // namespace OpTrait
} // namespace mlir

class QuantumMemory : public mlir::SideEffects::Resource::Base<QuantumMemory> {
    llvm::StringRef getName() final { return "QuantumMemory"; }
};

//===----------------------------------------------------------------------===//
// Quantum ops declarations.
//===----------------------------------------------------------------------===//

#include "Quantum/IR/QuantumDialect.h"
#include "Quantum/IR/QuantumEnums.h.inc"
#define GET_ATTRDEF_CLASSES
#include "Quantum/IR/QuantumAttributes.h.inc"
#define GET_OP_CLASSES
#include "Quantum/IR/QuantumOps.h.inc"
