// Copyright 2023 Xanadu Quantum Technologies Inc.

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
#include "mlir/IR/OpImplementation.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"

#include "Catalyst/IR/CatalystDialect.h"
#include "Catalyst/IR/CatalystOps.h"

using namespace mlir;
using namespace catalyst;

#define GET_OP_CLASSES
#include "Catalyst/IR/CatalystOps.cpp.inc"

void ListInitOp::getEffects(
    llvm::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>> &effects)
{
    effects.emplace_back(mlir::MemoryEffects::Allocate::get());
}

void CustomCallOp::getEffects(
    llvm::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>> &effects)
{
    // Assume all effects
    effects.emplace_back(mlir::MemoryEffects::Allocate::get());
    effects.emplace_back(mlir::MemoryEffects::Free::get());
    effects.emplace_back(mlir::MemoryEffects::Write::get());
    effects.emplace_back(mlir::MemoryEffects::Read::get());
}

void CallbackCallOp::getEffects(
    llvm::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>> &effects)
{
    // Assume all effects
    effects.emplace_back(mlir::MemoryEffects::Allocate::get());
    effects.emplace_back(mlir::MemoryEffects::Free::get());
    effects.emplace_back(mlir::MemoryEffects::Write::get());
    effects.emplace_back(mlir::MemoryEffects::Read::get());
}

LogicalResult CallbackCallOp::verifySymbolUses(SymbolTableCollection &symbolTable)
{
    auto callee = this->getCalleeAttr();
    auto sym = symbolTable.lookupNearestSymbolFrom(this->getOperation(), callee);
    if (!sym) {
        this->emitOpError("invalid function:") << callee;
        return failure();
    }

    return success();
}
