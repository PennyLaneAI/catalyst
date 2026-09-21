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

#include "Catalyst/IR/CatalystOps.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"

#include "Catalyst/IR/CatalystDialect.h"
#include "Catalyst/IR/RuntimeCABI.h"

using namespace mlir;
using namespace catalyst;

#define GET_OP_CLASSES
#include "Catalyst/IR/CatalystOps.cpp.inc"

void CustomCallOp::getEffects(
    llvm::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>
        &effects) {
    // Assume all effects
    effects.emplace_back(mlir::MemoryEffects::Allocate::get());
    effects.emplace_back(mlir::MemoryEffects::Free::get());
    effects.emplace_back(mlir::MemoryEffects::Write::get());
    effects.emplace_back(mlir::MemoryEffects::Read::get());
}

static bool isRuntimeCABIType(StringRef name, bool allowVoid) {
    std::optional<RuntimeCABIType> abi = classifyRuntimeCABIType(name);
    return abi.has_value() && (allowVoid || abi->kind != RuntimeCABIKind::Void);
}

LogicalResult RuntimeCallOp::verify() {
    StringRef result = getCResult();
    if (!isRuntimeCABIType(result, /*allowVoid=*/true)) {
        return emitOpError("unsupported native C ABI result type '") << result << "'";
    }

    unsigned numOut = 0;
    unsigned numStr = 0;
    unsigned numInputs = 0;
    for (Attribute attr : getCParams()) {
        StringRef param = cast<StringAttr>(attr).getValue();
        if (!isRuntimeCABIType(param, /*allowVoid=*/false)) {
            return emitOpError("unsupported native C ABI parameter type '") << param << "'";
        }
        if (param == "out") {
            ++numOut;
            continue;
        }
        if (param == "str") {
            ++numStr;
            continue;
        }
        if (param == "buf" && isDispatched()) {
            return emitOpError("a buf names memory in this process and cannot be dispatched to '")
                   << *getDispatch() << "'";
        }

        if (numInputs < getInputs().size()) {
            Type type = getInputs()[numInputs].getType();
            if (param == "buf" && !isa<RankedTensorType, MemRefType>(type)) {
                return emitOpError("buf parameter must be a ranked tensor or memref, but got ")
                       << type;
            }
            if (param != "buf" && isa<ShapedType>(type)) {
                return emitOpError("scalar/ptr parameter must remain an SSA scalar, but got ")
                       << type;
            }
        }
        ++numInputs;
    }

    if (getInputs().size() != numInputs) {
        return emitOpError("expected ")
               << numInputs << " input operand(s) for the declared C parameters, but got "
               << getInputs().size();
    }

    unsigned numStrings = getCStrings().has_value() ? getCStrings()->size() : 0;
    if (numStrings != numStr) {
        return emitOpError("expected ")
               << numStr << " compile-time string(s), but got " << numStrings;
    }

    // A dispatched call is rewritten into an executor.call before bufferization, so it must never
    // acquire destination buffers.
    if (isDispatched() && !getDestBuffers().empty()) {
        return emitOpError("a dispatched call must be lowered before bufferization");
    }
    if (!getOutTensors().empty() && !getDestBuffers().empty()) {
        return emitOpError("out tensors and destination buffers cannot both be present");
    }
    unsigned numOutValues = getOutTensors().size() + getDestBuffers().size();
    if (numOutValues != numOut) {
        return emitOpError("expected ")
               << numOut << " out tensor result(s) or destination buffer(s), but got "
               << numOutValues;
    }

    unsigned numScalars = result == "void" ? 0 : 1;
    if (getScalarResult().size() != numScalars) {
        return emitOpError("expected ") << numScalars << " scalar result(s) for C result type '"
                                        << result << "', but got " << getScalarResult().size();
    }

    return success();
}

void RuntimeCallOp::getEffects(
    llvm::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>
        &effects) {
    // Assume all effects
    effects.emplace_back(mlir::MemoryEffects::Allocate::get());
    effects.emplace_back(mlir::MemoryEffects::Free::get());
    effects.emplace_back(mlir::MemoryEffects::Write::get());
    effects.emplace_back(mlir::MemoryEffects::Read::get());
}

void CallbackCallOp::getEffects(
    llvm::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>
        &effects) {
    // Assume all effects
    effects.emplace_back(mlir::MemoryEffects::Allocate::get());
    effects.emplace_back(mlir::MemoryEffects::Free::get());
    effects.emplace_back(mlir::MemoryEffects::Write::get());
    effects.emplace_back(mlir::MemoryEffects::Read::get());
}

LogicalResult CallbackCallOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
    auto callee = this->getCalleeAttr();
    auto sym = symbolTable.lookupNearestSymbolFrom(this->getOperation(), callee);
    if (!sym) {
        this->emitOpError("invalid function:") << callee;
        return failure();
    }

    return success();
}

LogicalResult LaunchKernelOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
    auto callee = this->getCalleeAttr();
    SymbolOpInterface sym =
        symbolTable.lookupNearestSymbolFrom<SymbolOpInterface>(this->getOperation(), callee);
    if (sym && sym.getVisibility() == mlir::SymbolTable::Visibility::Public) {
        return success();
    }
    this->emitOpError("invalid function:") << callee;
    return failure();
}
