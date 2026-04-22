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

#include "Catalyst/Utils/ConstantResolve.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/Operation.h"

using namespace mlir;

namespace catalyst {

/// Extract a double from a constant-like Attribute.
static std::optional<double> attrToDouble(Attribute attr)
{
    if (auto floatAttr = dyn_cast<FloatAttr>(attr)) {
        return floatAttr.getValueAsDouble();
    }
    if (auto intAttr = dyn_cast<IntegerAttr>(attr)) {
        return static_cast<double>(intAttr.getValue().getSExtValue());
    }
    if (auto denseFP = dyn_cast<DenseFPElementsAttr>(attr)) {
        if (denseFP.isSplat() || denseFP.getNumElements() == 1) {
            return denseFP.getSplatValue<APFloat>().convertToDouble();
        }
    }
    if (auto denseInt = dyn_cast<DenseIntElementsAttr>(attr)) {
        if (denseInt.isSplat() || denseInt.getNumElements() == 1) {
            return static_cast<double>(denseInt.getSplatValue<APInt>().getSExtValue());
        }
    }
    return std::nullopt;
}

std::optional<double> resolveConstantArithmetic(Value val, Operation *op)
{
    // arith integer binary ops
    if (isa<arith::AddIOp, arith::SubIOp, arith::MulIOp>(op)) {
        auto lhs = resolveConstant(op->getOperand(0));
        auto rhs = resolveConstant(op->getOperand(1));
        if (!lhs || !rhs) {
            return std::nullopt;
        }
        if (isa<arith::AddIOp>(op)) {
            return *lhs + *rhs;
        }
        if (isa<arith::SubIOp>(op)) {
            return *lhs - *rhs;
        }
        return *lhs * *rhs;
    }

    // arith.addf
    if (isa<arith::AddFOp>(op)) {
        auto lhs = resolveConstant(op->getOperand(0));
        auto rhs = resolveConstant(op->getOperand(1));
        if (lhs && rhs) {
            return *lhs + *rhs;
        }
        return std::nullopt;
    }

    return std::nullopt;
}

std::optional<double> resolveConstant(Value val)
{
    if (!val) {
        return std::nullopt;
    }

    Operation *op = val.getDefiningOp();
    if (!op) {
        return std::nullopt;
    }

    // arith.constant
    if (auto constOp = dyn_cast<arith::ConstantOp>(op)) {
        return attrToDouble(constOp.getValue());
    }

    // arith.index_cast / arith.index_castui — pass through
    if (isa<arith::IndexCastOp, arith::IndexCastUIOp>(op)) {
        return resolveConstant(op->getOperand(0));
    }

    // Arithmetic operations
    auto res = resolveConstantArithmetic(val, op);
    if (res != std::nullopt) {
        return res;
    }

    // tensor.extract
    if (auto extractOp = dyn_cast<tensor::ExtractOp>(op)) {
        return resolveConstant(extractOp.getTensor());
    }

    // tensor.from_elements
    if (auto fromElemOp = dyn_cast<tensor::FromElementsOp>(op)) {
        if (fromElemOp.getNumOperands() > 0) {
            return resolveConstant(fromElemOp.getOperand(0));
        }
        return std::nullopt;
    }

    // stablehlo ops (matched by name to avoid header dependency)
    // OpName is used to avoid header dependency
    StringRef opName = op->getName().getStringRef();

    if (opName == "stablehlo.constant") {
        if (auto attr = op->getAttr("value")) {
            return attrToDouble(attr);
        }
        return std::nullopt;
    }

    if (opName == "stablehlo.convert" || opName == "stablehlo.broadcast_in_dim") {
        if (op->getNumOperands() > 0) {
            return resolveConstant(op->getOperand(0));
        }
        return std::nullopt;
    }

    if (opName == "stablehlo.add" || opName == "stablehlo.subtract") {
        auto lhs = resolveConstant(op->getOperand(0));
        auto rhs = resolveConstant(op->getOperand(1));
        if (!lhs || !rhs) {
            return std::nullopt;
        }
        return (opName == "stablehlo.add") ? *lhs + *rhs : *lhs - *rhs;
    }

    // Generic dense splat constant (catch-all via matchPattern)
    DenseIntElementsAttr denseAttr;
    if (op->getNumResults() > 0 && matchPattern(op->getResult(0), m_Constant(&denseAttr)) &&
        denseAttr.isSplat()) {
        return static_cast<double>(denseAttr.getSplatValue<llvm::APInt>().getSExtValue());
    }

    return std::nullopt;
}

std::optional<int64_t> resolveConstantInt(Value val)
{
    auto result = resolveConstant(val);
    if (!result) {
        return std::nullopt;
    }
    return static_cast<int64_t>(*result);
}

} // namespace catalyst
