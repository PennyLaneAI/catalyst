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

#include <cstddef>
#include <optional>
#include <utility>

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LogicalResult.h"

#include "QRef/IR/QRefInterfaces.h"
#include "QRef/IR/QRefOps.h"
#include "QRef/IR/QRefTypes.h"

namespace catalyst::quantum {

/// Adapts a reference-semantics gate to a decomposition rule's function ABI.
///
/// In register-mode rules, qref.get keeps the register and wire index directly attached to every
/// qubit reference. This lets lowering collect the register in constant time instead of walking
/// backwards through quantum.insert chains.
class ReferenceSignatureAnalyzer {
  public:
    explicit ReferenceSignatureAnalyzer(catalyst::qref::DecomposableGate gate) : gate(gate) {}

    mlir::FailureOr<llvm::SmallVector<mlir::Value>>
    prepareOperands(mlir::func::FuncOp rule, mlir::PatternRewriter &rewriter) {
        mlir::ArrayRef<mlir::Type> inputTypes = rule.getFunctionType().getInputs();
        auto qregIt = llvm::find_if(inputTypes, llvm::IsaPred<catalyst::qref::QuregType>);
        const bool hasQreg = qregIt != inputTypes.end();
        const size_t qregIndex = std::distance(inputTypes.begin(), qregIt);

        llvm::SmallVector<mlir::Type> nonQregTypes;
        llvm::copy_if(inputTypes, std::back_inserter(nonQregTypes),
                      [](mlir::Type type) { return !mlir::isa<catalyst::qref::QuregType>(type); });

        llvm::SmallVector<mlir::Value> nonQregOperands;
        auto params = mlir::dyn_cast<catalyst::qref::ParametrizedGate>(gate.getOperation());
        mlir::ValueRange paramValues = params ? params.getAllParams() : mlir::ValueRange{};
        if (mlir::failed(
                appendParams(rule, paramValues, nonQregTypes, nonQregOperands, rewriter))) {
            return mlir::failure();
        }

        mlir::Value qreg;
        if (hasQreg) {
            if (mlir::failed(
                    appendRegisterOperands(rule, nonQregTypes, nonQregOperands, qreg, rewriter))) {
                return mlir::failure();
            }
        } else if (mlir::failed(
                       appendQubitOperands(rule, nonQregTypes, nonQregOperands, rewriter))) {
            return mlir::failure();
        }

        if (nonQregOperands.size() != nonQregTypes.size()) {
            rule.emitError() << "decomposition rule has " << nonQregTypes.size()
                             << " non-register inputs but only " << nonQregOperands.size()
                             << " target-operation operands were available";
            return mlir::failure();
        }

        llvm::SmallVector<mlir::Value> operands = std::move(nonQregOperands);
        if (hasQreg) {
            operands.insert(operands.begin() + qregIndex, qreg);
        }
        return operands;
    }

  private:
    struct ReferenceIndex {
        mlir::Value value;
        mlir::IntegerAttr attr;
        mlir::Value qreg;
    };

    catalyst::qref::DecomposableGate gate;

    static size_t getElementCount(mlir::Type type) {
        if (auto tensorType = mlir::dyn_cast<mlir::RankedTensorType>(type)) {
            return tensorType.getNumElements() > 0 ? tensorType.getNumElements() : 1;
        }
        return 1;
    }

    static mlir::FailureOr<std::pair<size_t, size_t>>
    findTypeRange(mlir::ArrayRef<mlir::Type> types, size_t valueCount, size_t start) {
        size_t count = 0;
        size_t end = start;
        while (count < valueCount && end < types.size()) {
            count += getElementCount(types[end++]);
        }
        if (count != valueCount) {
            return mlir::failure();
        }
        return std::pair{start, end};
    }

    static mlir::FailureOr<mlir::Value> packValues(mlir::ValueRange values, mlir::Type type,
                                                   mlir::Location loc,
                                                   mlir::PatternRewriter &rewriter) {
        if (mlir::isa<mlir::RankedTensorType>(type)) {
            return mlir::tensor::FromElementsOp::create(rewriter, loc, type, values).getResult();
        }
        if (values.size() != 1 || values.front().getType() != type) {
            return mlir::failure();
        }
        return values.front();
    }

    static mlir::LogicalResult appendParams(mlir::func::FuncOp rule, mlir::ValueRange params,
                                            mlir::ArrayRef<mlir::Type> inputTypes,
                                            llvm::SmallVectorImpl<mlir::Value> &operands,
                                            mlir::PatternRewriter &rewriter) {
        if (params.empty()) {
            return mlir::success();
        }

        bool hasMatchingPrefix =
            params.size() <= inputTypes.size() &&
            llvm::all_of(llvm::enumerate(params), [&](auto indexedParam) {
                return indexedParam.value().getType() == inputTypes[indexedParam.index()];
            });
        if (hasMatchingPrefix) {
            llvm::append_range(operands, params);
            return mlir::success();
        }

        auto typeRange = findTypeRange(inputTypes, params.size(), operands.size());
        if (mlir::failed(typeRange)) {
            rule.emitError("decomposition rule parameter ABI does not match target operation");
            return mlir::failure();
        }

        size_t paramIndex = 0;
        for (mlir::Type type :
             inputTypes.slice(typeRange->first, typeRange->second - typeRange->first)) {
            size_t count = getElementCount(type);
            auto packed =
                packValues(params.slice(paramIndex, count), type, rule.getLoc(), rewriter);
            if (mlir::failed(packed)) {
                rule.emitError("decomposition rule parameter types do not match target operation");
                return mlir::failure();
            }
            operands.push_back(*packed);
            paramIndex += count;
        }
        return mlir::success();
    }

    static mlir::FailureOr<ReferenceIndex> getReferenceIndex(mlir::Value qubit) {
        auto get = qubit.getDefiningOp<catalyst::qref::GetOp>();
        if (!get) {
            return mlir::failure();
        }
        return ReferenceIndex{get.getIdx(), get.getIdxAttrAttr(), get.getQreg()};
    }

    static mlir::FailureOr<mlir::Value> packIndices(mlir::ArrayRef<ReferenceIndex> indices,
                                                    mlir::Type type, mlir::Location loc,
                                                    mlir::PatternRewriter &rewriter) {
        llvm::SmallVector<mlir::Value> values;
        values.reserve(indices.size());
        for (const ReferenceIndex &index : indices) {
            if (index.value) {
                values.push_back(index.value);
            } else if (index.attr) {
                values.push_back(mlir::arith::ConstantOp::create(rewriter, loc,
                                                                 index.attr.getType(), index.attr));
            } else {
                return mlir::failure();
            }
        }
        return packValues(values, type, loc, rewriter);
    }

    mlir::LogicalResult collectIndices(mlir::ValueRange qubits,
                                       llvm::SmallVectorImpl<ReferenceIndex> &indices,
                                       mlir::Value &qreg) {
        for (mlir::Value qubit : qubits) {
            auto index = getReferenceIndex(qubit);
            if (mlir::failed(index)) {
                gate.emitError("register-mode decomposition requires qubits produced by qref.get");
                return mlir::failure();
            }
            if (qreg && qreg != index->qreg) {
                gate.emitError("register-mode decomposition cannot span multiple qregs");
                return mlir::failure();
            }
            qreg = index->qreg;
            indices.push_back(*index);
        }
        return mlir::success();
    }

    mlir::LogicalResult appendPackedIndices(mlir::func::FuncOp rule,
                                            mlir::ArrayRef<ReferenceIndex> indices,
                                            mlir::ArrayRef<mlir::Type> inputTypes,
                                            llvm::SmallVectorImpl<mlir::Value> &operands,
                                            mlir::PatternRewriter &rewriter) {
        if (indices.empty()) {
            return mlir::success();
        }
        if (operands.size() >= inputTypes.size()) {
            rule.emitError("decomposition rule has no input for target-operation wire indices");
            return mlir::failure();
        }
        auto value = packIndices(indices, inputTypes[operands.size()], gate.getLoc(), rewriter);
        if (mlir::failed(value)) {
            rule.emitError("decomposition rule wire-index ABI does not match target operation");
            return mlir::failure();
        }
        operands.push_back(*value);
        return mlir::success();
    }

    mlir::LogicalResult appendRegisterOperands(mlir::func::FuncOp rule,
                                               mlir::ArrayRef<mlir::Type> inputTypes,
                                               llvm::SmallVectorImpl<mlir::Value> &operands,
                                               mlir::Value &qreg, mlir::PatternRewriter &rewriter) {
        if (auto op = mlir::dyn_cast<catalyst::qref::OperatorOp>(gate.getOperation());
            op && op.getQreg()) {
            qreg = op.getQreg();
            for (mlir::Value indices : op.getArrQubitIndices()) {
                operands.push_back(indices);
            }
            if (op.getArrCtrlIndices() && operands.size() < inputTypes.size()) {
                operands.push_back(op.getArrCtrlIndices());
            }
            if (op.getArrCtrlValues() && operands.size() < inputTypes.size()) {
                operands.push_back(op.getArrCtrlValues());
            }
            return mlir::success();
        }

        llvm::SmallVector<ReferenceIndex> wireIndices;
        llvm::SmallVector<ReferenceIndex> controlIndices;
        if (mlir::failed(collectIndices(gate.getNonCtrlQubitOperands(), wireIndices, qreg)) ||
            mlir::failed(collectIndices(gate.getCtrlQubitOperands(), controlIndices, qreg)) ||
            mlir::failed(appendPackedIndices(rule, wireIndices, inputTypes, operands, rewriter))) {
            return mlir::failure();
        }

        bool hasControlWireSlot = !controlIndices.empty() && operands.size() < inputTypes.size() &&
                                  mlir::isa<mlir::RankedTensorType>(inputTypes[operands.size()]) &&
                                  mlir::cast<mlir::RankedTensorType>(inputTypes[operands.size()])
                                      .getElementType()
                                      .isInteger(64);
        if (hasControlWireSlot && mlir::failed(appendPackedIndices(rule, controlIndices, inputTypes,
                                                                   operands, rewriter))) {
            return mlir::failure();
        }

        if (!gate.getCtrlValueOperands().empty() && operands.size() < inputTypes.size()) {
            auto values = packValues(gate.getCtrlValueOperands(), inputTypes[operands.size()],
                                     gate.getLoc(), rewriter);
            if (mlir::failed(values)) {
                rule.emitError(
                    "decomposition rule control-value ABI does not match target operation");
                return mlir::failure();
            }
            operands.push_back(*values);
        }
        return qreg ? mlir::success() : mlir::failure();
    }

    mlir::LogicalResult appendQubitOperands(mlir::func::FuncOp rule,
                                            mlir::ArrayRef<mlir::Type> inputTypes,
                                            llvm::SmallVectorImpl<mlir::Value> &operands,
                                            mlir::PatternRewriter &rewriter) {
        for (mlir::Value qubit : llvm::concat<const mlir::Value>(gate.getNonCtrlQubitOperands(),
                                                                 gate.getCtrlQubitOperands())) {
            if (operands.size() >= inputTypes.size()) {
                rule.emitError("decomposition rule has too few qubit inputs");
                return mlir::failure();
            }
            auto value = packValues(qubit, inputTypes[operands.size()], gate.getLoc(), rewriter);
            if (mlir::failed(value)) {
                rule.emitError("decomposition rule qubit ABI does not match target operation");
                return mlir::failure();
            }
            operands.push_back(*value);
        }

        if (!gate.getCtrlValueOperands().empty() && operands.size() < inputTypes.size()) {
            auto values = packValues(gate.getCtrlValueOperands(), inputTypes[operands.size()],
                                     gate.getLoc(), rewriter);
            if (mlir::failed(values)) {
                rule.emitError(
                    "decomposition rule control-value ABI does not match target operation");
                return mlir::failure();
            }
            operands.push_back(*values);
        }
        return mlir::success();
    }
};

} // namespace catalyst::quantum
