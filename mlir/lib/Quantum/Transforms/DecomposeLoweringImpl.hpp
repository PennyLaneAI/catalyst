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

#include <algorithm> // std::move_backward

#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/ErrorHandling.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/ValueRange.h"

#include "Quantum/IR/QuantumOps.h"
#include "Quantum/IR/QuantumTypes.h"
#include "Quantum/Utils/QubitIndex.h"

using namespace mlir;
using namespace catalyst::quantum;

namespace catalyst {
namespace quantum {

// The goal of this class is to analyze the signature of a custom operation to get the enough
// information to prepare the call operands and results for replacing the op to calling the
// decomposition function.
class BaseSignatureAnalyzer {
  protected:
    bool isValid = true;

    llvm::SmallVector<mlir::Value, 4> paramsStorage;

    // Unified Signature Structure: All parameters, regardless of source (params or theta),
    // are stored in a ValueRange for generalized processing.
    struct Signature {
        mlir::ValueRange params;
        mlir::ValueRange inQubits;
        mlir::ValueRange inCtrlQubits;
        mlir::ValueRange inCtrlValues;
        mlir::ValueRange outQubits;
        mlir::ValueRange outCtrlQubits;

        // Qreg mode specific information (assuming QubitIndex is defined)
        llvm::SmallVector<QubitIndex> inWireIndices;
        llvm::SmallVector<QubitIndex> inCtrlWireIndices;
        llvm::SmallVector<QubitIndex> outQubitIndices;
        llvm::SmallVector<QubitIndex> outCtrlQubitIndices;
    } signature;

    BaseSignatureAnalyzer(mlir::Operation *op, mlir::ValueRange params, mlir::ValueRange inQubits,
                          mlir::ValueRange inCtrlQubits, mlir::ValueRange inCtrlValues,
                          mlir::ValueRange outQubits, mlir::ValueRange outCtrlQubits,
                          bool enableQregMode)
        : paramsStorage(params.begin(), params.end()),
          signature(Signature{.params = mlir::ValueRange(paramsStorage),
                              .inQubits = inQubits,
                              .inCtrlQubits = inCtrlQubits,
                              .inCtrlValues = inCtrlValues,
                              .outQubits = outQubits,
                              .outCtrlQubits = outCtrlQubits,
                              .inWireIndices = {},
                              .inCtrlWireIndices = {},
                              .outQubitIndices = {},
                              .outCtrlQubitIndices = {}})
    {
        initializeQregMode(op, enableQregMode);
    }

    BaseSignatureAnalyzer(mlir::Operation *op, Value param, mlir::ValueRange inQubits,
                          mlir::ValueRange inCtrlQubits, mlir::ValueRange inCtrlValues,
                          mlir::ValueRange outQubits, mlir::ValueRange outCtrlQubits,
                          bool enableQregMode)
        : paramsStorage(mlir::ValueRange(param).begin(), mlir::ValueRange(param).end()),
          signature(Signature{.params = mlir::ValueRange(paramsStorage),
                              .inQubits = inQubits,
                              .inCtrlQubits = inCtrlQubits,
                              .inCtrlValues = inCtrlValues,
                              .outQubits = outQubits,
                              .outCtrlQubits = outCtrlQubits,
                              .inWireIndices = {},
                              .inCtrlWireIndices = {},
                              .outQubitIndices = {},
                              .outCtrlQubitIndices = {}})
    {
        initializeQregMode(op, enableQregMode);
    }

  public:
    virtual ~BaseSignatureAnalyzer() = default;

    // Public Methods (Identical to Original)
    operator bool() const { return isValid; }

    // Returns true if lateQreg is a later qreg in an insert chain from earlyQreg
    // Raise an error if both qregs are rooted at different allocations.
    bool isDescendantQreg(Value lateQreg, Value earlyQreg, quantum::AllocOp earlyQregRootAlloc)
    {
        while (lateQreg != earlyQreg) {
            if (auto insertOp = lateQreg.getDefiningOp<quantum::InsertOp>()) {
                lateQreg = insertOp.getInQreg();
            }
            else if (auto adjointOp = lateQreg.getDefiningOp<quantum::AdjointOp>()) {
                OpResult lateQregAsAdjointResult = cast<OpResult>(lateQreg);
                lateQreg = adjointOp.getOperand(lateQregAsAdjointResult.getResultNumber());
            }
            else if (auto allocOp = lateQreg.getDefiningOp<quantum::AllocOp>()) {
                assert(allocOp == earlyQregRootAlloc &&
                       "The qreg of the input wires should be the same");
                return false;
            }
            else {
                llvm_unreachable("Encountered unknown operation. A quantum register value can only "
                                 "be produced by alloc, insert and adjoint ops.");
            }
        }
        return true;
    }

    Value getUpdatedQreg(PatternRewriter &rewriter, Location loc)
    {
        // FIXME: This will cause an issue when the decomposition function has cross-qreg
        // inputs and outputs. Now, we just assume has only one qreg input, the global one exists.
        // raise an error if the qreg is not the same

        // Collect all qregs from input wires
        llvm::SetVector<Value> qregs;
        for (const auto &index : signature.inWireIndices) {
            qregs.insert(index.getReg());
        }
        for (const auto &index : signature.inCtrlWireIndices) {
            qregs.insert(index.getReg());
        }

        // Quick return if all qregs are the same
        if (qregs.size() == 1) {
            return qregs[0];
        }

        // Find the latest qreg in the insert chain
        Value latestQreg = qregs[0];

        quantum::AllocOp rootAllocOp;
        Value allocFinder = latestQreg;
        while (allocFinder) {
            if (auto insertOp = allocFinder.getDefiningOp<quantum::InsertOp>()) {
                allocFinder = insertOp.getInQreg();
                continue;
            }
            else if (auto adjointOp = allocFinder.getDefiningOp<quantum::AdjointOp>()) {
                OpResult lateQregAsAdjointResult = cast<OpResult>(allocFinder);
                allocFinder = adjointOp.getOperand(lateQregAsAdjointResult.getResultNumber());
                continue;
            }
            else if (auto allocOp = allocFinder.getDefiningOp<quantum::AllocOp>()) {
                rootAllocOp = allocOp;
                break;
            }
        }

        for (Value qreg : qregs) {
            if (isDescendantQreg(qreg, latestQreg, rootAllocOp)) {
                latestQreg = qreg;
            }
        }
        return latestQreg;
    }

    // Prepare the operands for calling the decomposition function
    // There are two cases:
    // 1. The first input is a qreg, which means the decomposition function is a qreg mode function
    // 2. Otherwise, the decomposition function is a qubit mode function
    //
    // Type signatures:
    // 1. qreg mode:
    //    - func(qreg, param*, inWires*, inCtrlWires*?, inCtrlValues*?) -> qreg
    // 2. qubit mode:
    //    - func(param*, inQubits*, inCtrlQubits*?, inCtrlValues*?) -> outQubits*
    llvm::SmallVector<Value> prepareCallOperands(func::FuncOp decompFunc, PatternRewriter &rewriter,
                                                 Location loc)
    {
        auto funcType = decompFunc.getFunctionType();
        auto funcInputs = funcType.getInputs();

        SmallVector<Type> funcInputsNoQreg;
        for (auto t : funcInputs) {
            if (!isa<quantum::QuregType>(t)) {
                funcInputsNoQreg.push_back(t);
            }
        }

        SmallVector<Value> operands(funcInputs.size());

        auto qregIt = llvm::find_if(decompFunc.getFunctionType().getInputs(),
                                    [](mlir::Type t) { return isa<quantum::QuregType>(t); });
        int qregIdx = std::distance(decompFunc.getFunctionType().getInputs().begin(), qregIt);
        bool hasQreg = (qregIt != decompFunc.getFunctionType().getInputs().end());

        int operandIdx = 0;
        if (!signature.params.empty()) {
            auto [startIdx, endIdx] =
                findParamTypeRange(funcInputsNoQreg, signature.params.size(), operandIdx);
            ArrayRef<Type> paramsTypes =
                ArrayRef<Type>(funcInputsNoQreg).slice(startIdx, endIdx - startIdx);
            auto updatedParams = generateParams(signature.params, paramsTypes, rewriter, loc);
            for (Value param : updatedParams) {
                operands[operandIdx++] = param;
            }
        }

        if (hasQreg) {
            for (const auto &indices : {signature.inWireIndices, signature.inCtrlWireIndices}) {
                if (!indices.empty()) {
                    operands[operandIdx] =
                        fromTensorOrAsIs(indices, funcInputsNoQreg[operandIdx], rewriter, loc);
                    operandIdx++;
                }
            }
        }
        else {
            for (auto inQubit : signature.inQubits) {
                operands[operandIdx] =
                    fromTensorOrAsIs(inQubit, funcInputsNoQreg[operandIdx], rewriter, loc);
                operandIdx++;
            }

            for (auto inCtrlQubit : signature.inCtrlQubits) {
                operands[operandIdx] =
                    fromTensorOrAsIs(inCtrlQubit, funcInputsNoQreg[operandIdx], rewriter, loc);
                operandIdx++;
            }
        }

        if (!signature.inCtrlValues.empty()) {
            operands[operandIdx] = fromTensorOrAsIs(signature.inCtrlValues,
                                                    funcInputsNoQreg[operandIdx], rewriter, loc);
            operandIdx++;
        }

        if (hasQreg) {
            Value updatedQreg = getUpdatedQreg(rewriter, loc);

            for (auto [i, qubit] : llvm::enumerate(signature.inQubits)) {
                const QubitIndex &index = signature.inWireIndices[i];
                updatedQreg =
                    quantum::InsertOp::create(rewriter, loc, updatedQreg.getType(), updatedQreg,
                                              index.getValue(), index.getAttr(), qubit);
            }
            std::move_backward(operands.begin() + qregIdx, operands.end() - 1, operands.end());
            operands[qregIdx] = updatedQreg;
        }

        return operands;
    }

    // Prepare the results for the call operation
    SmallVector<Value> prepareCallResultForQreg(func::CallOp callOp, PatternRewriter &rewriter)
    {
        assert(callOp.getNumResults() == 1 && "only one qreg result for qreg mode is allowed");

        auto qreg = callOp.getResult(0);
        assert(isa<quantum::QuregType>(qreg.getType()) && "only allow to have qreg result");

        SmallVector<Value> newResults;
        rewriter.setInsertionPointAfter(callOp);

        for (const auto &indices : {signature.outQubitIndices, signature.outCtrlQubitIndices}) {
            for (const auto &index : indices) {
                auto extractOp = quantum::ExtractOp::create(
                    rewriter, callOp.getLoc(), rewriter.getType<quantum::QubitType>(), qreg,
                    index.getValue(), index.getAttr());
                newResults.emplace_back(extractOp.getResult());
            }
        }

        return newResults;
    }

  private:
    Value fromTensorOrAsIs(ValueRange values, Type type, PatternRewriter &rewriter, Location loc)
    {
        if (isa<RankedTensorType>(type)) {
            return tensor::FromElementsOp::create(rewriter, loc, type, values);
        }
        return values.front();
    }

    static size_t getElementsCount(Type type)
    {
        if (isa<RankedTensorType>(type)) {
            auto tensorType = cast<RankedTensorType>(type);
            return tensorType.getNumElements() > 0 ? tensorType.getNumElements() : 1;
        }
        return 1;
    }

    // Helper function to find the range of function input types that correspond to params
    static std::pair<size_t, size_t> findParamTypeRange(ArrayRef<Type> funcInputs,
                                                        size_t sigParamCount, size_t startIdx = 0)
    {
        size_t paramTypeCount = 0;
        size_t paramTypeEnd = startIdx;

        while (paramTypeCount < sigParamCount) {
            assert(paramTypeEnd < funcInputs.size() &&
                   "param type end should be less than function input size");
            paramTypeCount += getElementsCount(funcInputs[paramTypeEnd]);
            paramTypeEnd++;
        }

        assert(paramTypeCount == sigParamCount &&
               "param type count should be equal to signature param count");

        return {startIdx, paramTypeEnd};
    }

    // generate params for calling the decomposition function based on function type requirements
    SmallVector<Value> generateParams(ValueRange signatureParams, ArrayRef<Type> funcParamTypes,
                                      PatternRewriter &rewriter, Location loc)
    {
        SmallVector<Value> operands;
        size_t sigParamIdx = 0;

        for (Type funcParamType : funcParamTypes) {
            const size_t numElements = getElementsCount(funcParamType);

            // collect numElements of signature params
            SmallVector<Value> tensorElements;
            for (size_t i = 0; i < numElements && sigParamIdx < signatureParams.size(); i++) {
                tensorElements.push_back(signatureParams[sigParamIdx++]);
            }
            operands.push_back(fromTensorOrAsIs(tensorElements, funcParamType, rewriter, loc));
        }

        return operands;
    }

    Value fromTensorOrAsIs(ArrayRef<QubitIndex> indices, Type type, PatternRewriter &rewriter,
                           Location loc)
    {
        SmallVector<Value> values;
        for (const QubitIndex &index : indices) {
            if (index.isValue()) {
                values.emplace_back(index.getValue());
            }
            else if (index.isAttr()) {
                auto attr = index.getAttr();
                auto constantValue = arith::ConstantOp::create(rewriter, loc, attr.getType(), attr);
                values.emplace_back(constantValue);
            }
        }

        if (isa<RankedTensorType>(type)) {
            return tensor::FromElementsOp::create(rewriter, loc, type, values);
        }

        assert(values.size() == 1 && "number of values should be 1 for non-tensor type");
        return values.front();
    }

    void initializeQregMode(mlir::Operation *op, bool enableQregMode)
    {
        if (!enableQregMode || !op) {
            return;
        }

        // input wire indices
        for (mlir::Value qubit : signature.inQubits) {
            const QubitIndex index = getExtractIndex(qubit);
            if (!index) {
                op->emitError("Cannot get index for input qubit");
                isValid = false;
                return;
            }
            signature.inWireIndices.emplace_back(index);
        }

        // input ctrl wire indices
        for (mlir::Value ctrlQubit : signature.inCtrlQubits) {
            const QubitIndex index = getExtractIndex(ctrlQubit);
            if (!index) {
                op->emitError("Cannot get index for ctrl qubit");
                isValid = false;
                return;
            }
            signature.inCtrlWireIndices.emplace_back(index);
        }

        assert((signature.inWireIndices.size() + signature.inCtrlWireIndices.size()) > 0 &&
               "inWireIndices or inCtrlWireIndices should not be empty");

        // Output qubit indices are the same as input qubit indices
        signature.outQubitIndices = signature.inWireIndices;
        signature.outCtrlQubitIndices = signature.inCtrlWireIndices;
    }
};

class CustomOpSignatureAnalyzer : public BaseSignatureAnalyzer {
  public:
    CustomOpSignatureAnalyzer() = delete;

    CustomOpSignatureAnalyzer(CustomOp op, bool enableQregMode)
        : BaseSignatureAnalyzer(op, op.getParams(), op.getNonCtrlQubitOperands(),
                                op.getCtrlQubitOperands(), op.getCtrlValueOperands(),
                                op.getNonCtrlQubitResults(), op.getCtrlQubitResults(),
                                enableQregMode)
    {
    }
};

class PauliRotOpSignatureAnalyzer : public BaseSignatureAnalyzer {
  public:
    PauliRotOpSignatureAnalyzer() = delete;

    PauliRotOpSignatureAnalyzer(PauliRotOp op, bool enableQregMode)
        : BaseSignatureAnalyzer(op, op.getAngle(), op.getNonCtrlQubitOperands(),
                                op.getCtrlQubitOperands(), op.getCtrlValueOperands(),
                                op.getNonCtrlQubitResults(), op.getCtrlQubitResults(),
                                enableQregMode)
    {
    }
};
class MultiRZOpSignatureAnalyzer : public BaseSignatureAnalyzer {
  public:
    MultiRZOpSignatureAnalyzer() = delete;

    MultiRZOpSignatureAnalyzer(MultiRZOp op, bool enableQregMode)
        : BaseSignatureAnalyzer(op, op.getTheta(), op.getNonCtrlQubitOperands(),
                                op.getCtrlQubitOperands(), op.getCtrlValueOperands(),
                                op.getNonCtrlQubitResults(), op.getCtrlQubitResults(),
                                enableQregMode)
    {
    }
};

} // namespace quantum
} // namespace catalyst
