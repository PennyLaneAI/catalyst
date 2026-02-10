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

#include <variant>

#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSet.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/ValueRange.h"

#include "Quantum/IR/QuantumOps.h"

using namespace mlir;
using namespace catalyst::quantum;

namespace catalyst {
namespace quantum {

/// A struct to represent qubit indices in quantum operations.
///
/// This struct provides a way to handle qubit indices that can be either:
/// - A runtime Value (for dynamic indices computed at runtime)
/// - An IntegerAttr (for compile-time constant indices)
/// - Invalid/uninitialized (represented by std::monostate)
/// And a qreg value to represent the qreg that the index belongs to
///
/// The struct uses std::variant to ensure only one type is active at a time,
/// preventing invalid states.
///
/// Example usage:
///   QubitIndex dynamicIdx(operandValue);     // Runtime qubit index
///   QubitIndex staticIdx(IntegerAttr::get(...)); // Compile-time constant
///   QubitIndex invalidIdx;                   // Uninitialized state
///
///   if (dynamicIdx) {                        // Check if valid
///     if (dynamicIdx.isValue()) {            // Check if runtime value
///       Value idx = dynamicIdx.getValue();   // Get the Value
///     }
///   }
class QubitIndex {
  private:
    // use monostate to represent the invalid index
    std::variant<std::monostate, Value, IntegerAttr> index;
    Value qreg;

  public:
    QubitIndex() : index(std::monostate()), qreg(nullptr) {}
    QubitIndex(Value val, Value qreg) : index(val), qreg(qreg) {}
    QubitIndex(IntegerAttr attr, Value qreg) : index(attr), qreg(qreg) {}

    bool isValue() const { return std::holds_alternative<Value>(index); }
    bool isAttr() const { return std::holds_alternative<IntegerAttr>(index); }
    operator bool() const { return isValue() || isAttr(); }
    Value getReg() const { return qreg; }
    Value getValue() const { return isValue() ? std::get<Value>(index) : nullptr; }
    IntegerAttr getAttr() const { return isAttr() ? std::get<IntegerAttr>(index) : nullptr; }
};

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

    Value getUpdatedQreg(PatternRewriter &rewriter, Location loc)
    {
        // FIXME: This will cause an issue when the decomposition function has cross-qreg
        // inputs and outputs. Now, we just assume has only one qreg input, the global one exists.
        // raise an error if the qreg is not the same
        Value qreg = signature.inWireIndices[0].getReg();

        bool sameQreg = true;
        for (const auto &index : signature.inWireIndices) {
            sameQreg &= index.getReg() == qreg;
        }
        for (const auto &index : signature.inCtrlWireIndices) {
            sameQreg &= index.getReg() == qreg;
        }

        assert(sameQreg && "The qreg of the input wires should be the same");
        return qreg;
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

        SmallVector<Value> operands(funcInputs.size());

        int operandIdx = 0;
        if (isa<quantum::QuregType>(funcInputs[0])) {
            Value updatedQreg = getUpdatedQreg(rewriter, loc);
            for (auto [i, qubit] : llvm::enumerate(signature.inQubits)) {
                const QubitIndex &index = signature.inWireIndices[i];
                updatedQreg =
                    quantum::InsertOp::create(rewriter, loc, updatedQreg.getType(), updatedQreg,
                                              index.getValue(), index.getAttr(), qubit);
            }
            operands[operandIdx++] = updatedQreg;

            if (!signature.params.empty()) {
                auto [startIdx, endIdx] =
                    findParamTypeRange(funcInputs, signature.params.size(), operandIdx);
                ArrayRef<Type> paramsTypes = funcInputs.slice(startIdx, endIdx - startIdx);
                auto updatedParams = generateParams(signature.params, paramsTypes, rewriter, loc);
                for (Value param : updatedParams) {
                    operands[operandIdx++] = param;
                }
            }

            for (const auto &indices : {signature.inWireIndices, signature.inCtrlWireIndices}) {
                if (!indices.empty()) {
                    operands[operandIdx] =
                        fromTensorOrAsIs(indices, funcInputs[operandIdx], rewriter, loc);
                    operandIdx++;
                }
            }
        }
        else {
            if (!signature.params.empty()) {
                auto [startIdx, endIdx] =
                    findParamTypeRange(funcInputs, signature.params.size(), operandIdx);
                ArrayRef<Type> paramsTypes = funcInputs.slice(startIdx, endIdx - startIdx);
                auto updatedParams = generateParams(signature.params, paramsTypes, rewriter, loc);
                for (Value param : updatedParams) {
                    operands[operandIdx++] = param;
                }
            }

            for (auto inQubit : signature.inQubits) {
                operands[operandIdx] =
                    fromTensorOrAsIs(inQubit, funcInputs[operandIdx], rewriter, loc);
                operandIdx++;
            }

            for (auto inCtrlQubit : signature.inCtrlQubits) {
                operands[operandIdx] =
                    fromTensorOrAsIs(inCtrlQubit, funcInputs[operandIdx], rewriter, loc);
                operandIdx++;
            }
        }

        if (!signature.inCtrlValues.empty()) {
            operands[operandIdx] =
                fromTensorOrAsIs(signature.inCtrlValues, funcInputs[operandIdx], rewriter, loc);
            operandIdx++;
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
        if (!enableQregMode || !op)
            return;

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

    QubitIndex getExtractIndex(Value qubit)
    {
        while (qubit) {
            if (auto extractOp = qubit.getDefiningOp<quantum::ExtractOp>()) {
                if (Value idx = extractOp.getIdx()) {
                    return QubitIndex(idx, extractOp.getQreg());
                }
                if (IntegerAttr idxAttr = extractOp.getIdxAttrAttr()) {
                    return QubitIndex(idxAttr, extractOp.getQreg());
                }
            }

            if (auto gate = dyn_cast_or_null<quantum::QuantumGate>(qubit.getDefiningOp())) {
                auto qubitOperands = gate.getQubitOperands();
                auto qubitResults = gate.getQubitResults();
                auto it =
                    llvm::find_if(qubitResults, [&](Value result) { return result == qubit; });

                if (it != qubitResults.end()) {
                    size_t resultIndex = std::distance(qubitResults.begin(), it);
                    if (resultIndex < qubitOperands.size()) {
                        qubit = qubitOperands[resultIndex];
                        continue;
                    }
                }
            }
            else if (auto measureOp = dyn_cast_or_null<quantum::MeasureOp>(qubit.getDefiningOp())) {
                qubit = measureOp.getInQubit();
                continue;
            }

            break;
        }

        return QubitIndex();
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
