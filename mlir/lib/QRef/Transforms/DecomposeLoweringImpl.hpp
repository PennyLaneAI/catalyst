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
#include <cassert>
#include <cstddef>
#include <iterator>
#include <utility>

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/ErrorHandling.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LLVM.h"

#include "Quantum/IR/QuantumInterfaces.h"
#include "Quantum/IR/QuantumOps.h"
#include "Quantum/IR/QuantumTypes.h"
#include "Quantum/Utils/QubitIndex.h"

using namespace mlir;

namespace catalyst {
namespace qref {

// The goal of this class is to analyze the signature of a custom operation to get the enough
// information to prepare the operands and results for replacing the op with the decomposition
// function.
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

        // Qreg mode specific information (assuming QubitIndex is defined)
        llvm::SmallVector<QubitIndex> inWireIndices;
        llvm::SmallVector<QubitIndex> inCtrlWireIndices;
    } signature;

    BaseSignatureAnalyzer(mlir::Operation *op, mlir::ValueRange params, mlir::ValueRange inQubits,
                          mlir::ValueRange inCtrlQubits, mlir::ValueRange inCtrlValues,
                          bool enableQregMode)
        : paramsStorage(params.begin(), params.end()),
          signature(Signature{.params = mlir::ValueRange(paramsStorage),
                              .inQubits = inQubits,
                              .inCtrlQubits = inCtrlQubits,
                              .inCtrlValues = inCtrlValues,
                              .inWireIndices = {},
                              .inCtrlWireIndices = {}}) {
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
                              .inWireIndices = {},
                              .inCtrlWireIndices = {}}) {
        initializeQregMode(op, enableQregMode);
    }

  public:
    virtual ~BaseSignatureAnalyzer() = default;

    // Public Methods (Identical to Original)
    operator bool() const { return isValid; }

    // The register a rule receives is simply the (unique) register all input qubits were taken
    // from.
    mlir::Value getRegister() {
        llvm::SetVector<mlir::Value> regs;
        for (const auto &index : signature.inWireIndices) {
            regs.insert(index.getReg());
        }
        for (const auto &index : signature.inCtrlWireIndices) {
            regs.insert(index.getReg());
        }
        assert(regs.size() == 1 &&
               "register-mode decomposition rule cannot span multiple qregs yet");
        return regs.front();
    }

    // Prepare the operands for the decomposition function
    // There are two cases:
    // 1. The first input is a qreg, which means the decomposition function is a qreg mode function
    // 2. Otherwise, the decomposition function is a qubit mode function
    //
    // Type signatures:
    // 1. qreg mode:
    //    - func(qreg, param*, inWires*, inCtrlWires*?, inCtrlValues*?) -> qreg
    // 2. qubit mode:
    //    - func(param*, inQubits*, inCtrlQubits*?, inCtrlValues*?) -> outQubits*
    llvm::SmallVector<Value> prepareOperands(func::FuncOp rule, PatternRewriter &rewriter,
                                             Location loc) {
        auto funcType = rule.getFunctionType();
        auto funcInputs = funcType.getInputs();

        SmallVector<Type> funcInputsNoQreg;
        for (auto t : funcInputs) {
            if (!isa<qref::QuregType>(t)) {
                funcInputsNoQreg.push_back(t);
            }
        }

        SmallVector<Value> operands(funcInputs.size());

        auto qregIt = llvm::find_if(rule.getFunctionType().getInputs(),
                                    [](mlir::Type t) { return isa<qref::QuregType>(t); });
        int qregIdx = std::distance(rule.getFunctionType().getInputs().begin(), qregIt);
        bool hasQreg = (qregIt != rule.getFunctionType().getInputs().end());

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
        } else {
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

        // Pass the control values only if the rule has a slot for them
        if (!signature.inCtrlValues.empty() &&
            operandIdx < static_cast<int>(funcInputsNoQreg.size())) {
            operands[operandIdx] = fromTensorOrAsIs(signature.inCtrlValues,
                                                    funcInputsNoQreg[operandIdx], rewriter, loc);
            operandIdx++;
        }

        if (hasQreg) {
            Value reg = getRegister();
            if (!reg) {
                return {};
            }
            std::move_backward(operands.begin() + qregIdx, operands.end() - 1, operands.end());
            operands[qregIdx] = reg;
        }

        return operands;
    }

  private:
    Value fromTensorOrAsIs(ValueRange values, Type type, PatternRewriter &rewriter, Location loc) {
        if (isa<RankedTensorType>(type)) {
            return tensor::FromElementsOp::create(rewriter, loc, type, values);
        }
        return values.front();
    }

    static size_t getElementsCount(Type type) {
        if (isa<RankedTensorType>(type)) {
            auto tensorType = cast<RankedTensorType>(type);
            return tensorType.getNumElements() > 0 ? tensorType.getNumElements() : 1;
        }
        return 1;
    }

    // Helper function to find the range of function input types that correspond to params
    static std::pair<size_t, size_t> findParamTypeRange(ArrayRef<Type> funcInputs,
                                                        size_t sigParamCount, size_t startIdx = 0) {
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

    // generate params for the decomposition function based on function type requirements
    SmallVector<Value> generateParams(ValueRange signatureParams, ArrayRef<Type> funcParamTypes,
                                      PatternRewriter &rewriter, Location loc) {
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
                           Location loc) {
        SmallVector<Value> values;
        for (const QubitIndex &index : indices) {
            if (index.isValue()) {
                values.emplace_back(index.getValue());
            } else if (index.isAttr()) {
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

    void initializeQregMode(mlir::Operation *op, bool enableQregMode) {
        if (!enableQregMode || !op) {
            return;
        }

        // input wire indices
        for (mlir::Value qubit : signature.inQubits) {
            const QubitIndex index = getQubitRefIndex(qubit);
            if (!index) {
                op->emitError("Cannot get index for input qubit");
                isValid = false;
                return;
            }
            signature.inWireIndices.emplace_back(index);
        }

        // input ctrl wire indices
        for (mlir::Value ctrlQubit : signature.inCtrlQubits) {
            const QubitIndex index = getQubitRefIndex(ctrlQubit);
            if (!index) {
                op->emitError("Cannot get index for ctrl qubit");
                isValid = false;
                return;
            }
            signature.inCtrlWireIndices.emplace_back(index);
        }

        assert((signature.inWireIndices.size() + signature.inCtrlWireIndices.size()) > 0 &&
               "inWireIndices or inCtrlWireIndices should not be empty");
    }
};

class DecomposableGateSignatureAnalyzer : public BaseSignatureAnalyzer {
  public:
    DecomposableGateSignatureAnalyzer() = delete;

    DecomposableGateSignatureAnalyzer(DecomposableGate op, bool enableQregMode)
        : BaseSignatureAnalyzer(op,
                                isa<ParametrizedGate>(op.getOperation())
                                    ? cast<ParametrizedGate>(op.getOperation()).getAllParams()
                                    : mlir::ValueRange{},
                                op.getNonCtrlQubitOperands(), op.getCtrlQubitOperands(),
                                op.getCtrlValueOperands(), enableQregMode) {}
};

} // namespace qref
} // namespace catalyst
