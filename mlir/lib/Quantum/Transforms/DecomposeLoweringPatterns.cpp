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

#define DEBUG_TYPE "decompose-lowering"

#include <variant>

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/ValueRange.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSet.h"

#include "Quantum/IR/QuantumOps.h"
#include "Quantum/Transforms/Patterns.h"

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
struct QubitIndex {
    // use monostate to represent the invalid index
    std::variant<std::monostate, Value, IntegerAttr> index;

    QubitIndex() : index(std::monostate()) {}
    QubitIndex(Value val) : index(val) {}
    QubitIndex(IntegerAttr attr) : index(attr) {}

    bool isValue() const { return std::holds_alternative<Value>(index); }
    bool isAttr() const { return std::holds_alternative<IntegerAttr>(index); }
    operator bool() const { return isValue() || isAttr(); }
    Value getValue() const { return isValue() ? std::get<Value>(index) : nullptr; }
    IntegerAttr getAttr() const { return isAttr() ? std::get<IntegerAttr>(index) : nullptr; }
};

// The goal of this class is to analyze the signature of a custom operation to get the enough
// information to prepare the call operands and results for replacing the op to calling the
// decomposition function.
class OpSignatureAnalyzer {
  public:
    OpSignatureAnalyzer() = delete;
    OpSignatureAnalyzer(CustomOp op, bool enableQregMode)
        : signature(OpSignature{
              .params = op.getParams(),
              .inQubits = op.getInQubits(),
              .inCtrlQubits = op.getInCtrlQubits(),
              .inCtrlValues = op.getInCtrlValues(),
              .outQubits = op.getOutQubits(),
              .outCtrlQubits = op.getOutCtrlQubits(),
          })
    {
        if (!enableQregMode)
            return;

        signature.sourceQreg = getSourceQreg(signature.inQubits.front());
        if (!signature.sourceQreg) {
            op.emitError("Cannot get source qreg");
            isValid = false;
            return;
        }

        // input wire indices
        for (Value qubit : signature.inQubits) {
            const QubitIndex index = getExtractIndex(qubit);
            if (!index) {
                op.emitError("Cannot get index for input qubit");
                isValid = false;
                return;
            }
            signature.inWireIndices.emplace_back(index);
        }

        // input ctrl wire indices
        for (Value ctrlQubit : signature.inCtrlQubits) {
            const QubitIndex index = getExtractIndex(ctrlQubit);
            if (!index) {
                op.emitError("Cannot get index for ctrl qubit");
                isValid = false;
                return;
            }
            signature.inCtrlWireIndices.emplace_back(index);
        }

        // Output qubit indices are the same as input qubit indices
        signature.outQubitIndices = signature.inWireIndices;
        signature.outCtrlQubitIndices = signature.inCtrlWireIndices;
    }

    operator bool() const { return isValid; }

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
            Value updatedQreg = signature.sourceQreg;
            for (auto [i, qubit] : llvm::enumerate(signature.inQubits)) {
                const QubitIndex &index = signature.inWireIndices[i];
                updatedQreg =
                    rewriter.create<quantum::InsertOp>(loc, updatedQreg.getType(), updatedQreg,
                                                       index.getValue(), index.getAttr(), qubit);
            }

            operands[operandIdx++] = updatedQreg;
            if (!signature.params.empty()) {
                operands[operandIdx] =
                    fromTensorOrAsIs(signature.params, funcInputs[operandIdx], rewriter, loc);
                operandIdx++;
            }
            if (!signature.inWireIndices.empty()) {
                operands[operandIdx] = fromTensorOrAsIs(signature.inWireIndices,
                                                        funcInputs[operandIdx], rewriter, loc);
                operandIdx++;
            }
            if (!signature.inCtrlWireIndices.empty()) {
                operands[operandIdx] = fromTensorOrAsIs(signature.inCtrlWireIndices,
                                                        funcInputs[operandIdx], rewriter, loc);
                operandIdx++;
            }
        }
        else {
            if (!signature.params.empty()) {
                operands[operandIdx] =
                    fromTensorOrAsIs(signature.params, funcInputs[operandIdx], rewriter, loc);
                operandIdx++;
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
        for (const QubitIndex &index : signature.outQubitIndices) {
            auto extractOp = rewriter.create<quantum::ExtractOp>(
                callOp.getLoc(), rewriter.getType<quantum::QubitType>(), qreg, index.getValue(),
                index.getAttr());
            newResults.emplace_back(extractOp.getResult());
        }
        for (const QubitIndex &index : signature.outCtrlQubitIndices) {
            auto extractOp = rewriter.create<quantum::ExtractOp>(
                callOp.getLoc(), rewriter.getType<quantum::QubitType>(), qreg, index.getValue(),
                index.getAttr());
            newResults.emplace_back(extractOp.getResult());
        }
        return newResults;
    }

  private:
    bool isValid = true;

    struct OpSignature {
        ValueRange params;
        ValueRange inQubits;
        ValueRange inCtrlQubits;
        ValueRange inCtrlValues;
        ValueRange outQubits;
        ValueRange outCtrlQubits;

        // Qreg mode specific information
        Value sourceQreg = nullptr;
        SmallVector<QubitIndex> inWireIndices;
        SmallVector<QubitIndex> inCtrlWireIndices;
        SmallVector<QubitIndex> outQubitIndices;
        SmallVector<QubitIndex> outCtrlQubitIndices;
    } signature;

    Value fromTensorOrAsIs(ValueRange values, Type type, PatternRewriter &rewriter, Location loc)
    {
        if (isa<RankedTensorType>(type)) {
            return rewriter.create<tensor::FromElementsOp>(loc, type, values);
        }
        return values.front();
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
                auto constantValue = rewriter.create<arith::ConstantOp>(loc, attr.getType(), attr);
                values.emplace_back(constantValue);
            }
        }

        if (isa<RankedTensorType>(type)) {
            return rewriter.create<tensor::FromElementsOp>(loc, type, values);
        }

        assert(values.size() == 1 && "number of values should be 1 for non-tensor type");
        return values.front();
    }

    Value getSourceQreg(Value qubit)
    {
        while (qubit) {
            if (auto extractOp = qubit.getDefiningOp<quantum::ExtractOp>()) {
                return extractOp.getQreg();
            }

            if (auto customOp = dyn_cast_or_null<quantum::CustomOp>(qubit.getDefiningOp())) {
                if (customOp.getQubitOperands().empty()) {
                    break;
                }
                qubit = customOp.getQubitOperands()[0];
            }
        }

        return nullptr;
    }

    QubitIndex getExtractIndex(Value qubit)
    {
        while (qubit) {
            if (auto extractOp = qubit.getDefiningOp<quantum::ExtractOp>()) {
                if (Value idx = extractOp.getIdx()) {
                    return QubitIndex(idx);
                }
                if (IntegerAttr idxAttr = extractOp.getIdxAttrAttr()) {
                    return QubitIndex(idxAttr);
                }
            }

            if (auto customOp = dyn_cast_or_null<quantum::CustomOp>(qubit.getDefiningOp())) {
                auto qubitOperands = customOp.getQubitOperands();
                auto qubitResults = customOp.getQubitResults();
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

            break;
        }

        return QubitIndex();
    }
};

struct DecomposeLoweringRewritePattern : public OpRewritePattern<CustomOp> {
  private:
    const llvm::StringMap<func::FuncOp> &decompositionRegistry;
    const llvm::StringSet<llvm::MallocAllocator> &targetGateSet;

  public:
    DecomposeLoweringRewritePattern(MLIRContext *context,
                                    const llvm::StringMap<func::FuncOp> &registry,
                                    const llvm::StringSet<llvm::MallocAllocator> &gateSet)
        : OpRewritePattern(context), decompositionRegistry(registry), targetGateSet(gateSet)
    {
    }

    LogicalResult matchAndRewrite(CustomOp op, PatternRewriter &rewriter) const override
    {
        StringRef gateName = op.getGateName();

        // Only decompose the op if it is not in the target gate set
        if (targetGateSet.contains(gateName)) {
            return failure();
        }

        // Find the corresponding decomposition function for the op
        auto it = decompositionRegistry.find(gateName);
        if (it == decompositionRegistry.end()) {
            return failure();
        }
        func::FuncOp decompFunc = it->second;

        // Here is the assumption that the decomposition function must have at least one input and
        // one result
        assert(decompFunc.getFunctionType().getNumInputs() > 0 &&
               "Decomposition function must have at least one input");
        assert(decompFunc.getFunctionType().getNumResults() == 1 &&
               "Decomposition function must have exactly one result");

        auto enableQreg = isa<quantum::QuregType>(decompFunc.getFunctionType().getInput(0));
        auto analyzer = OpSignatureAnalyzer(op, enableQreg);
        assert(analyzer && "Analyzer should be valid");

        rewriter.setInsertionPointAfter(op);
        auto callOperands = analyzer.prepareCallOperands(decompFunc, rewriter, op.getLoc());
        auto callOp =
            rewriter.create<func::CallOp>(op.getLoc(), decompFunc.getFunctionType().getResults(),
                                          decompFunc.getSymName(), callOperands);

        // Replace the op with the call op and adjust the insert ops for the qreg mode
        if (callOp.getNumResults() == 1 && isa<quantum::QuregType>(callOp.getResult(0).getType())) {
            auto results = analyzer.prepareCallResultForQreg(callOp, rewriter);
            rewriter.replaceOp(op, results);
        }
        else {
            rewriter.replaceOp(op, callOp->getResults());
        }

        return success();
    }
};

void populateDecomposeLoweringPatterns(RewritePatternSet &patterns,
                                       const llvm::StringMap<func::FuncOp> &decompositionRegistry,
                                       const llvm::StringSet<llvm::MallocAllocator> &targetGateSet)
{
    patterns.add<DecomposeLoweringRewritePattern>(patterns.getContext(), decompositionRegistry,
                                                  targetGateSet);
}

} // namespace quantum
} // namespace catalyst
