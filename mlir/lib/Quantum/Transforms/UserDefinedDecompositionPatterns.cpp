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

#define DEBUG_TYPE "user-defined-decomposition"

#include <variant>

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/ValueRange.h"
#include "llvm/ADT/StringSet.h"

#include "Quantum/IR/QuantumOps.h"
#include "Quantum/Transforms/Patterns.h"

using namespace mlir;
using namespace catalyst::quantum;

namespace catalyst {
namespace quantum {

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
          })
    {
        if (!enableQregMode)
            return;

        signature.sourceQreg = getSourceQreg(signature.inQubits.front());
        for (Value qubit : signature.inQubits) {
            const QubitIndex index = getExtractIndex(qubit);
            assert(index && "Cannot get index for input qubit");
            signature.inWireIndices.emplace_back(index);
        }

        for (Value ctrlQubit : signature.inCtrlQubits) {
            const QubitIndex index = getExtractIndex(ctrlQubit);
            assert(index && "Cannot get index for ctrl qubit");
            signature.inCtrlWireIndices.emplace_back(index);
        }

        for (Value outQubit : signature.outQubits) {
            const QubitIndex insertIndex = getInsertIndex(outQubit);
            assert(insertIndex && "Cannot find insert index for result qubit");
            signature.outQubitIndices.emplace_back(insertIndex);
        }
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
            operands[operandIdx++] = signature.sourceQreg;
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
        return newResults;
    }

    // Update the insert ops of the output qubits to use the new qreg
    void replaceInsertOpsQreg(Value newQreg, PatternRewriter &rewriter)
    {
        SmallVector<quantum::InsertOp> insertOpsToUpdate;

        for (Value outQubit : signature.outQubits) {
            for (Operation *user : outQubit.getUsers()) {
                if (auto insertOp = dyn_cast<quantum::InsertOp>(user)) {
                    rewriter.replaceOp(insertOp, newQreg);
                }
            }
        }
    }

  private:
    struct OpSignature {
        ValueRange params;
        ValueRange inQubits;
        ValueRange inCtrlQubits;
        ValueRange inCtrlValues;
        ValueRange outQubits;

        // Qreg mode specific information
        Value sourceQreg = nullptr;
        SmallVector<QubitIndex> inWireIndices;
        SmallVector<QubitIndex> inCtrlWireIndices;
        SmallVector<QubitIndex> outQubitIndices;
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
        if (auto extractOp = qubit.getDefiningOp<quantum::ExtractOp>()) {
            return extractOp.getQreg();
        }
        return nullptr;
    }

    QubitIndex getExtractIndex(Value qubit)
    {
        if (auto extractOp = qubit.getDefiningOp<quantum::ExtractOp>()) {
            if (Value idx = extractOp.getIdx()) {
                return QubitIndex(idx);
            }
            if (IntegerAttr idxAttr = extractOp.getIdxAttrAttr()) {
                return QubitIndex(idxAttr);
            }
        }
        return QubitIndex();
    }

    QubitIndex getInsertIndex(Value qubit)
    {
        for (Operation *user : qubit.getUsers()) {
            if (auto insertOp = dyn_cast<quantum::InsertOp>(user)) {
                if (Value idx = insertOp.getIdx()) {
                    return QubitIndex(idx);
                }
                if (IntegerAttr idxAttr = insertOp.getIdxAttrAttr()) {
                    return QubitIndex(idxAttr);
                }
            }
        }
        return QubitIndex();
    }
};

struct UserDefinedDecompositionRewritePattern : public OpRewritePattern<CustomOp> {
  private:
    const llvm::StringMap<func::FuncOp> &decompositionRegistry;
    const llvm::StringSet<llvm::MallocAllocator> &targetGateSet;

  public:
    UserDefinedDecompositionRewritePattern(MLIRContext *context,
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
        auto callOperands = analyzer.prepareCallOperands(decompFunc, rewriter, op.getLoc());
        auto callOp =
            rewriter.create<func::CallOp>(op.getLoc(), decompFunc.getFunctionType().getResults(),
                                          decompFunc.getSymName(), callOperands);

        // Replace the op with the call op and adjust the insert ops for the qreg mode
        if (callOp.getNumResults() == 1 && isa<quantum::QuregType>(callOp.getResult(0).getType())) {
            Value newQreg = callOp.getResult(0);
            // Since the call op is a qreg mode function, we need to extract the qubits from the
            // qreg to fit the original op's output qubits
            auto results = analyzer.prepareCallResultForQreg(callOp, rewriter);
            analyzer.replaceInsertOpsQreg(newQreg, rewriter);
            rewriter.eraseOp(op);
        }
        else {
            rewriter.replaceOp(op, callOp->getResults());
        }

        return success();
    }
};

void populateUserDefinedDecompositionPatterns(
    RewritePatternSet &patterns, const llvm::StringMap<func::FuncOp> &decompositionRegistry,
    const llvm::StringSet<llvm::MallocAllocator> &targetGateSet)
{
    patterns.add<UserDefinedDecompositionRewritePattern>(patterns.getContext(),
                                                         decompositionRegistry, targetGateSet);
}

} // namespace quantum
} // namespace catalyst
