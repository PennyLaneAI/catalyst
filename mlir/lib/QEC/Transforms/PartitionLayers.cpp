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

#define DEBUG_TYPE "partition-layers"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"

#include "QEC/IR/QECDialect.h"
#include "QEC/Utils/PauliStringWrapper.h"

using namespace mlir;
using namespace catalyst::qec;

namespace catalyst {
namespace qec {

#define GEN_PASS_DEF_PARTITIONLAYERSPASS
#define GEN_PASS_DECL_PARTITIONLAYERSPASS
#include "QEC/Transforms/Passes.h.inc"

// TODO: Move this class outside the pass
// - Add deconstructor
class QECLayer {
    static inline llvm::DenseMap<Value, int> qubitValueToIndex;
    static inline llvm::DenseMap<Operation *, std::vector<int>> opToIndex;
    static inline int MAX_INDEX;

  public:
    std::vector<QECOpInterface> ops;
    llvm::SetVector<Value> operands;
    llvm::SetVector<Value> results;

    QECLayer() {}
    QECLayer(std::vector<QECOpInterface> ops) : ops(ops) {}
    QECLayer(LayerOp layer)
    {
        // TODO: Initialize by LayerOp
    }

    bool empty() { return ops.empty(); }

    std::vector<int> getQubitIndexFrom(QECOpInterface op)
    {
        std::vector<int> qubitIndexes;

        if (opToIndex.contains(op)) {
            return opToIndex.at(op);
        }

        for (auto [inQubit, outQubit] : llvm::zip(op.getInQubits(), op.getOutQubits())) {
            if (qubitValueToIndex.contains(inQubit)) {
                int index = qubitValueToIndex.at(inQubit);
                qubitIndexes.push_back(index);
                qubitValueToIndex[outQubit] = index;
            }
            else {
                // TODO: Implement to handle quantum.extract op
                qubitValueToIndex[outQubit] = MAX_INDEX;
                qubitIndexes.push_back(MAX_INDEX);
                MAX_INDEX++;
            }
        }

        opToIndex[op] = qubitIndexes;

        return qubitIndexes;
    }

    bool actOnSameQubits(QECOpInterface op)
    {
        auto qubitIndex = getQubitIndexFrom(op);
        std::set<int> layerIndexes;

        for (auto layerOp : ops) {
            auto layerOpQubitIndex = getQubitIndexFrom(layerOp);
            std::set<int> indexSet(layerOpQubitIndex.begin(), layerOpQubitIndex.end());
            layerIndexes.insert(indexSet.begin(), indexSet.end());
        }

        for (auto idx : qubitIndex) {
            if (layerIndexes.count(idx) > 0) {
                return true; // Found an overlap
            }
        }

        return false;
    }

    // Commute two ops if they act on the same qubits based on qubit indexes on that layer
    bool commute(QECOpInterface fromOp, QECOpInterface toOp)
    {
        auto lhsQubitIndexes = getQubitIndexFrom(fromOp);
        auto rhsQubitIndexes = getQubitIndexFrom(toOp);

        llvm::SetVector<int> qubits;
        qubits.insert(lhsQubitIndexes.begin(), lhsQubitIndexes.end());
        qubits.insert(rhsQubitIndexes.begin(), rhsQubitIndexes.end());

        PauliWord lhsPauliWord = expandPauliWord(qubits, lhsQubitIndexes, fromOp);
        PauliWord rhsPauliWord = expandPauliWord(qubits, rhsQubitIndexes, toOp);

        auto lhsPSWrapper = PauliStringWrapper::from_pauli_word(lhsPauliWord);
        auto rhsPSWrapper = PauliStringWrapper::from_pauli_word(rhsPauliWord);

        return lhsPSWrapper.commutes(rhsPSWrapper);
    }

    // Commute an op to all the ops in the layer
    bool commuteToLayer(QECOpInterface op)
    {
        for (auto layerOp : ops) {
            if (!commute(op, layerOp)) {
                return false;
            }
        }
        return true;
    }

    bool insert(QECOpInterface op)
    {
        // Gate can be inserted if:
        // 1. It acts on the different qubits that gates in the layer act on
        // 2. Or it commutes with all the gates in the layer
        if (!actOnSameQubits(op) || commuteToLayer(op)) {
            ops.emplace_back(op);
            setInOutOperands(op);
            return true;
        }

        return false;
    }

  private:
    llvm::DenseMap<Value, Value> inToOutValue;
    llvm::DenseMap<Value, Value> outToInValue;

    void setInOutOperands(QECOpInterface op)
    {
        for (auto [operandValOpt, resultValOpt] :
             llvm::zip_longest(op->getOperands(), op->getResults())) {
            Value operandValue = operandValOpt.value_or(nullptr);
            Value resultValue = resultValOpt.value_or(nullptr);

            // TODO: Optimize these two cases
            // Set inOperand
            if (operandValue != nullptr) {
                if (outToInValue.contains(operandValue)) {
                    auto originValue = outToInValue[operandValue];
                    outToInValue[resultValue] = originValue;
                    outToInValue.erase(operandValue);
                }
                else {
                    outToInValue[resultValue] = operandValue;
                    operands.insert(operandValue);
                }
            }

            // Set outResult
            if (resultValue != nullptr) {
                if (inToOutValue.contains(resultValue)) {
                    auto lastOutValue = inToOutValue[resultValue];
                    inToOutValue[operandValue] = lastOutValue;
                    inToOutValue.erase(resultValue);
                }
                else {
                    inToOutValue[operandValue] = resultValue;
                    results.insert(resultValue);
                }
            }
        }
    }
};

bool constructLayer(QECLayer layer, IRRewriter &writer)
{
    if (layer.empty())
        return false;

    auto loc = layer.ops.front().getLoc();
    auto inOperands = ValueRange(layer.operands.getArrayRef());
    auto outResults = ValueRange(layer.results.getArrayRef());

    writer.setInsertionPointAfter(layer.ops.back());

    auto layerOp = writer.create<qec::LayerOp>(
        loc, inOperands, outResults,
        [&](OpBuilder &builder, Location loc, ValueRange operands, ValueRange results) {
            // Map the input operands to the layerOp's arguments
            IRMapping mapper;
            mapper.map(inOperands, operands);
            llvm::SmallVector<Value> newResults(results.begin(), results.end());

            for (auto op : layer.ops) {
                auto newOp = op->clone(mapper);
                builder.insert(newOp);

                // Config the output results and map to the yield op
                for (auto [i, result] : llvm::enumerate(outResults)) {
                    for (auto [j, newResult] : llvm::enumerate(op->getResults())) {
                        if (result == newResult) {
                            newResults[i] = newOp->getResult(j);
                        }
                    }
                }
            }
            builder.create<qec::YieldOp>(loc, newResults);
        });

    // Replace the use of the original operations outside the layer with the new results
    for (auto [prevResult, newResult] : llvm::zip(outResults, layerOp->getResults())) {
        prevResult.replaceAllUsesWith(newResult);
    }

    // Erase original operations outside the layer
    for (auto op : layer.ops) {
        writer.eraseOp(op);
    }

    return true;
}

struct PartitionLayersPass : public impl::PartitionLayersPassBase<PartitionLayersPass> {
    using PartitionLayersPassBase::PartitionLayersPassBase;

    void runOnOperation() final
    {
        MLIRContext *context = &getContext();
        mlir::IRRewriter writer(context);

        QECLayer currentLayer;

        // Group the ops into layers in memory
        getOperation()->walk([&](QECOpInterface op) {
            // Skip if the parent is a layer op
            if (isa<LayerOp>(op->getParentOp())) {
                return WalkResult::skip();
            }

            // TODO: Handle the case where the its parent is control flow or any container op

            if (currentLayer.insert(op)) {
                return WalkResult::advance();
            }

            if (!currentLayer.empty()) {
                constructLayer(currentLayer, writer);
            }

            currentLayer = QECLayer();
            currentLayer.insert(op);

            return WalkResult::advance();
        });

        // Construct the last layer
        if (!currentLayer.empty()) {
            constructLayer(currentLayer, writer);
        }
    };
};
} // namespace qec

std::unique_ptr<Pass> createPartitionLayersPass()
{
    return std::make_unique<PartitionLayersPass>();
}

} // namespace catalyst
