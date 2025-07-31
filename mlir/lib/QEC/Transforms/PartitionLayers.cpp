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

#include <cstddef>
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

    Operation *getParentLayer()
    {
        if (ops.empty())
            return nullptr;

        return ops.back()->getParentOp();
    }

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

    bool isSameBlock(QECOpInterface op) { return empty() || op->getParentOp() == getParentLayer(); }

    bool insert(QECOpInterface op)
    {
        // Gate can be inserted if:
        // 1. It acts on the different qubits that gates in the layer act on
        // 2. Or it commutes with all the gates in the layer
        // 3. And it is must be in the same block
        if (!isSameBlock(op))
            return false;

        // TODO: switch order of these two cases may improve the performance
        if (!actOnSameQubits(op))
            return insertToLayer(op);

        if (commuteToLayer(op))
            return insertToLayer(op);

        return false;
    }

  private:
    llvm::DenseMap<Value, Value> outToInValue;

    bool inline insertToLayer(QECOpInterface op)
    {
        ops.emplace_back(op);
        setInOutOperands(op);
        return true;
    }

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
                if (results.contains(operandValue)) {
                    results.remove(operandValue);
                }
                results.insert(resultValue);
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

    // Erase the ops in the layer one by one
    auto maxIter = layer.ops.size() * 2;
    while (!layer.ops.empty() || maxIter > 0) {
        for (auto op : llvm::reverse(layer.ops)) {
            if (op->use_empty()) {
                writer.eraseOp(op);
                layer.ops.erase(std::remove(layer.ops.begin(), layer.ops.end(), op),
                                layer.ops.end());
            }
        }
        maxIter--;
    }

    assert(layer.ops.empty() && "Layer ops should be empty after construction");

    return true;
}

bool shouldSkip(QECOpInterface op, QECLayer &currentLayer)
{
    // Skip if the ancestor(s) is a layer op
    auto parentOp = op->getParentOp();
    while (parentOp != nullptr) {
        if (isa<LayerOp>(parentOp))
            return true;

        parentOp = parentOp->getParentOp();
    }

    return false;
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
            if (shouldSkip(op, currentLayer))
                return WalkResult::skip();

            // Try to insert the op into the current layer
            if (currentLayer.insert(op))
                return WalkResult::advance();

            // Construct the current layer
            constructLayer(currentLayer, writer);

            // Reset the current layer and insert the op
            currentLayer = QECLayer();
            currentLayer.insert(op);

            return WalkResult::advance();
        });

        // Construct the last layer
        constructLayer(currentLayer, writer);
    };
};
} // namespace qec

std::unique_ptr<Pass> createPartitionLayersPass()
{
    return std::make_unique<PartitionLayersPass>();
}

} // namespace catalyst
