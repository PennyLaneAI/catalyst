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

#include "QEC/Utils/QECLayer.h"

using namespace mlir;
using namespace catalyst::qec;

namespace catalyst {
namespace qec {

#define GEN_PASS_DEF_PARTITIONLAYERSPASS
#define GEN_PASS_DECL_PARTITIONLAYERSPASS
#include "QEC/Transforms/Passes.h.inc"

void eraseUnusedOps(QECLayer &layer, IRRewriter &writer)
{
    // Erase standalone QEC ops in the layer one by one
    // Max iteration is set to avoid infinite loop
    int maxIter = layer.ops.size() * 2;
    while (!layer.ops.empty() && maxIter > 0) {
        for (auto op : llvm::reverse(layer.ops)) {
            // Only erase if not part of register operation and not used
            if (op->use_empty()) {
                writer.eraseOp(op);
                layer.ops.erase(std::remove(layer.ops.begin(), layer.ops.end(), op),
                                layer.ops.end());
            }
        }
        maxIter--;
    }

    assert(layer.ops.empty() && "Layer ops should be empty after construction");
}

void constructLayer(QECLayer layer, IRRewriter &writer)
{
    if (layer.empty())
        return;

    auto loc = layer.ops.front().getLoc();
    auto inOperands = ValueRange(layer.operands.getArrayRef());
    llvm::SmallVector<Value> orderedResults = layer.getResultsOrderedByTypeThenOperand();
    auto outResults = ValueRange(orderedResults);

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
    eraseUnusedOps(layer, writer);
}

bool isParentLayerOp(QECOpInterface op)
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

        // Create thread-local context for this pass instance
        QECLayerContext layerContext;
        QECLayer currentLayer(&layerContext);

        // Group the ops into layers in memory
        getOperation()->walk([&](QECOpInterface op) {
            // Skip if the parent is a layer op
            if (isParentLayerOp(op))
                return WalkResult::skip();

            // Try to insert the op into the current layer
            if (currentLayer.insert(op))
                return WalkResult::advance();

            // Construct the current layer
            constructLayer(currentLayer, writer);

            // Reset the current layer and insert the op
            currentLayer = QECLayer(&layerContext);
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
