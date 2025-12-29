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

#define GEN_PASS_DECL_PARTITIONLAYERSPASS
#define GEN_PASS_DEF_PARTITIONLAYERSPASS
#include "QEC/Transforms/Passes.h.inc"

void eraseUnusedOps(QECLayer &layer, IRRewriter &writer)
{
    // Erase original QEC ops from the block after wiring the layer results
    // Bound iterations to avoid pathological loops
    int maxIter = static_cast<int>(layer.getOps().size()) * 2;
    while (!layer.empty() && maxIter > 0) {
        for (auto op : llvm::reverse(layer.getOps())) {
            // Only erase if now unused
            if (op->use_empty()) {
                writer.eraseOp(op);
                layer.eraseOp(op);
            }
        }
        maxIter--;
    }

    assert(layer.empty() && "Expected no remaining ops after layer erasure");
}

void constructLayer(QECLayer &layer, IRRewriter &writer)
{
    if (layer.empty())
        return;

    auto loc = layer.getOps().front()->getLoc();
    auto inOperands = ValueRange(layer.getOperands().getArrayRef());
    llvm::SmallVector<Value> orderedResults = layer.getResultsOrderedByTypeThenOperand();
    auto outResults = ValueRange(orderedResults);

    OpBuilder::InsertionGuard guard(writer);
    writer.setInsertionPointAfter(layer.getOps().back());

    auto layerOp = writer.create<qec::LayerOp>(
        loc, inOperands, outResults,
        [&](OpBuilder &builder, Location loc, ValueRange operands, ValueRange results) {
            // Map input operands to the layer's block arguments (block arguments are entries)
            IRMapping mapper;
            mapper.map(inOperands, operands);
            llvm::SmallVector<Value> newResults(results.begin(), results.end());

            for (const auto &op : layer.getOps()) {
                auto newOp = op->clone(mapper);
                builder.insert(newOp);

                // Ensure subsequent clones remap uses of this op's results
                mapper.map(op->getResults(), newOp->getResults());

                // Rewrite yield operands to the cloned results where applicable
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

    // Replace all uses of the original SSA results with the new layer results
    for (auto [prevResult, newResult] : llvm::zip(outResults, layerOp->getResults())) {
        prevResult.replaceAllUsesWith(newResult);
    }

    // Erase the original ops now that uses are replaced
    eraseUnusedOps(layer, writer);
}

bool isParentLayerOp(QECOpInterface op)
{
    // Skip ops nested inside an existing qec.layer region
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

        // Create per-pass context for layer construction
        QECLayerContext layerContext;
        QECLayer currentLayer(&layerContext);

        // Accumulate consecutive QEC ops into a layer
        getOperation()->walk([&](QECOpInterface op) {
            // Skip ops nested inside an existing qec.layer region
            if (isParentLayerOp(op))
                return WalkResult::skip();

            // Try to insert the op into the current layer
            if (currentLayer.insert(op))
                return WalkResult::skip();

            constructLayer(currentLayer, writer);

            // Start a new layer and insert the op
            currentLayer = QECLayer(&layerContext);
            currentLayer.insert(op);

            return WalkResult::advance();
        });

        constructLayer(currentLayer, writer);
    };
};

} // namespace qec
} // namespace catalyst
