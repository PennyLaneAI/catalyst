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

#include "llvm/ADT/SetVector.h"
// #include "llvm/Support/Casting.h"

#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"

#include "QEC/IR/QECDialect.h"
#include "QEC/IR/QECOpInterfaces.h"
#include "Quantum/IR/QuantumDialect.h"
#include "mlir/IR/Block.h"

namespace catalyst {
namespace qec {

class QECLayer;
/// Thread-safe context for QEC layer operations.
/// Each pass instance should create its own context to ensure
/// thread safety and deterministic behavior.
class QECLayerContext {
  public:
    llvm::MapVector<LayerOp, std::unique_ptr<QECLayer>> layers;

    // Clear all per-pass cached layer objects.
    void clear() { layers.clear(); }

    QECLayerContext() = default;
    ~QECLayerContext() = default;

    // Disable copy to prevent accidental sharing
    QECLayerContext(const QECLayerContext &) = delete;
    QECLayerContext &operator=(const QECLayerContext &) = delete;

    // Allow move semantics
    QECLayerContext(QECLayerContext &&) = default;
    QECLayerContext &operator=(QECLayerContext &&) = default;
};

class QECLayer {
  private:
    llvm::DenseMap<mlir::Value, mlir::Value> resultToOperand;
    QECLayerContext *context; // Reference to thread-local context

    // Cached canonical entry qubit set for the layer
    llvm::DenseSet<mlir::Value> layerEntryQubits;

    // Per-layer deterministic mapping from any seen qubit value to its
    // canonical entry (the region argument it originated from in this layer)
    llvm::DenseMap<mlir::Value, mlir::Value> localQubitToEntry;
    // Cache per-op entry qubits in operand order for this layer only
    llvm::DenseMap<mlir::Operation *, std::vector<mlir::Value>> localOpToEntryQubits;

    // Compute and cache entry qubits for an op using only per-layer state
    void computeAndCacheEntryQubitsForOp(QECOpInterface op);

    void insertToLayer(QECOpInterface op);
    void updateResultAndOperand(QECOpInterface op);

  public:
    std::vector<QECOpInterface> ops;
    llvm::SetVector<mlir::Value> operands;
    llvm::SetVector<mlir::Value> results;
    LayerOp layerOp;

    QECLayer(QECLayerContext *ctx) : context(ctx) {}
    QECLayer(QECLayerContext *ctx, const std::vector<QECOpInterface> &initialOps) : context(ctx)
    {
        // Initialize the cached index set for existing ops
        for (auto op : initialOps) {
            insertToLayer(op);
        }
    }

    QECLayer(QECLayerContext *ctx, LayerOp layerOp) : context(ctx)
    {
        layerOp->walk([&](QECOpInterface op) { insertToLayer(op); });
        this->layerOp = layerOp;

        // Also consider qubit-typed operands of the layer terminator (qec.yield)
        // as entries to seed local and layer-level mappings.
        if (auto *term = layerOp.getBody()->getTerminator()) {
            if (auto yield = llvm::dyn_cast<YieldOp>(term)) {
                for (mlir::Value yOperand : yield->getOperands()) {
                    if (!llvm::isa<catalyst::quantum::QubitType>(yOperand.getType())) {
                        continue;
                    }

                    mlir::Value entry = yOperand;
                    if (resultToOperand.contains(yOperand)) {
                        entry = resultToOperand[yOperand];
                    }

                    localQubitToEntry[yOperand] = entry;
                    if (llvm::isa<mlir::BlockArgument>(entry)) {
                        layerEntryQubits.insert(entry);
                    }
                }
            }
        }
    }

    static QECLayer &build(QECLayerContext *ctx, LayerOp layerOp)
    {
        if (ctx->layers.contains(layerOp)) {
            return *ctx->layers[layerOp];
        }
        auto layer = std::make_unique<QECLayer>(ctx, layerOp);
        ctx->layers[layerOp] = std::move(layer);
        return *ctx->layers[layerOp];
    }

    ~QECLayer()
    {
        resultToOperand.clear();
        ops.clear();
        operands.clear();
        results.clear();
        layerEntryQubits.clear();
    }

    inline bool empty() { return ops.empty(); }

    mlir::Operation *getParentLayer();

    std::vector<mlir::Value> getEntryQubitsFrom(QECOpInterface op);
    std::vector<mlir::Value> getEntryQubitsFrom(YieldOp yieldOp);
    std::vector<mlir::Value> getEntryQubitsFrom(mlir::Operation *op);

    bool actOnDisjointQubits(QECOpInterface op);

    // Commute two ops if they act on the same qubits based on qubit indexes on that layer
    bool commute(QECOpInterface fromOp, QECOpInterface toOp);
    bool commuteToLayer(QECOpInterface op);

    // Check if the op is in the same block as the layer
    bool isSameBlock(QECOpInterface op);

    // Ensure the new op does not have extract(insert) op before(after) existing ops
    bool hasNoExtractAfter(QECOpInterface op);
    bool hasNoInsertAfter(QECOpInterface op);

    // Op can be inserted if:
    // 1. It is in the same block
    // 2. It does not have extract(insert) op before(after) existing ops
    // 3. It acts on disjoint qubits
    // 4. Or it commutes with all the ops in the layer
    bool insert(QECOpInterface op);

    // Get the current context (for debugging)
    QECLayerContext *getContext() const { return context; }

    // Return layer results ordered as:
    //  1. All non-qubit (classical) results first, preserving program order
    //  2. Followed by qubit results ordered to match the order of layer operands
    llvm::SmallVector<mlir::Value> getResultsOrderedByTypeThenOperand() const;
};

} // namespace qec
} // namespace catalyst
