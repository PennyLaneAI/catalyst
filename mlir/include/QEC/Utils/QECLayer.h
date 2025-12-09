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

#include "QEC/IR/QECDialect.h"
#include "QEC/IR/QECOpInterfaces.h"
#include "QEC/IR/QECOps.h"
#include "Quantum/IR/QuantumDialect.h"

namespace catalyst {
namespace qec {

class QECLayer;
/// Each pass instance creates its own context to ensure
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

    // Per-layer deterministic mapping from any seen qubit SSA value to its
    // canonical entry (the block argument it originated from in this layer)
    llvm::DenseMap<mlir::Value, mlir::Value> localQubitToEntry;

    // Internal storage
    std::vector<QECOpInterface> ops;
    llvm::SetVector<mlir::Value> operands;
    llvm::SetVector<mlir::Value> results;

    // Resolve the canonical entry for a given qubit value by chasing
    // local and result->operand mappings deterministically.
    mlir::Value resolveEntry(mlir::Value v) const;

    void updateResultAndOperand(QECOpInterface op);

  public:
    LayerOp layerOp;

    // Disallow copying; allow moves
    QECLayer(const QECLayer &) = delete;
    QECLayer &operator=(const QECLayer &) = delete;
    QECLayer(QECLayer &&) = default;
    QECLayer &operator=(QECLayer &&) = default;
    ~QECLayer() = default;

    QECLayer(QECLayerContext *ctx) : context(ctx) {}
    QECLayer(QECLayerContext *ctx, const std::vector<QECOpInterface> &initialOps) : context(ctx)
    {
        // Initialize the cached index set for existing ops
        std::for_each(initialOps.begin(), initialOps.end(),
                      [this](QECOpInterface op) { insertToLayer(op); });
    }

    inline bool empty() const { return ops.empty(); }

    const std::vector<QECOpInterface> &getOps() const { return ops; }
    const llvm::SetVector<mlir::Value> &getOperands() const { return operands; }
    const llvm::SetVector<mlir::Value> &getResults() const { return results; }

    // Mutator for removing an op record from the layer bookkeeping
    void eraseOp(QECOpInterface op);

    std::vector<mlir::Value> getEntryQubitsFrom(QECOpInterface op);
    std::vector<mlir::Value> getEntryQubitsFrom(YieldOp yieldOp);

    bool actOnDisjointQubits(QECOpInterface op);

    // Commute two ops if they act on the same qubits based on qubit indexes on that layer
    // NOTE: This is used only for QECLayer where LayerOp is present.
    bool commute(QECOpInterface src, QECOpInterface dst);
    bool commuteToLayer(QECOpInterface op);

    // Check if the op is in the same block as the layer
    bool isSameBlock(QECOpInterface op) const;

    // Extract ops defining operands must be before existing ops;
    // no insert users before existing ops
    bool extractsAreBeforeExistingOps(QECOpInterface op) const;
    bool insertsAreAfterExistingOps(QECOpInterface op) const;

    // Directly insert an op to the layer without checking commutation
    void insertToLayer(QECOpInterface op);

    // Op can be inserted into the layer if:
    // 1. It is in the same block
    // 2. It does not have extract(insert) op before(after) existing ops
    // 3. It acts on disjoint qubits
    // 4. Or it commutes with all the ops in the layer
    bool insert(QECOpInterface op);

    // Get the current context
    QECLayerContext *getContext() const { return context; }

    // Return layer results ordered as:
    //  1. Non-qubit (classical) results first, preserving program order
    //  2. Qubit results grouped by their originating operand (entry qubit) order
    llvm::SmallVector<mlir::Value> getResultsOrderedByTypeThenOperand() const;
};

} // namespace qec
} // namespace catalyst
