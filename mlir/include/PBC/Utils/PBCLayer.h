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

#include "PBC/IR/PBCDialect.h"
#include "PBC/IR/PBCOpInterfaces.h"
#include "PBC/IR/PBCOps.h"
#include "Quantum/IR/QuantumDialect.h"

namespace catalyst {
namespace pbc {

class PBCLayer;
/// Each pass instance creates its own context to ensure
/// thread safety and deterministic behavior.
class PBCLayerContext {
  public:
    llvm::MapVector<LayerOp, std::unique_ptr<PBCLayer>> layers;

    // Clear all per-pass cached layer objects.
    void clear() { layers.clear(); }

    PBCLayerContext() = default;
    ~PBCLayerContext() = default;

    // Disable copy to prevent accidental sharing
    PBCLayerContext(const PBCLayerContext &) = delete;
    PBCLayerContext &operator=(const PBCLayerContext &) = delete;

    // Allow move semantics
    PBCLayerContext(PBCLayerContext &&) = default;
    PBCLayerContext &operator=(PBCLayerContext &&) = default;
};

class PBCLayer {
  private:
    llvm::DenseMap<mlir::Value, mlir::Value> resultToOperand;
    PBCLayerContext *context; // Reference to thread-local context

    // Cached canonical entry qubit set for the layer
    llvm::DenseSet<mlir::Value> layerEntryQubits;

    // Per-layer deterministic mapping from any seen qubit SSA value to its
    // canonical entry (the block argument it originated from in this layer)
    llvm::DenseMap<mlir::Value, mlir::Value> localQubitToEntry;

    // Internal storage
    std::vector<PBCOpInterface> ops;
    llvm::SetVector<mlir::Value> operands;
    llvm::SetVector<mlir::Value> results;

    // Resolve the canonical entry for a given qubit value by chasing
    // local and result->operand mappings deterministically.
    mlir::Value resolveEntry(mlir::Value v) const;

    void updateResultAndOperand(PBCOpInterface op);

  public:
    LayerOp layerOp;

    // Disallow copying; allow moves
    PBCLayer(const PBCLayer &) = delete;
    PBCLayer &operator=(const PBCLayer &) = delete;
    PBCLayer(PBCLayer &&) = default;
    PBCLayer &operator=(PBCLayer &&) = default;
    ~PBCLayer() = default;

    PBCLayer(PBCLayerContext *ctx) : context(ctx) {}
    PBCLayer(PBCLayerContext *ctx, const std::vector<PBCOpInterface> &initialOps) : context(ctx)
    {
        // Initialize the cached index set for existing ops
        std::for_each(initialOps.begin(), initialOps.end(),
                      [this](PBCOpInterface op) { insertToLayer(op); });
    }

    inline bool empty() const { return ops.empty(); }

    const std::vector<PBCOpInterface> &getOps() const { return ops; }
    const llvm::SetVector<mlir::Value> &getOperands() const { return operands; }
    const llvm::SetVector<mlir::Value> &getResults() const { return results; }

    // Mutator for removing an op record from the layer bookkeeping
    void eraseOp(PBCOpInterface op);

    std::vector<mlir::Value> getEntryQubitsFrom(PBCOpInterface op);
    std::vector<mlir::Value> getEntryQubitsFrom(YieldOp yieldOp);

    bool actOnDisjointQubits(PBCOpInterface op);

    // Commute two ops if they act on the same qubits based on qubit indexes on that layer
    // NOTE: This is used only for PBCLayer where LayerOp is present.
    bool commute(PBCOpInterface src, PBCOpInterface dst);
    bool commuteToLayer(PBCOpInterface op);

    // Check if the op is in the same block as the layer
    bool isSameBlock(PBCOpInterface op) const;

    // Extract ops defining operands must be before existing ops;
    // no insert users before existing ops
    bool extractsAreBeforeExistingOps(PBCOpInterface op) const;
    bool insertsAreAfterExistingOps(PBCOpInterface op) const;

    // Directly insert an op to the layer without checking commutation
    void insertToLayer(PBCOpInterface op);

    // Op can be inserted into the layer if:
    // 1. It is in the same block
    // 2. It does not have extract(insert) op before(after) existing ops
    // 3. It acts on disjoint qubits
    // 4. Or it commutes with all the ops in the layer
    bool insert(PBCOpInterface op);

    // Get the current context
    PBCLayerContext *getContext() const { return context; }

    // Return layer results ordered as:
    //  1. Non-qubit (classical) results first, preserving program order
    //  2. Qubit results grouped by their originating operand (entry qubit) order
    llvm::SmallVector<mlir::Value> getResultsOrderedByTypeThenOperand() const;
};

} // namespace pbc
} // namespace catalyst
