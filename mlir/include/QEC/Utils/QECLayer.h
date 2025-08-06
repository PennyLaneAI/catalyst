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

#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SetVector.h"

#include "QEC/IR/QECDialect.h"
#include "QEC/IR/QECOpInterfaces.h"

namespace catalyst {
namespace qec {

/// Thread-safe context for QEC layer operations.
/// Each pass instance should create its own context to ensure
/// thread safety and deterministic behavior.
class QECLayerContext {
  public:
    llvm::MapVector<mlir::Value, int> qubitValueToIndex;
    llvm::MapVector<mlir::Operation *, std::vector<int>> opToIndex;
    int MAX_INDEX = 0; // Maximum index assigned so far

    void clear()
    {
        qubitValueToIndex.clear();
        opToIndex.clear();
        MAX_INDEX = 0;
    }

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

    // Cached qubit index set
    llvm::DenseSet<int> layerQubitIndexes;

    void insertToLayer(QECOpInterface op);
    void updateResultAndOperand(QECOpInterface op);

  public:
    std::vector<QECOpInterface> ops;
    llvm::SetVector<mlir::Value> operands;
    llvm::SetVector<mlir::Value> results;

    QECLayer(QECLayerContext *ctx) : context(ctx) {}
    QECLayer(QECLayerContext *ctx, std::vector<QECOpInterface> ops) : context(ctx), ops(ops)
    {
        // Initialize the cached index set for existing ops
        for (auto op : ops) {
            auto qubitIndexes = getQubitIndexFrom(op);
            layerQubitIndexes.insert(qubitIndexes.begin(), qubitIndexes.end());
        }
    }

    ~QECLayer()
    {
        resultToOperand.clear();
        ops.clear();
        operands.clear();
        results.clear();
        layerQubitIndexes.clear();
    }

    inline bool empty() { return ops.empty(); }

    mlir::Operation *getParentLayer();

    std::vector<int> getQubitIndexFrom(QECOpInterface op);
    void setQubitIndexFrom(QECOpInterface op);

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
};

} // namespace qec
} // namespace catalyst
