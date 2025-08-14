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

#include "QEC/Utils/QECLayer.h"
#include "QEC/Utils/PauliStringWrapper.h"
#include "Quantum/IR/QuantumOps.h" // for quantum.extract op

using namespace catalyst::qec;

namespace catalyst {
namespace qec {

void QECLayer::insertToLayer(QECOpInterface op)
{
    ops.emplace_back(op);
    updateResultAndOperand(op);

    // Update the cached entry qubit set when inserting
    auto entryQubits = getEntryQubitsFrom(op);
    layerEntryQubits.insert(entryQubits.begin(), entryQubits.end());
}

void QECLayer::updateResultAndOperand(QECOpInterface op)
{
    // Ensure layer operand set contains canonical origins for any input qubits
    llvm::SmallVector<Value> inQubits(op.getInQubits().begin(), op.getInQubits().end());
    llvm::SmallVector<Value> outQubits(op.getOutQubits().begin(), op.getOutQubits().end());

    // Map each input qubit to its canonical origin (entry) for this layer
    llvm::SmallVector<Value> inputOrigins;
    inputOrigins.reserve(inQubits.size());
    for (Value in : inQubits) {
        // If the operand was a previously exposed result, remove it from layer results
        if (results.contains(in)) {
            results.remove(in);
        }

        Value origin = in;
        if (resultToOperand.contains(in)) {
            origin = resultToOperand[in];
        }
        inputOrigins.push_back(origin);
        operands.insert(origin);
    }

    // Insert non-qubit results first (e.g., classical measurements)
    for (Value r : op->getResults()) {
        if (!llvm::isa<catalyst::quantum::QubitType>(r.getType())) {
            results.insert(r);
        }
    }

    // Pair qubit outputs with corresponding input origins and update mappings
    for (auto [origin, out] : llvm::zip(inputOrigins, outQubits)) {
        resultToOperand[out] = origin;
        results.insert(out);
    }

    // For QEC ops, number of out qubits matches number of in qubits.
}

Operation *QECLayer::getParentLayer()
{
    if (ops.empty())
        return nullptr;

    return ops.back()->getParentOp();
}

void QECLayer::computeAndCacheEntryQubitsForOp(QECOpInterface op)
{
    std::vector<Value> entryQubits;
    entryQubits.reserve(op.getInQubits().size());

    for (auto [inQubit, outQubit] : llvm::zip(op.getInQubits(), op.getOutQubits())) {
        // Resolve entry for inQubit within this layer
        Value entry;
        if (localQubitToEntry.contains(inQubit)) {
            entry = localQubitToEntry[inQubit];
        }
        else {
            // If inQubit is a region argument of this layer, it is the entry;
            // otherwise, if it is produced by a previous op in this layer,
            // we should have mapped its defining value to an entry already.
            // Fallback: use inQubit itself.
            entry = inQubit;
        }

        entryQubits.push_back(entry);
        // Propagate the entry to the result value inside this layer
        localQubitToEntry[outQubit] = entry;
    }

    localOpToEntryQubits[op] = std::move(entryQubits);
}

std::vector<Value> QECLayer::getEntryQubitsFrom(QECOpInterface op)
{
    if (localOpToEntryQubits.contains(op)) {
        return localOpToEntryQubits[op];
    }

    computeAndCacheEntryQubitsForOp(op);
    return localOpToEntryQubits[op];
}

std::vector<Value> QECLayer::getEntryQubitsFrom(YieldOp yieldOp)
{
    std::vector<Value> entries;
    entries.reserve(yieldOp->getNumOperands());
    for (Value yOperand : yieldOp->getOperands()) {
        if (!llvm::isa<catalyst::quantum::QubitType>(yOperand.getType())) {
            continue;
        }
        Value entry = yOperand;
        if (localQubitToEntry.contains(yOperand)) {
            entry = localQubitToEntry[yOperand];
        }
        else if (resultToOperand.contains(yOperand)) {
            entry = resultToOperand[yOperand];
        }
        entries.push_back(entry);
    }
    return entries;
}

std::vector<Value> QECLayer::getEntryQubitsFrom(Operation *op)
{
    if (auto qecIface = llvm::dyn_cast<QECOpInterface>(op)) {
        return getEntryQubitsFrom(qecIface);
    }
    if (auto yield = llvm::dyn_cast<YieldOp>(op)) {
        return getEntryQubitsFrom(yield);
    }
    return {};
}

bool QECLayer::actOnDisjointQubits(QECOpInterface op)
{
    auto entryQubits = getEntryQubitsFrom(op);

    // Check for overlap with cached index set
    for (auto q : entryQubits) {
        if (layerEntryQubits.contains(q)) {
            return false; // Found an overlap
        }
    }

    return true;
}

// Commute two ops if they act on the same qubits based on qubit indexes on that layer
bool QECLayer::commute(QECOpInterface fromOp, QECOpInterface toOp)
{
    auto lhsEntryQubits = getEntryQubitsFrom(fromOp);
    auto rhsEntryQubits = getEntryQubitsFrom(toOp);

    llvm::SetVector<Value> qubits;
    qubits.insert(lhsEntryQubits.begin(), lhsEntryQubits.end());
    qubits.insert(rhsEntryQubits.begin(), rhsEntryQubits.end());

    PauliWord lhsPauliWord = expandPauliWord(qubits, lhsEntryQubits, fromOp);
    PauliWord rhsPauliWord = expandPauliWord(qubits, rhsEntryQubits, toOp);

    auto lhsPSWrapper = PauliStringWrapper::from_pauli_word(lhsPauliWord);
    auto rhsPSWrapper = PauliStringWrapper::from_pauli_word(rhsPauliWord);

    return lhsPSWrapper.commutes(rhsPSWrapper);
}

// Commute an op to all the ops in the layer
bool QECLayer::commuteToLayer(QECOpInterface op)
{
    for (auto existingOp : ops) {
        if (!commute(op, existingOp)) {
            return false;
        }
    }
    return true;
}

bool QECLayer::isSameBlock(QECOpInterface op) { return op->getParentOp() == getParentLayer(); }

// Check if the op has extract op that must be occurred before the operations in layers
bool QECLayer::hasNoExtractAfter(QECOpInterface op)
{
    for (auto existingOp : ops) {
        for (auto operand : op->getOperands()) {
            auto defOp = operand.getDefiningOp();
            if (auto extractOp = llvm::dyn_cast_or_null<quantum::ExtractOp>(defOp)) {
                // If extractOp must be occurred before the operations in layers
                if (!extractOp->isBeforeInBlock(existingOp)) {
                    return false;
                }
            }
        }
    }
    return true;
}

// Ensure the new op does not have insert op before existing ops
bool QECLayer::hasNoInsertAfter(QECOpInterface op)
{
    for (auto existingOp : ops) {
        for (auto result : op->getResults()) {
            for (auto user : result.getUsers()) {
                if (auto insertOp = llvm::dyn_cast<quantum::InsertOp>(user)) {
                    if (insertOp->isBeforeInBlock(existingOp)) {
                        return false;
                    }
                }
            }
        }
    }
    return true;
}

bool QECLayer::insert(QECOpInterface op)
{
    if (empty()) {
        insertToLayer(op);
        return true;
    }

    // 1. It is in the same block
    // 2. It does not have extract(insert) op before(after) existing ops
    if (!isSameBlock(op) || !hasNoExtractAfter(op) || !hasNoInsertAfter(op))
        return false;

    // 3. It acts on disjoint qubits
    // 4. Or it commutes with all the ops in the layer
    if (actOnDisjointQubits(op)) {
        insertToLayer(op);
        return true;
    }

    if (commuteToLayer(op)) {
        insertToLayer(op);
        return true;
    }

    return false;
}

llvm::SmallVector<mlir::Value> QECLayer::getResultsOrderedByTypeThenOperand() const
{
    llvm::SmallVector<mlir::Value> ordered;

    // 1. Classical first (stable)
    for (mlir::Value v : results) {
        if (!llvm::isa<catalyst::quantum::QubitType>(v.getType())) {
            ordered.push_back(v);
        }
    }

    // 2. Qubits grouped by operand order via origin mapping (stable nested scan)
    for (mlir::Value operand : operands) {
        for (mlir::Value v : results) {
            if (!llvm::isa<catalyst::quantum::QubitType>(v.getType()))
                continue;
            auto it = resultToOperand.find(v);
            if (it != resultToOperand.end() && it->second == operand) {
                ordered.push_back(v);
            }
        }
    }

    return ordered;
}

} // namespace qec
} // namespace catalyst
