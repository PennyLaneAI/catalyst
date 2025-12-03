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

#include "llvm/ADT/STLExtras.h"

#include "QEC/IR/QECOps.h"
#include "QEC/Utils/PauliStringWrapper.h"
#include "QEC/Utils/QECLayer.h"
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

void QECLayer::eraseOp(QECOpInterface op) { llvm::erase(ops, op); }

void QECLayer::updateResultAndOperand(QECOpInterface op)
{
    // Ensure layer operand set contains canonical origins for any input qubits
    ValueRange inQubits = op.getInQubits();
    ValueRange outQubits = op.getOutQubits();

    // Map each input qubit to its canonical origin (entry) for this layer
    llvm::SmallVector<Value> inputOrigins;
    inputOrigins.reserve(inQubits.size());

    for (Value in : inQubits) {
        // If the operand was a previously exposed result, remove it from layer results
        if (results.contains(in)) {
            results.remove(in);
        }

        if (resultToOperand.contains(in)) {
            in = resultToOperand[in];
        }
        inputOrigins.push_back(in);
        operands.insert(in);
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
}

std::vector<Value> QECLayer::getEntryQubitsFrom(QECOpInterface op)
{
    std::vector<Value> entryQubits;
    entryQubits.reserve(op.getInQubits().size());

    for (auto [inQubit, outQubit] : llvm::zip(op.getInQubits(), op.getOutQubits())) {
        // Resolve entry for inQubit within this layer
        // If inQubit is a region argument of this layer, it is the entry;
        // otherwise, if it is produced by a previous op in this layer,
        // we should have mapped its defining value to an entry already.
        // Fallback: use inQubit itself.
        Value entry = localQubitToEntry.contains(inQubit) ? localQubitToEntry[inQubit] : inQubit;

        entryQubits.push_back(entry);
        // Propagate the entry to the result value inside this layer
        localQubitToEntry[outQubit] = entry;
    }

    return entryQubits;
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

bool QECLayer::actOnDisjointQubits(QECOpInterface op)
{
    // Check for overlap with cached index set
    return llvm::none_of(getEntryQubitsFrom(op),
                         [&](const auto &q) { return layerEntryQubits.contains(q); });
}

// Commute two ops if they act on the same qubits based on qubit indexes on that layer
bool QECLayer::commute(QECOpInterface src, QECOpInterface dst)
{
    auto srcEntryQubits = getEntryQubitsFrom(src);
    auto dstEntryQubits = getEntryQubitsFrom(dst);

    auto normalizedOps = normalizePPROps(src, dst, srcEntryQubits, dstEntryQubits);

    return normalizedOps.first.commutes(normalizedOps.second);
}

// Commute an op to all the ops in the layer
bool QECLayer::commuteToLayer(QECOpInterface op)
{
    return llvm::all_of(ops, [&](auto existingOp) { return commute(op, existingOp); });
}

bool QECLayer::isSameBlock(QECOpInterface op) const
{
    if (ops.empty())
        return true;
    return op->getBlock() == ops.back()->getBlock();
}

// Check if the op has extract op that must be occurred before the operations in layers
bool QECLayer::extractsAreBeforeExistingOps(QECOpInterface op) const
{
    for (auto existingOp : ops) {
        for (auto operand : op->getOperands()) {
            auto defOp = operand.getDefiningOp();
            // Only meaningful to compare within the same block
            if (auto extractOp = llvm::dyn_cast_or_null<quantum::ExtractOp>(defOp)) {
                if (extractOp->getBlock() == existingOp->getBlock() &&
                    !extractOp->isBeforeInBlock(existingOp)) {
                    return false;
                }
            }
        }
    }
    return true;
}

// Ensure the new op does not have insert op before existing ops
bool QECLayer::insertsAreAfterExistingOps(QECOpInterface op) const
{
    for (auto existingOp : ops) {
        for (auto result : op->getResults()) {
            for (auto user : result.getUsers()) {
                if (auto insertOp = llvm::dyn_cast<quantum::InsertOp>(user)) {
                    if (insertOp->getBlock() == existingOp->getBlock() &&
                        insertOp->isBeforeInBlock(existingOp)) {
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
    // 2. Extracts for operands occur before, and there are no inserts before existing ops
    if (!isSameBlock(op) || !extractsAreBeforeExistingOps(op) || !insertsAreAfterExistingOps(op)) {
        return false;
    }

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

    // 1. Collect classical first in program order
    for (const auto &v : results) {
        if (!llvm::isa<catalyst::quantum::QubitType>(v.getType())) {
            ordered.push_back(v);
        }
    }

    // 2. Group qubit results by their originating operand (entry qubit) order
    // Build buckets: origin -> list of results (preserve result order)
    llvm::DenseMap<mlir::Value, llvm::SmallVector<mlir::Value>> originToQubitResults;
    originToQubitResults.reserve(results.size());
    for (const auto &v : results) {
        if (!llvm::isa<catalyst::quantum::QubitType>(v.getType())) {
            continue;
        }
        if (auto it = resultToOperand.find(v); it != resultToOperand.end()) {
            originToQubitResults[it->second].push_back(v);
        }
    }

    for (const auto &origin : operands) {
        if (auto it = originToQubitResults.find(origin); it != originToQubitResults.end()) {
            ordered.append(it->second.begin(), it->second.end());
        }
    }

    return ordered;
}

} // namespace qec
} // namespace catalyst
