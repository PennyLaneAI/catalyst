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
    for (auto [operandValOpt, resultValOpt] :
         llvm::zip_longest(op->getOperands(), op->getResults())) {
        Value operandValue = operandValOpt.value_or(nullptr);
        Value resultValue = resultValOpt.value_or(nullptr);

        // update operands
        if (operandValue != nullptr) {
            if (resultToOperand.contains(operandValue)) {
                auto originValue = resultToOperand[operandValue];
                resultToOperand[resultValue] = originValue;
                resultToOperand.erase(operandValue);
            }
            else {
                resultToOperand[resultValue] = operandValue;
                operands.insert(operandValue);
            }
        }

        // update results
        if (resultValue != nullptr) {
            if (results.contains(operandValue)) {
                results.remove(operandValue);
            }
            results.insert(resultValue);
        }
    }
}

Operation *QECLayer::getParentLayer()
{
    if (ops.empty())
        return nullptr;

    return ops.back()->getParentOp();
}

void QECLayer::setEntryQubitsFrom(QECOpInterface op)
{
    assert(context != nullptr && "QECLayerContext cannot be null");
    std::vector<Value> entryQubits;

    for (auto [inQubit, outQubit] : llvm::zip(op.getInQubits(), op.getOutQubits())) {
        if (context->qubitValueToEntry.contains(inQubit)) {
            Value entry = context->qubitValueToEntry[inQubit];
            entryQubits.push_back(entry);
            context->qubitValueToEntry[outQubit] = entry;
        }
        else {
            context->qubitValueToEntry[outQubit] = inQubit;
            entryQubits.push_back(inQubit);
        }
    }
    context->opToEntryQubits[op] = entryQubits;
}

std::vector<Value> QECLayer::getEntryQubitsFrom(QECOpInterface op)
{
    assert(context != nullptr && "QECLayerContext cannot be null");
    if (context->opToEntryQubits.contains(op)) {
        return context->opToEntryQubits[op];
    }

    setEntryQubitsFrom(op);
    return context->opToEntryQubits[op];
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

} // namespace qec
} // namespace catalyst
