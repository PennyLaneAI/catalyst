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

#define DEBUG_TYPE "t-layer-reduction"

#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"

#include "QEC/IR/QECOpInterfaces.h"
#include "QEC/Utils/PauliStringWrapper.h"
#include "QEC/Utils/QECLayer.h"
#include <mlir/IR/IRMapping.h>

using namespace mlir;
using namespace catalyst::qec;

// TLayerReduction: layer non-Clifford PPR ops, commute left to reduce depth,
// and merge equal neighbors where possible.

namespace catalyst {
namespace qec {

#define GEN_PASS_DEF_TLAYERREDUCTIONPASS
#define GEN_PASS_DECL_TLAYERREDUCTIONPASS
#include "QEC/Transforms/Passes.h.inc"

// Return the SSA value of `qubit` visible at the program point of `op`.
// Walk back through QEC PPR chains (out->in) until defined before `op` or a
// block argument. Returns nullptr if a non-QEC op defines it.
Value getReachingValueAt(Value qubit, QECOpInterface op)
{
    // We want to find the qubit that is used by op.
    // e.g., op is first PPR, while qubit can be the operands from the third PPR.
    // %0_q0, %0_q1 = qec.ppr ["X", "X"](8) %arg0, %arg1 // <- op
    // %1_q0, %1_q1 = qec.ppr ["Z", "X"](8) %0_q0, %0_q1
    // %2_q0, %2_q1 = qec.ppr ["X", "X"](8) %1_q0, %1_q1
    //                                        ^- qubit that is used by op
    // So if qubit is %1_q0, we want to return %arg0.

    assert(qubit != nullptr && "Qubit should not be nullptr");

    auto defOp = qubit.getDefiningOp();

    if (!defOp || defOp->isBeforeInBlock(op))
        return qubit;

    auto qecOp = llvm::dyn_cast<QECOpInterface>(defOp);

    if (!qecOp)
        return nullptr;

    auto outQubits = qecOp.getOutQubits();
    auto inQubits = qecOp.getInQubits();
    assert((inQubits.size() == outQubits.size()) &&
           "PPR op should have the same number of input and output qubits");

    auto pos = std::distance(outQubits.begin(), llvm::find(outQubits, qubit));
    Value inQubit = inQubits[pos];

    if (qecOp == op)
        return inQubit;

    return getReachingValueAt(inQubit, op);
}

// For each in-qubit of `srcOp`, return the SSA value visible at `dstOp`.
// One per operand; nullptr if a non-QEC definition intervenes.
// Precondition: same block with `dstOp` before `srcOp`. Consider caching.
std::vector<Value> getInQubitReachingValuesAt(QECOpInterface srcOp, QECOpInterface dstOp)
{
    std::vector<Value> dominanceQubits;
    for (auto inQubit : srcOp.getInQubits()) {
        Value v = getReachingValueAt(inQubit, dstOp);
        dominanceQubits.emplace_back(v);
    }
    return dominanceQubits;
}

std::pair<bool, QECOpInterface> checkCommutationAndFindMerge(QECOpInterface rhsOp,
                                                             QECLayer &lhsLayer)
{
    QECOpInterface mergeOp = nullptr;
    for (auto lhsOp : lhsLayer.getOps()) {
        if (lhsOp->getBlock() != rhsOp->getBlock())
            return std::pair(false, nullptr);

        assert(lhsOp != rhsOp && "lshOp and rhsOp should not be equal");
        assert(lhsOp->getBlock() == rhsOp->getBlock() &&
               "lhsOp and rhsOp should be in the same block");
        assert(lhsOp->isBeforeInBlock(rhsOp) && "lhsOp should be before rhsOp");

        std::vector<Value> lhsInQubits(lhsOp.getInQubits().begin(), lhsOp.getInQubits().end());

        // Reaching in-qubit values of `rhsOp` at the program point of `lhsOp`.
        std::vector<Value> rhsOpInQubitsFromLhsOp = getInQubitReachingValuesAt(rhsOp, lhsOp);
        if (llvm::any_of(rhsOpInQubitsFromLhsOp, [](Value qubit) { return qubit == nullptr; })) {
            // Missing reaching value (intervening non-PPR); commutation fails.
            // TODO: Handle non-PPR not directly dominated by `lhsOp`.
            return std::pair(false, nullptr);
        }

        // Normalize to Pauli strings
        auto normalizedOps = normalizePPROps(lhsOp, rhsOp, lhsInQubits, rhsOpInQubitsFromLhsOp);

        if (!normalizedOps.first.commutes(normalizedOps.second))
            return std::pair(false, nullptr);

        // Equal normalized Pauli strings => merge candidate
        auto canMerge = equal(normalizedOps.first, normalizedOps.second);
        if (canMerge)
            mergeOp = lhsOp;
    }

    return std::pair(true, mergeOp);
}

void moveOpToLayer(QECOpInterface rhsOp, QECLayer &rhsLayer, QECOpInterface mergeOp,
                   QECLayer &lhsLayer, IRRewriter &writer)
{
    //    lhsLayer   :  rhsLayer
    //    ┌───────┐  :  ┌───────┐
    //   ─┤ lhsOp ├──:──┤ rhsOp ├─
    //    └───────┘  :  └───────┘

    // Move `rhsOp` into `lhsLayer` and remap its operands to values visible at `lhsOp`.

    auto lhsOp = lhsLayer.getOps().back();

    if (mergeOp)
        lhsOp = mergeOp;

    auto newOp = rhsOp.clone();

    // Map operands via values at `lhsOp`; if unused by `lhsOp`, use the value
    // available before `lhsOp`.
    std::vector<Value> newOperands;
    IRMapping lhsInOutQubits;
    lhsInOutQubits.map(lhsOp.getInQubits(), lhsOp.getOutQubits());

    for (auto opnd : rhsOp.getInQubits()) {
        auto derivedOpnd = getReachingValueAt(opnd, lhsOp);
        assert(derivedOpnd != nullptr && "Qubit should not be nullptr");

        if (auto out = lhsInOutQubits.lookupOrNull(derivedOpnd)) {
            derivedOpnd = out;
        }
        newOperands.emplace_back(derivedOpnd);
    }

    newOp->setOperands(newOperands);

    // Replace uses of newOp's in-qubits with its out-qubits.
    for (auto [inNewOp, outNewOp] : llvm::zip(newOp.getInQubits(), newOp.getOutQubits())) {
        writer.replaceAllUsesExcept(inNewOp, outNewOp, newOp);
    }

    writer.setInsertionPointAfter(lhsOp);
    writer.insert(newOp);
    lhsLayer.insertToLayer(newOp);

    rhsLayer.eraseOp(rhsOp);
    // Rewire users of the erased `rhsOp` to its original operands.
    writer.replaceAllUsesWith(rhsOp->getResults(), rhsOp->getOperands());
    writer.eraseOp(rhsOp);
}

struct TLayerReductionPass : impl::TLayerReductionPassBase<TLayerReductionPass> {
    using TLayerReductionPassBase::TLayerReductionPassBase;

    void runOnOperation() final
    {
        MLIRContext *context = &getContext();
        IRRewriter writer(context);
        QECLayerContext layerContext;
        QECLayer currentLayer(&layerContext);

        std::vector<QECLayer> layers;

        // 1) Build initial layers:
        // - try to commute non-Clifford PPR into current layer;
        // - else start a new layer.
        getOperation()->walk([&](PPRotationOp op) {
            if (op.isClifford())
                return WalkResult::advance();
            auto [isCommute, _] = checkCommutationAndFindMerge(op, currentLayer);
            if (isCommute) {
                currentLayer.insertToLayer(op);
                return WalkResult::advance();
            }

            layers.emplace_back(std::move(currentLayer));

            // Start a new layer and insert the op
            currentLayer = QECLayer(&layerContext);
            currentLayer.insert(op);

            return WalkResult::advance();
        });
        layers.emplace_back(std::move(currentLayer));

        // 2) Reduce depth
        auto layerSize = layers.size();
        bool changed = true;

        do {
            changed = false;
            for (size_t idx = layerSize - 1; idx > 0; --idx) {
                auto &currentLayer = layers[idx];
                auto &prevLayer = layers[idx - 1];

                for (QECOpInterface op : currentLayer.getOps()) {
                    auto [isCommute, mergeOp] = checkCommutationAndFindMerge(op, prevLayer);
                    if (isCommute) {
                        moveOpToLayer(op, currentLayer, mergeOp, prevLayer, writer);
                        changed = true;
                    }
                }

                if (currentLayer.getOps().empty()) {
                    layers.erase(layers.begin() + idx);
                    layerSize--;
                    changed = true;
                }
            }
        } while (changed);
    };
};
} // namespace qec

/// Create a pass for lowering operations in the `QECDialect`.
std::unique_ptr<Pass> createTLayerReductionPass()
{
    return std::make_unique<qec::TLayerReductionPass>();
}

} // namespace catalyst
