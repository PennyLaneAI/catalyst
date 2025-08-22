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

namespace catalyst {
namespace qec {

#define GEN_PASS_DEF_TLAYERREDUCTIONPASS
#define GEN_PASS_DECL_TLAYERREDUCTIONPASS
#include "QEC/Transforms/Passes.h.inc"

// This aims to find the qubit that may be used by op
// If not used by op, it returns the qubits that happens before op.
Value findPredecessorValueOfOp(Value qubit, QECOpInterface op)
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

    IRMapping outInMap;
    outInMap.map(qecOp.getOutQubits(), qecOp.getInQubits());

    auto inQubit = outInMap.lookup(qubit);

    if (qecOp == op)
        return inQubit;

    return findPredecessorValueOfOp(inQubit, op);
}

bool commute(QECOpInterface lhsOp, QECOpInterface rhsOp)
{
    assert(lhsOp != rhsOp && "lshOp and rhsOp should not be equal");
    assert(lhsOp->getBlock() == rhsOp->getBlock() && "lhsOp and rhsOp should be in the same block");
    assert(lhsOp->isBeforeInBlock(rhsOp) && "lhsOp should be before rhsOp");

    std::vector<Value> lhsInQubits(lhsOp.getInQubits().begin(), lhsOp.getInQubits().end());

    // Track and collect the qubits that are acted on by rhsOp and
    // are predecessors of qubits acted on by lhsOp
    std::vector<Value> rhsOpInQubitsFromLhsOp;
    for (auto inQubit : rhsOp.getInQubits()) {
        Value v = findPredecessorValueOfOp(inQubit, lhsOp);
        if (!v) {
            // No predecessor found for the qubit
            // This means there is a non-PPR op between lhsOp and rhsOp
            // and the commutation check fails.
            // TODO: Handle the case where the non-PPR is not directly dominated by lhsOp
            return false;
        }
        rhsOpInQubitsFromLhsOp.emplace_back(v);
    }

    // Normalize the ops to get the Pauli strings
    auto normalizedOps = normalizePPROps(lhsOp, rhsOp, lhsInQubits, rhsOpInQubitsFromLhsOp);

    return normalizedOps.first.commutes(normalizedOps.second);
}

bool commuteOps(QECOpInterface rhsOp, QECLayer &lhsLayer)
{
    return llvm::all_of(lhsLayer.getOps(), [rhsOp](QECOpInterface lhsOp) {
        return lhsOp->getBlock() == rhsOp->getBlock() && commute(lhsOp, rhsOp);
    });
}

void moveOpToLayer(QECOpInterface rhsOp, QECLayer &rhsLayer, QECLayer &lhsLayer, IRRewriter &writer)
{
    //    lhsLayer   :  rhsLayer
    //    ┌───────┐  :  ┌───────┐
    //   ─┤ lhsOp ├──:──┤ rhsOp ├─
    //    └───────┘  :  └───────┘

    // We want to move rhsOp to the lhsLayer
    // and replace the uses of the qubits in the rhsOp with the qubits in the lhsOp

    auto lhsOp = lhsLayer.getOps().back();
    auto newOp = rhsOp.clone();

    // Mapped qubits to the corresponding operands of lhsOp
    // Note: If there are qubits that are not used by lhsOp,
    // its to use the qubits that are before the `lhsOp` in the IR.
    // This is required so we can perform the commutation check in `commute`
    std::vector<Value> newOperands;
    IRMapping lhsInOutQubits;
    lhsInOutQubits.map(lhsOp.getInQubits(), lhsOp.getOutQubits());

    for (auto opnd : rhsOp.getInQubits()) {
        auto derivedOpnd = findPredecessorValueOfOp(opnd, lhsOp);
        assert(derivedOpnd != nullptr && "Qubit should not be nullptr");

        if (auto out = lhsInOutQubits.lookupOrNull(derivedOpnd)) {
            derivedOpnd = out;
        }
        newOperands.emplace_back(derivedOpnd);
    }

    newOp->setOperands(newOperands);

    // Replace the uses of the qubits in the newOp with the derived operands.
    for (auto [inNewOp, outNewOp] : llvm::zip(newOp.getInQubits(), newOp.getOutQubits())) {
        writer.replaceAllUsesExcept(inNewOp, outNewOp, newOp);
    }

    writer.setInsertionPointAfter(lhsOp);
    writer.insert(newOp);
    lhsLayer.insertToLayer(newOp);

    rhsLayer.eraseOp(rhsOp);
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

        // 1. Initialize the layers
        getOperation()->walk([&](PPRotationOp op) {
            if (op.isClifford())
                return WalkResult::advance();

            if (commuteOps(op, currentLayer)) {
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

        // 2. Perform the depth reduction
        auto layerSize = layers.size();
        bool changed = true;

        do {
            changed = false;
            for (size_t idx = layerSize - 1; idx > 0; --idx) {
                auto &currentLayer = layers[idx];
                auto &prevLayer = layers[idx - 1];

                for (QECOpInterface op : currentLayer.getOps()) {
                    if (commuteOps(op, prevLayer)) {
                        moveOpToLayer(op, currentLayer, prevLayer, writer);
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
