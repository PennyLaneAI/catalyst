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

Value findPredecessorValueOfOp(Value qubit, QECOpInterface op)
{
    assert(qubit != nullptr && "Qubit should not be nullptr");

    auto defOp = qubit.getDefiningOp();

    if (!defOp)
        return qubit;

    if (defOp->isBeforeInBlock(op)) {
        return qubit;
    }

    auto qecOp = llvm::dyn_cast<QECOpInterface>(defOp);

    if (!qecOp)
        return nullptr;

    IRMapping outInMap;
    outInMap.map(qecOp.getOutQubits(), qecOp.getInQubits());

    auto inQubit = outInMap.lookup(qubit);

    if (qecOp == op) {
        return inQubit;
    }

    return findPredecessorValueOfOp(inQubit, op);
}

bool commute(QECOpInterface lhsOp, QECOpInterface rhsOp)
{
    if (!lhsOp->isBeforeInBlock(rhsOp))
        return false;

    std::vector<Value> lhsInQubits(lhsOp.getInQubits().begin(), lhsOp.getInQubits().end());

    // Collect all qubits that are acted on by lhsOp and rhsOp
    llvm::SetVector<Value> qubits;
    qubits.insert(lhsInQubits.begin(), lhsInQubits.end());

    // Track and collect the qubits that are acted on by rhsOp and
    // are predecessors of qubits acted on by lhsOp
    std::vector<Value> rhsOpInQubitsFromLhsOp;
    for (auto inQubit : rhsOp.getInQubits()) {
        Value v = findPredecessorValueOfOp(inQubit, lhsOp);
        if (!v) {
            llvm::outs() << "No predecessor found for " << inQubit << "\n";
            return false;
        }
        rhsOpInQubitsFromLhsOp.emplace_back(v);
    }
    qubits.insert(rhsOpInQubitsFromLhsOp.begin(), rhsOpInQubitsFromLhsOp.end());

    // Normalize the ops to get the Pauli strings
    auto normalizedOps = normalizePPROps(lhsOp, rhsOp, lhsInQubits, rhsOpInQubitsFromLhsOp);

    return normalizedOps.first.commutes(normalizedOps.second);
}

bool commuteAll(QECOpInterface rhsOp, QECLayer &lhsLayer)
{
    for (auto lhsOp : lhsLayer.getOps()) {
        if (lhsOp->getBlock() != rhsOp->getBlock())
            return false;

        assert(lhsOp != rhsOp && "lshOp and rhsOp should not be equal");

        if (!commute(lhsOp, rhsOp)) {
            return false;
        }
    }

    return true;
}

void moveOpToLayer(QECOpInterface rhsOp, QECLayer &rhsLayer, QECLayer &lhsLayer, IRRewriter &writer)
{

    auto lhsOp = lhsLayer.getOps().back();
    auto newOp = rhsOp.clone();

    // map input to output of lhsOp
    IRMapping lhsMap;
    lhsMap.map(lhsOp.getInQubits(), lhsOp.getOutQubits());

    std::vector<Value> newOperands;

    for (auto oldOpnd : rhsOp.getInQubits()) {
        auto val = findPredecessorValueOfOp(oldOpnd, lhsOp);
        assert(val != nullptr && "Qubit should not be nullptr");

        if (auto out = lhsMap.lookupOrNull(val)) {
            val = out;
        }
        newOperands.emplace_back(val);
    }

    newOp->setOperands(newOperands);

    // Update users
    for (auto [inNewOp, outNewOp] : llvm::zip(newOp.getInQubits(), newOp.getOutQubits())) {
        writer.replaceAllUsesExcept(inNewOp, outNewOp, newOp);
    }

    writer.setInsertionPointAfter(lhsOp);
    writer.insert(newOp);
    lhsLayer.insertToLayer(newOp);

    rhsLayer.removeOpRecord(rhsOp);
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

        getOperation()->walk([&](QECOpInterface op) {
            if (commuteAll(op, currentLayer)) {
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

        // for (auto &layer : layers) {
        //     llvm::outs() << "\nlayer ---------\n";
        //     for (auto op : layer.getOps()) {
        //         op->dump();
        //     }
        //     llvm::outs() << "\n";
        // }

        auto layerSize = layers.size();
        bool changed = true;
        while (changed) {
            changed = false;
            for (auto [idx, layer] : llvm::enumerate(layers)) {
                if (idx + 1 >= layerSize)
                    break;

                auto &nextLayer = layers[idx + 1];

                for (QECOpInterface op : nextLayer.getOps()) {
                    if (commuteAll(op, layer)) {
                        // move op to the beginning of next layer
                        // hoist op to the end of current layer
                        moveOpToLayer(op, nextLayer, layer, writer);
                        llvm::outs() << "COMMUTE\n";

                        if (nextLayer.getOps().empty()) {
                            layers.erase(layers.begin() + idx + 1);
                            layerSize--;
                        }

                        changed = true;
                    }
                }
            }
        }
    };
};
} // namespace qec

/// Create a pass for lowering operations in the `QECDialect`.
std::unique_ptr<Pass> createTLayerReductionPass()
{
    return std::make_unique<qec::TLayerReductionPass>();
}

} // namespace catalyst
