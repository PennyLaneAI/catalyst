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

#define DEBUG_TYPE "reduce-t-depth"

#include <vector>

#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"

#include "PBC/IR/PBCOps.h"
#include "PBC/Utils/PBCLayer.h"
#include "PBC/Utils/PBCOpUtils.h"
#include "PBC/Utils/PauliStringWrapper.h"

using namespace mlir;
using namespace catalyst::pbc;

// TLayerReduction: layer non-Clifford PPR ops, commute left to reduce depth,
// and merge equal neighbors where possible.

namespace catalyst {
namespace pbc {

#define GEN_PASS_DECL_TLAYERREDUCTIONPASS
#define GEN_PASS_DEF_TLAYERREDUCTIONPASS
#include "PBC/Transforms/Passes.h.inc"

// Check whether `rhsOp` can commute left across every op in `lhsLayer` (same block),
// and, if so, identify a merge `mergeOp` candidate in `lhsLayer`.
// This `mergeOp` later on, will be used in `mergePPR` function.
std::pair<bool, PBCOpInterface> checkCommutationAndFindMerge(PBCOpInterface rhsOp,
                                                             PBCLayer &lhsLayer)
{
    PBCOpInterface mergeOp = nullptr;
    for (auto lhsOp : lhsLayer.getOps()) {
        if (lhsOp->getBlock() != rhsOp->getBlock()) {
            return std::pair(false, nullptr);
        }

        assert(lhsOp != rhsOp && "lshOp and rhsOp should not be equal");
        assert(lhsOp->isBeforeInBlock(rhsOp) && "lhsOp should be before rhsOp");

        // Reaching in-qubit values of `rhsOp` at the program point of `lhsOp`.
        std::vector<Value> rhsOpInQubitsFromLhsOp = getInQubitReachingValuesAt(rhsOp, lhsOp);
        if (llvm::any_of(rhsOpInQubitsFromLhsOp, [](Value qubit) { return qubit == nullptr; })) {
            // Missing reaching value (intervening non-PPR); commutation fails.
            // TODO: Handle non-PPR not directly dominated by `lhsOp`.
            return std::pair(false, nullptr);
        }

        // Normalize to Pauli strings
        auto normalizedOps =
            normalizePPROps(lhsOp, rhsOp, lhsOp.getInQubits(), rhsOpInQubitsFromLhsOp);

        // TODO: Handle PPRotationArbitraryOp properly

        if (!normalizedOps.first.commutes(normalizedOps.second)) {
            return std::pair(false, nullptr);
        }

        // Equal normalized Pauli strings => merge candidate
        auto canMerge = normalizedOps.first == normalizedOps.second;
        if (canMerge) {
            mergeOp = lhsOp;
        }
    }

    return std::pair(true, mergeOp);
}

void moveOpToLayer(PBCOpInterface rhsOp, PBCLayer &rhsLayer, PBCOpInterface mergeOp,
                   PBCLayer &lhsLayer, IRRewriter &writer)
{
    //    lhsLayer   :  rhsLayer
    //    ┌───────┐  :  ┌───────┐
    //   ─┤ lhsOp ├──:──┤ rhsOp ├─
    //    └───────┘  :  └───────┘

    // Move `rhsOp` into `lhsLayer` and remap its operands to values visible at `lhsOp`.

    auto lhsOp = lhsLayer.getOps().back();

    if (mergeOp) {
        lhsOp = mergeOp;
    }

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
    writer.replaceOp(rhsOp, rhsOp->getOperands());
}

// Merge `rhsOp` into `mergeOp` in lhsLayer when equal under normalization.
// To merge this we keep and update the rotation kind of `mergeOp` in lhsLayer,
// then just remove the `rhsOp` from the rhsLayer.
void mergePPR(PBCOpInterface rhsOp, PBCLayer &rhsLayer, PBCOpInterface mergeOp, IRRewriter &writer)
{
    auto mergeOpPprOp = dyn_cast<PPRotationOp>(mergeOp.getOperation());
    assert(mergeOpPprOp != nullptr && "Op is not a PPRotationOp");

    int16_t signedRk = static_cast<int16_t>(mergeOpPprOp.getRotationKind());
    mergeOpPprOp.setRotationKind(static_cast<uint16_t>(signedRk / 2));

    rhsLayer.eraseOp(rhsOp);
    writer.replaceOp(rhsOp, rhsOp->getOperands());
}

struct TLayerReductionPass : impl::TLayerReductionPassBase<TLayerReductionPass> {
    using TLayerReductionPassBase::TLayerReductionPassBase;

    void runOnOperation() final
    {
        MLIRContext *context = &getContext();
        IRRewriter writer(context);
        PBCLayerContext layerContext;
        PBCLayer currentLayer(&layerContext);

        std::vector<PBCLayer> layers;

        // 1) Build initial layers:
        // - try to commute non-Clifford PPR into current layer;
        // - else start a new layer.
        getOperation()->walk([&](PPRotationOp op) {
            if (op.isClifford()) {
                return WalkResult::skip();
            }

            auto [isCommute, _] = checkCommutationAndFindMerge(op, currentLayer);
            if (isCommute) {
                currentLayer.insertToLayer(op);
                return WalkResult::skip();
            }

            layers.emplace_back(std::move(currentLayer));

            // Start a new layer and insert the op
            currentLayer = PBCLayer(&layerContext);
            currentLayer.insertToLayer(op);

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

                for (PBCOpInterface op : currentLayer.getOps()) {
                    auto [isCommute, mergeOp] = checkCommutationAndFindMerge(op, prevLayer);

                    if (isCommute && mergeOp) {
                        mergePPR(op, currentLayer, mergeOp, writer);
                        changed = true;
                    }
                    else if (isCommute) {
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

} // namespace pbc
} // namespace catalyst
