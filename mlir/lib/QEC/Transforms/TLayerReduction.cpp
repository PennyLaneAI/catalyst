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

#include <algorithm>
#include <cstddef>
#include <iterator>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SetVector.h>
#include <map>
#include <mlir/Support/WalkResult.h>
#define DEBUG_TYPE "t-layer-reduction"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"

#include "QEC/IR/QECDialect.h"
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

mlir::IRMapping getArgMapper(LayerOp layerOp)
{
    // layerOp.getBody()->getArguments() are the arguments that used in body
    // layerOp.getOperands() are from outside

    /// Map layer block arguments (body arguments) -> LayerOp operands (outer SSA values).
    /// Note: getInitArgs() equals getOperands() for this op.
    mlir::IRMapping mapper;
    mapper.map(layerOp.getBody()->getArguments(), layerOp.getOperands());
    return mapper;
}

PauliStringWrapper buildPauliStringWrapper(llvm::SetVector<Value> qubits,
                                           std::vector<Value> entryOperands, QECOpInterface op)
{
    PauliWord pauliWord = expandPauliWord(qubits, entryOperands, op);
    return PauliStringWrapper::from_pauli_word(pauliWord);
}

bool commute(QECOpInterface op, QECLayer &rhsLayer, QECLayer &lhsLayer)
{
    // Resolve in-region values to outer SSA values via the layer interface:
    // op-local values -> layer block arguments (entry qubits) -> LayerOp operands.
    // Example:
    //   rhsLayer = layerOp(A=q0, B=q1){
    //       K:2 = PPR["X","Y"](8) A, B
    //       M:2 = PPR["Z","Z"](8) K#0, K#1   <-
    //   }
    //
    // Map K#0 -> A (block arg), then A -> q0 (LayerOp operand).

    /// Build a mapper from layer block arguments to LayerOp operands.
    auto srcBlockArgToOperand = getArgMapper(rhsLayer.layerOp);
    auto dstBlockArgToOperand = getArgMapper(lhsLayer.layerOp);

    IRMapping dstResultMap;
    // assert(lhsLayer.layerOp.getResults().size() == lhsLayer.layerOp->getOperands().size() &&
    //        "lhsLayer results and operands should have the same size");
    dstResultMap.map(lhsLayer.layerOp.getResults(), lhsLayer.layerOp->getOperands());

    std::vector<Value> opEntriesDst;
    /// Map the op’s entry qubits (src block arguments) to dst LayerOp operands.
    IRMapping opArgMap;
    for (auto opArg : rhsLayer.getEntryQubitsFrom(op)) {
        auto srcOperand = srcBlockArgToOperand.lookupOrNull(opArg);
        assert(srcOperand && "Current operand's qubit should be found in the layer");

        // It is fine if the qubit is not used in the dst-layer,
        // so opArgMap value can be nullptr
        auto mappedOpDst = dstResultMap.lookupOrNull(srcOperand);
        opArgMap.map(opArg, mappedOpDst);
        opEntriesDst.emplace_back(mappedOpDst);
    };

    for (auto dstOp : lhsLayer.ops) {
        std::vector<Value> dstEntries;

        for (auto dstArg : lhsLayer.getEntryQubitsFrom(dstOp)) {
            auto dstOperand = dstBlockArgToOperand.lookupOrDefault(dstArg);
            dstEntries.emplace_back(dstOperand);
        }

        llvm::SetVector<Value> qubits;
        qubits.insert(opEntriesDst.begin(), opEntriesDst.end());
        qubits.insert(dstEntries.begin(), dstEntries.end());

        PauliStringWrapper dstPauli = buildPauliStringWrapper(qubits, dstEntries, dstOp);
        PauliStringWrapper opPauli = buildPauliStringWrapper(qubits, opEntriesDst, op);

        assert(op != dstOp && "op should not be the same as dstOp");

        if (!opPauli.commutes(dstPauli)) {
            return false;
        }
    }

    return true;
}

void moveOpToLayer(QECOpInterface rhsOp, QECLayer &rhsLayer, QECLayer &lhsLayer, IRRewriter &writer)
{
    auto lhsLayerOp = lhsLayer.layerOp;

    // set the operands
    auto rhsRegion = rhsLayer.getEntryQubitsFrom(rhsOp);
    auto rhsRegionToEntry = getArgMapper(rhsLayer.layerOp);
    std::vector<Value> rhsOperands;
    std::transform(rhsRegion.begin(), rhsRegion.end(), std::back_inserter(rhsOperands),
                   [&](Value v) { return rhsRegionToEntry.lookupOrNull(v); });

    auto lhsOperandSize =
        std::max(rhsOperands.size(), static_cast<size_t>(lhsLayerOp->getNumResults()));

    std::vector<Value> lhsOperandForRhsOp;
    lhsOperandForRhsOp.reserve(lhsOperandSize);

    for (auto [idx, rhsOperand] : llvm::enumerate(rhsOperands)) {
        auto lhsOperands = lhsLayerOp->getOperands();
        auto lhsResults = lhsLayerOp->getResults();

        auto it = std::find(lhsResults.begin(), lhsResults.end(), rhsOperand);
        if (it != lhsResults.end()) {
            auto position = std::distance(lhsResults.begin(), it);
            lhsOperandForRhsOp.emplace_back(lhsOperands[position]);
        }
        else {
            lhsOperandForRhsOp.emplace_back(rhsOperand);
        }
    }

    // For each desired outer operand, use the existing block argument if present;
    // otherwise append a new operand and matching block argument.
    auto lhsArgToOperand = getArgMapper(lhsLayerOp);
    std::vector<Value> rhsOpOperands;
    rhsOpOperands.reserve(lhsOperandSize);

    for (auto lhsOperand : lhsOperandForRhsOp) {
        Value entryArg = nullptr;
        for (BlockArgument ba : lhsLayerOp.getBody()->getArguments()) {
            Value mapped = lhsArgToOperand.lookupOrNull(ba);
            if (mapped && mapped == lhsOperand) {
                entryArg = ba;
                break;
            }
        }

        if (entryArg) {
            rhsOpOperands.emplace_back(entryArg);
        }
        else {
            // Add new operand to the LayerOp and the corresponding block argument.
            lhsLayerOp->insertOperands(lhsLayerOp->getNumOperands(), lhsOperand);
            BlockArgument newEntry =
                lhsLayerOp.getBody()->addArgument(lhsOperand.getType(), lhsOperand.getLoc());
            rhsOpOperands.emplace_back(newEntry);
        }
    }

    auto cloneRhsOp = rhsOp->clone();
    writer.setInsertionPoint(lhsLayerOp.getBody()->getTerminator());
    writer.insert(cloneRhsOp);
    // Maintain operand segment sizes for ops with AttrSizedOperandSegments
    if (auto ppr = llvm::dyn_cast<PPRotationOp>(cloneRhsOp)) {
        ppr.getInQubitsMutable().assign(rhsOpOperands);
    }
    else {
        cloneRhsOp->setOperands(rhsOpOperands);
    }
    auto yieldOp = lhsLayerOp.getBody()->getTerminator();
    yieldOp->setOperands(cloneRhsOp->getResults());

    // Rewire the next (rhs) layer to consume the results of the lhs layer.
    // For each entry block argument used by the moved op in rhs layer, set the
    // corresponding operand of the rhs LayerOp to the matching result of lhs LayerOp.
    // This ensures %argX in the next layer header is sourced from %0#X rather than
    // the original outer SSA value when appropriate.
    if (!rhsLayer.ops.empty()) {
        auto rhsRegion = rhsLayer.getEntryQubitsFrom(rhsOp);
        for (auto [i, entry] : llvm::enumerate(rhsRegion)) {
            if (auto ba = llvm::dyn_cast<BlockArgument>(entry)) {
                unsigned argIdx = ba.getArgNumber();
                if (i < lhsLayerOp->getNumResults()) {
                    rhsLayer.layerOp->setOperand(argIdx, lhsLayerOp->getResult(i));
                }
            }
        }
    }

    writer.replaceAllUsesWith(rhsOp->getResults(), rhsOp->getOperands());

    lhsLayer.insert(llvm::cast<QECOpInterface>(cloneRhsOp));
    std::erase(rhsLayer.ops, rhsOp);
    writer.eraseOp(rhsOp);
}

struct TLayerReductionPass : impl::TLayerReductionPassBase<TLayerReductionPass> {
    using TLayerReductionPassBase::TLayerReductionPassBase;

    void runOnOperation() final
    {
        MLIRContext *context = &getContext();
        IRRewriter writer(context);
        QECLayerContext layerContext;

        bool isChange = true;

        while (isChange) {
            isChange = false;

            getOperation()->walk([&](LayerOp currentLayerOp) {
                if (currentLayerOp->getNumOperands() != currentLayerOp->getNumResults())
                    WalkResult::skip();

                QECLayer &currentLayer = QECLayer::build(&layerContext, currentLayerOp);

                Operation *nextNode = currentLayerOp->getNextNode();

                if (LayerOp nextLayerOp = llvm::dyn_cast_or_null<LayerOp>(nextNode)) {
                    if (nextLayerOp->getNumOperands() != currentLayerOp.getNumResults())
                        WalkResult::skip();

                    QECLayer &nextLayer = QECLayer::build(&layerContext, nextLayerOp);
                    for (QECOpInterface op : nextLayer.ops) {
                        if (commute(op, nextLayer, currentLayer)) {
                            moveOpToLayer(op, nextLayer, currentLayer, writer);
                            isChange = true;
                            break;
                        }
                    }
                }

                if (currentLayer.empty()) {
                    writer.replaceAllUsesWith(currentLayerOp.getResults(),
                                              currentLayerOp->getOperands());
                    writer.eraseOp(currentLayerOp);
                    isChange = true;
                }
            });
        }
        layerContext.clear();
    };
};
} // namespace qec

/// Create a pass for lowering operations in the `QECDialect`.
std::unique_ptr<Pass> createTLayerReductionPass()
{
    return std::make_unique<qec::TLayerReductionPass>();
}

} // namespace catalyst
