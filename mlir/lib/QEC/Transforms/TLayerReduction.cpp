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
    // layerOp.getInitArgs() is same as layerOp.getOperands()
    // layerOp.getBody()->getArguments() are the arguments that used in body
    // layerOp.getOperands() are from outside
    mlir::IRMapping mapper;
    mapper.map(layerOp.getBody()->getArguments(), layerOp.getOperands());

    return mapper;
}

bool commute(QECOpInterface rhsOp, QECLayer &rhsLayer, QECLayer &lhsLayer)
{

    auto rhsLayerOp = rhsLayer.layerOp;
    auto lhsLayerOp = lhsLayer.layerOp;

    auto rhsLayerOperandMapper = getArgMapper(rhsLayerOp);
    auto lhsLayerOperandMapper = getArgMapper(lhsLayerOp);

    auto lhsOperands = rhsLayer.operands;
    auto rhsOperands = lhsLayer.operands;

    IRMapping lhsMapper;
    lhsMapper.map(lhsLayerOp.getResults(), lhsLayerOp->getOperands());

    std::vector<Value> rhsEntryOperands;
    // Map RHS op from local region to the outside LHS operands
    IRMapping rhsMapper;
    for (auto rhsRegionArg : rhsLayer.getEntryQubitsFrom(rhsOp)) {
        // rhsRegionArg.dump();
        auto rhsInOperand = rhsLayerOperandMapper.lookupOrNull(rhsRegionArg);
        assert(rhsInOperand && "Current operands's qubit should be found in the layer");

        // It is fine if the qubit is not used in the lhs layer,
        // so rhsMapper value can be nullptr
        auto entryRhsVal = lhsMapper.lookupOrNull(rhsInOperand);
        rhsMapper.map(rhsRegionArg, entryRhsVal);
        rhsEntryOperands.emplace_back(entryRhsVal);
    };

    for (auto lhsOp : lhsLayer.ops) {
        std::vector<Value> lhsEntryOperands;

        for (auto lhsRegionArg : lhsLayer.getEntryQubitsFrom(lhsOp)) {
            auto lhsInOperand = lhsLayerOperandMapper.lookupOrDefault(lhsRegionArg);
            lhsEntryOperands.emplace_back(lhsInOperand);
        }

        llvm::SetVector<Value> qubits;
        qubits.insert(rhsEntryOperands.begin(), rhsEntryOperands.end());
        qubits.insert(lhsEntryOperands.begin(), lhsEntryOperands.end());

        PauliWord lhsPauliWord = expandPauliWord(qubits, lhsEntryOperands, lhsOp);
        PauliWord rhsPauliWord = expandPauliWord(qubits, rhsEntryOperands, rhsOp);

        PauliStringWrapper lhsPSWrapper = PauliStringWrapper::from_pauli_word(lhsPauliWord);
        PauliStringWrapper rhsPSWrapper = PauliStringWrapper::from_pauli_word(rhsPauliWord);

        assert(rhsOp != lhsOp && "rhsOp should not same lhsOp");
        if (!rhsPSWrapper.commutes(lhsPSWrapper)) {
            return false;
        }
    }
    return true;
}

// from the rhs-layer to lhs-layer
void moveOpToLayer(QECOpInterface rhsOp, QECLayer &rhsLayer, QECLayer &lhsLayer, IRRewriter &writer)
{

    auto lhsLayerOp = lhsLayer.layerOp;
    auto yieldOp = lhsLayerOp.getBody()->getTerminator();

    auto cloneRhsOp = rhsOp->clone();
    writer.setInsertionPoint(lhsLayerOp.getBody()->getTerminator());
    writer.insert(cloneRhsOp);
    cloneRhsOp->setOperands(yieldOp->getOperands());
    yieldOp->setOperands(cloneRhsOp->getResults());

    // writer.replaceOp(rhsOp, rhsOp->getPrevNode());
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
            // for each layer i do:
            getOperation()->walk([&](LayerOp currentLayerOp) {
                QECLayer &currentLayer = QECLayer::build(&layerContext, currentLayerOp);

                Operation *nextNode = currentLayerOp->getNextNode();
                if (LayerOp nextLayerOp = llvm::dyn_cast_or_null<LayerOp>(nextNode)) {
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
