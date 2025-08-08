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

#include "QEC/IR/QECOpInterfaces.h"
#include <llvm/Support/Casting.h>
#define DEBUG_TYPE "t-layer-reduction"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"

#include "QEC/IR/QECDialect.h"
#include "QEC/Utils/PauliStringWrapper.h"
#include "QEC/Utils/QECLayer.h"

using namespace mlir;
using namespace catalyst::qec;

namespace catalyst {
namespace qec {

#define GEN_PASS_DEF_TLAYERREDUCTIONPASS
#define GEN_PASS_DECL_TLAYERREDUCTIONPASS
#include "QEC/Transforms/Passes.h.inc"

bool commute(QECOpInterface op, QECLayer &fromLayer, QECLayer &toLayer)
{
    // TODO: Implement this
    return true;
}

void moveOpToLayer(QECOpInterface op, QECLayer &fromLayer, QECLayer &toLayer)
{
    // fromLayer.insert(op);
    // toLayer.insert(op);
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
            // for each layer i do:
            getOperation()->walk([&](LayerOp currentLayerOp) {
                QECLayer currentLayer = QECLayer::build(&layerContext, currentLayerOp);

                auto nextNode = currentLayerOp->getNextNode();
                if (auto nextLayerOp = llvm::dyn_cast_or_null<LayerOp>(nextNode)) {
                    auto nextLayer = QECLayer::build(&layerContext, nextLayerOp);
                    for (auto op : currentLayer.ops) {
                        if (commute(op, currentLayer, nextLayer)) {
                            moveOpToLayer(op, currentLayer, nextLayer);
                            isChange = true;
                            llvm::outs() << "Commuting\n";
                        }
                    }
                }
            });

            isChange = false;
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
