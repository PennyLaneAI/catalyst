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

#define DEBUG_TYPE "graph-decomposition"

#include "Catalyst/IR/CatalystDialect.h"
#include "Quantum/IR/QuantumOps.h"
#include "Quantum/Transforms/Patterns.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/Debug.h"

using namespace llvm;
using namespace mlir;
using namespace catalyst::quantum;

namespace catalyst {
namespace quantum {

#define GEN_PASS_DEF_GRAPHDECOMPOSITIONPASS
#define GEN_PASS_DECL_GRAPHDECOMPOSITIONPASS
#include "Quantum/Transforms/Passes.h.inc"

struct GraphDecompositionPass : impl::GraphDecompositionPassBase<GraphDecompositionPass> {
    using GraphDecompositionPassBase::GraphDecompositionPassBase;

    void runOnOperation() final
    {
        LLVM_DEBUG(dbgs() << "graph decomposition pass"
                          << "\n");

        Operation *module = getOperation();

        RewritePatternSet patternsCanonicalization(&getContext());
        catalyst::quantum::CustomOp::getCanonicalizationPatterns(patternsCanonicalization,
                                                                 &getContext());
        if (failed(applyPatternsGreedily(module, std::move(patternsCanonicalization)))) {
            return signalPassFailure();
        }
        RewritePatternSet patterns(&getContext());
        populateGraphDecompositionPatterns(patterns);
        if (failed(applyPatternsGreedily(module, std::move(patterns)))) {
            return signalPassFailure();
        }
    }
};

} // namespace quantum

std::unique_ptr<Pass> createGraphDecompositionPass()
{
    return std::make_unique<quantum::GraphDecompositionPass>();
}

} // namespace catalyst
