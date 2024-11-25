// Copyright 2024 Xanadu Quantum Technologies Inc.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#define DEBUG_TYPE "ions-decomposition"

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

#define GEN_PASS_DEF_IONSDECOMPOSITIONPASS
#define GEN_PASS_DECL_IONSDECOMPOSITIONPASS
#include "Quantum/Transforms/Passes.h.inc"

struct IonsDecompositionPass : impl::IonsDecompositionPassBase<IonsDecompositionPass> {
    using IonsDecompositionPassBase::IonsDecompositionPassBase;

    void runOnOperation() final
    {
        LLVM_DEBUG(dbgs() << "ions decomposition pass"
                          << "\n");

        Operation *module = getOperation();

        Operation *targetfunc;

        WalkResult result = module->walk([&](func::FuncOp op) {
            StringRef funcName = op.getSymName();

            if (funcName != FuncNameOpt) {
                // not the function to run the pass on, visit the next function
                return WalkResult::advance();
            }
            targetfunc = op;
            return WalkResult::interrupt();
        });

        if (!result.wasInterrupted()) {
            // Never met a target function
            // Do nothing and exit!
            return;
        }

        RewritePatternSet patternsCanonicalization(&getContext());
        catalyst::quantum::CustomOp::getCanonicalizationPatterns(patternsCanonicalization,
                                                                 &getContext());
        if (failed(applyPatternsAndFoldGreedily(module, std::move(patternsCanonicalization)))) {
            return signalPassFailure();
        }
        RewritePatternSet patterns(&getContext());
        populateIonsDecompositionPatterns(patterns);
        if (failed(applyPatternsAndFoldGreedily(module, std::move(patterns)))) {
            return signalPassFailure();
        }
    }
};

} // namespace quantum

std::unique_ptr<Pass> createIonsDecompositionPass()
{
    return std::make_unique<quantum::IonsDecompositionPass>();
}

} // namespace catalyst
