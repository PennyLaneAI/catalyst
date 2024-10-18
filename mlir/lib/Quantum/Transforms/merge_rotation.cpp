// Copyright 2023 Xanadu Quantum Technologies Inc.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#define DEBUG_TYPE "merge-rotation"

#include <chrono>

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

#define GEN_PASS_DEF_MERGEROTATIONSPASS
#define GEN_PASS_DECL_MERGEROTATIONSPASS
#include "Quantum/Transforms/Passes.h.inc"

struct MergeRotationsPass : impl::MergeRotationsPassBase<MergeRotationsPass> {
    using MergeRotationsPassBase::MergeRotationsPassBase;

    void runOnOperation() final
    {
        LLVM_DEBUG(dbgs() << "merge rotation pass"
                          << "\n");
        auto start = std::chrono::high_resolution_clock::now();

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

        RewritePatternSet patterns(&getContext());
        populateMergeRotationsPatterns(patterns);

        if (failed(applyPatternsAndFoldGreedily(targetfunc, std::move(patterns)))) {
            return signalPassFailure();
        }

        auto stop = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
        llvm::errs() << "merge rotation pass runtime: " << duration.count() << " microseconds\n";
    }
};

} // namespace quantum

std::unique_ptr<Pass> createMergeRotationsPass()
{
    return std::make_unique<quantum::MergeRotationsPass>();
}

} // namespace catalyst
