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

#define DEBUG_TYPE "remove-chained-self-inverse"

#include <memory>
#include <vector>

#include "llvm/Support/Debug.h"
#include "llvm/Support/Errc.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "Catalyst/IR/CatalystDialect.h"
#include "Quantum/IR/QuantumOps.h"
#include "Quantum/Transforms/Patterns.h"

using namespace llvm;
using namespace mlir;
using namespace catalyst::quantum;

namespace catalyst {
namespace quantum {

#define GEN_PASS_DEF_REMOVECHAINEDSELFINVERSEPASS
#define GEN_PASS_DECL_REMOVECHAINEDSELFINVERSEPASS
#include "Quantum/Transforms/Passes.h.inc"

struct RemoveChainedSelfInversePass
    : impl::RemoveChainedSelfInversePassBase<RemoveChainedSelfInversePass> {
    using RemoveChainedSelfInversePassBase::RemoveChainedSelfInversePassBase;

    void runOnOperation() final
    {
        LLVM_DEBUG(dbgs() << "remove chained self inverse pass"
                          << "\n");

        // Run cse pass before running remove-chained-self-inverse,
        // to aid identifiying equivalent SSA values when verifying
        // the gates have the same params
        MLIRContext *ctx = &getContext();
        auto pm = PassManager::on<ModuleOp>(ctx);
        pm.addPass(mlir::createCSEPass());

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
        populateSelfInversePatterns(patterns);

        if (failed(applyPatternsAndFoldGreedily(targetfunc, std::move(patterns)))) {
            return signalPassFailure();
        }
    }
};

} // namespace quantum

std::unique_ptr<Pass> createRemoveChainedSelfInversePass()
{
    return std::make_unique<quantum::RemoveChainedSelfInversePass>();
}

} // namespace catalyst
