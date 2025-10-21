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

#define DEBUG_TYPE "rs-decomposition"

#include <memory>
#include <vector>

#include "llvm/Support/Debug.h"
#include "llvm/Support/Errc.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"

#include "Catalyst/IR/CatalystDialect.h"
#include "Quantum/IR/QuantumOps.h"
#include "Quantum/Transforms/Patterns.h"

using namespace llvm;
using namespace mlir;
using namespace catalyst::quantum;

namespace catalyst {
namespace quantum {

#define GEN_PASS_DEF_RSDECOMPOSITIONPASS
#define GEN_PASS_DECL_RSDECOMPOSITIONPASS
#include "Quantum/Transforms/Passes.h.inc"

struct RZPattern : public mlir::OpRewritePattern<CustomOp> {
    using mlir::OpRewritePattern<CustomOp>::OpRewritePattern;

    mlir::LogicalResult matchAndRewrite(CustomOp op, mlir::PatternRewriter &rewriter) const override
    {
        LLVM_DEBUG(dbgs() << "matching the following operation:\n" << op << "\n");
        return failure();
    }
};

void populateRZPattern(RewritePatternSet &patterns)
{
    patterns.add<RZPattern>(patterns.getContext(), 1);
}

struct RSDecompositionPass : impl::RSDecompositionPassBase<RSDecompositionPass> {
    using RSDecompositionPassBase::RSDecompositionPassBase;

    void runOnOperation() final
    {
        LLVM_DEBUG(dbgs() << "perform RS decomposition" << "\n");

        Operation *module = getOperation();

        RewritePatternSet patterns(&getContext());
        populateRZPattern(patterns);

        if (failed(applyPatternsGreedily(module, std::move(patterns)))) {
            return signalPassFailure();
        }
    }
};

} // namespace quantum

std::unique_ptr<Pass> createRSDecompositionPass()
{
    return std::make_unique<quantum::RSDecompositionPass>();
}

} // namespace catalyst
