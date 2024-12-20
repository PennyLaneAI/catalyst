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

#define DEBUG_TYPE "static-costum"

#include "Catalyst/IR/CatalystDialect.h"
#include "Quantum/IR/QuantumOps.h"
#include "Quantum/Transforms/Patterns.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/Debug.h"

using namespace llvm;
using namespace mlir;
using namespace catalyst::quantum;

namespace catalyst {

namespace quantum {

#define GEN_PASS_DEF_STATICCUSTOMLOWERINGPASS
#define GEN_PASS_DECL_STATICCUSTOMLOWERINGPASS
#include "Quantum/Transforms/Passes.h.inc"

struct StaticCustomLoweringPass : impl::StaticCustomLoweringPassBase<StaticCustomLoweringPass> {
    using StaticCustomLoweringPassBase::StaticCustomLoweringPassBase;
    void runOnOperation() final
    {
        LLVM_DEBUG(dbgs() << "static custom op lowering pass"
                          << "\n");
        auto module = getOperation();
        auto &context = getContext();
        RewritePatternSet patterns(&context);
        ConversionTarget target(context);

        target.addLegalOp<CustomOp>();
        target.addLegalOp<mlir::arith::ConstantOp>();
        target.addLegalOp<GlobalPhaseOp>();
        target.addLegalOp<MultiRZOp>();
        target.addIllegalOp<StaticCustomOp>();

        populateStaticCustomPatterns(patterns);

        if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
            return signalPassFailure();
        }
    }
};

} // namespace quantum

std::unique_ptr<Pass> createStaticCustomLoweringPass()
{
    return std::make_unique<StaticCustomLoweringPass>();
}

} // namespace catalyst
