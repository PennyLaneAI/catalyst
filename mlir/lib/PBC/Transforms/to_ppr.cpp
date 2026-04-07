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

#define DEBUG_TYPE "to-ppr"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "Quantum/IR/QuantumOps.h"

#include "PBC/IR/PBCDialect.h"
#include "PBC/IR/PBCOps.h"
#include "PBC/Transforms/Patterns.h"

using namespace mlir;
using namespace catalyst::quantum;
using namespace catalyst::pbc;

namespace catalyst {
namespace pbc {

#define GEN_PASS_DEF_TOPPRPASS
#include "PBC/Transforms/Passes.h.inc"

struct ToPPRPass : impl::ToPPRPassBase<ToPPRPass> {
    using ToPPRPassBase::ToPPRPassBase;

    void runOnOperation() final
    {
        auto ctx = &getContext();
        ConversionTarget target(*ctx);

        // Convert MeasureOp, CustomOp, and PauliRotOp
        target.addIllegalOp<quantum::MeasureOp>();
        target.addIllegalOp<quantum::CustomOp>();

        // Conversion target is PBCDialect
        target.addLegalDialect<pbc::PBCDialect>();
        target.addLegalDialect<mlir::arith::ArithDialect>();
        target.addLegalOp<GlobalPhaseOp>();

        RewritePatternSet patterns(ctx);
        populateToPPRPatterns(patterns);

        if (failed(applyPartialConversion(getOperation(), target, std::move(patterns)))) {
            return signalPassFailure();
        }

        // Add canonicalization pass
        RewritePatternSet canonPatterns(ctx);
        catalyst::pbc::PPRotationOp::getCanonicalizationPatterns(canonPatterns, ctx);
        catalyst::pbc::PPRotationArbitraryOp::getCanonicalizationPatterns(canonPatterns, ctx);
        if (failed(applyPatternsGreedily(getOperation(), std::move(canonPatterns)))) {
            return signalPassFailure();
        }
    }
};

} // namespace pbc
} // namespace catalyst
