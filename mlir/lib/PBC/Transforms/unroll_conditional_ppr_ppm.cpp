// Copyright 2026 Xanadu Quantum Technologies Inc.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#define DEBUG_TYPE "unroll-conditional-ppr-ppm"

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "PBC/IR/PBCDialect.h"
#include "PBC/IR/PBCOps.h"
#include "PBC/Transforms/Patterns.h"

using namespace mlir;
using namespace catalyst::pbc;

namespace catalyst {
namespace pbc {

#define GEN_PASS_DECL_UNROLLCONDITIONALPPRPPMPASS
#define GEN_PASS_DEF_UNROLLCONDITIONALPPRPPMPASS
#include "PBC/Transforms/Passes.h.inc"

struct UnrollConditionalPPRPPMPass
    : impl::UnrollConditionalPPRPPMPassBase<UnrollConditionalPPRPPMPass> {
    using UnrollConditionalPPRPPMPassBase::UnrollConditionalPPRPPMPassBase;

    void runOnOperation() final
    {
        auto ctx = &getContext();
        auto module = getOperation();

        ConversionTarget target(*ctx);

        // Convert SelectPPMeasurementOp
        target.addIllegalOp<pbc::SelectPPMeasurementOp>();

        // Conversion target is PBCDialect
        target.addLegalDialect<pbc::PBCDialect>();
        target.addLegalDialect<mlir::scf::SCFDialect>();

        RewritePatternSet patterns(ctx);
        populateUnrollConditionalPPRPPMPatterns(patterns);

        if (failed(applyPatternsGreedily(module, std::move(patterns)))) {
            return signalPassFailure();
        }
    }
};

} // namespace pbc
} // namespace catalyst
