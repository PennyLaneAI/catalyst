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

#define DEBUG_TYPE "convert-to-qec"

#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"

#include "QEC/IR/QECDialect.h"
#include "QEC/Transforms/Patterns.h"
#include "Quantum/IR/QuantumOps.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

using namespace mlir;
using namespace catalyst::quantum;
using namespace catalyst::qec;

namespace catalyst {
namespace qec {

#define GEN_PASS_DEF_LOWERINGTOQECPASS
#define GEN_PASS_DECL_LOWERINGTOQECPASS
#include "QEC/Transforms/Passes.h.inc"

struct LoweringToQECPass : impl::LoweringToQECPassBase<LoweringToQECPass> {
    using LoweringToQECPassBase::LoweringToQECPassBase;

    void runOnOperation() final
    {
        ConversionTarget target(getContext());

        target.addLegalDialect<arith::ArithDialect, func::FuncDialect, qec::QECDialect>();

        target.addIllegalOp<quantum::CustomOp>();
        target.addIllegalOp<quantum::MeasureOp>();

        RewritePatternSet patterns(&getContext());
        populateQECLoweringPatterns(patterns);

        if (failed(applyPartialConversion(getOperation(), target, std::move(patterns)))) {
            return signalPassFailure();
        }
    }
};

} // namespace qec

/// Create a pass for lowering operations in the `QECDialect`.
std::unique_ptr<mlir::Pass> createLowerToQECPass()
{
    return std::make_unique<qec::LoweringToQECPass>();
}

} // namespace catalyst
