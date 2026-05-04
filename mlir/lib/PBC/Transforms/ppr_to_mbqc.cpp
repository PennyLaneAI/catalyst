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

#define DEBUG_TYPE "ppr-to-mbqc"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"  // for PassManager
#include "mlir/Transforms/Passes.h" // for createCSEPass

#include "PBC/IR/PBCOps.h"
#include "PBC/Transforms/Patterns.h"
#include "Quantum/IR/QuantumDialect.h"

using namespace llvm;
using namespace mlir;
using namespace catalyst;
using namespace catalyst::pbc;

namespace catalyst {
namespace pbc {

#define GEN_PASS_DEF_PPRTOMBQCPASS
#define GEN_PASS_DECL_PPRTOMBQCPASS
#include "PBC/Transforms/Passes.h.inc"

struct PPRToMBQCPass : public impl::PPRToMBQCPassBase<PPRToMBQCPass> {
    using PPRToMBQCPassBase::PPRToMBQCPassBase;

    void runOnOperation() final
    {
        auto ctx = &getContext();
        ConversionTarget target(*ctx);

        target.addIllegalOp<pbc::PPRotationOp>();
        target.addIllegalOp<pbc::PPMeasurementOp>();

        // The Dialects that we want to convert to
        target.addLegalDialect<catalyst::quantum::QuantumDialect>();
        target.addLegalDialect<mlir::arith::ArithDialect>();

        RewritePatternSet patterns(ctx);
        populatePPRToMBQCPatterns(patterns);

        if (failed(applyPartialConversion(getOperation(), target, std::move(patterns)))) {
            return signalPassFailure();
        }

        // Run CSE to deduplicate constants and trivial common constants
        PassManager pm(ctx);
        pm.addPass(createCSEPass());
        if (failed(pm.run(getOperation()))) {
            return signalPassFailure();
        }
    }
};

} // namespace pbc
} // namespace catalyst
