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

#define DEBUG_TYPE "ppm-compilation"

#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "QEC/IR/QECDialect.h"
#include "QEC/Transforms/Patterns.h"
#include "Quantum/IR/QuantumOps.h"

using namespace llvm;
using namespace mlir;
using namespace catalyst;
using namespace catalyst::qec;

namespace catalyst {
namespace qec {

#define GEN_PASS_DEF_PPMCOMPILATIONPASS
#define GEN_PASS_DECL_PPMCOMPILATIONPASS
#include "QEC/Transforms/Passes.h.inc"

struct PPMCompilationPass : public impl::PPMCompilationPassBase<PPMCompilationPass> {
    using PPMCompilationPassBase::PPMCompilationPassBase;

    void runOnOperation() final
    {
        auto ctx = &getContext();
        auto module = getOperation();

        // Phase 1: Convert Clifford+T to PPR representation
        {
            ConversionTarget target(*ctx);
            target.addIllegalDialect<quantum::QuantumDialect>();
            target.addLegalOp<quantum::InitializeOp, quantum::FinalizeOp>();
            target.addLegalOp<quantum::DeviceInitOp, quantum::DeviceReleaseOp>();
            target.addLegalOp<quantum::AllocOp, quantum::DeallocOp>();
            target.addLegalOp<quantum::InsertOp, quantum::ExtractOp>();
            target.addLegalDialect<qec::QECDialect>();

            RewritePatternSet patterns(ctx);
            populateCliffordTToPPRPatterns(patterns);

            if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
                return signalPassFailure();
            }
        }

        // Phase 2: Commute Clifford gates past T gates using PPR representation
        {
            RewritePatternSet patterns(ctx);
            populateCommutePPRPatterns(patterns, maxPauliSize);

            if (failed(applyPatternsGreedily(module, std::move(patterns)))) {
                return signalPassFailure();
            }
        }

        // Phase 3: Absorb Clifford gates into measurement operations
        {
            RewritePatternSet patterns(ctx);
            populateMergePPRIntoPPMPatterns(patterns, maxPauliSize);

            if (failed(applyPatternsGreedily(module, std::move(patterns)))) {
                return signalPassFailure();
            }
        }

        // Phase 4: Decompose non-Clifford PPRs into PPMs
        {
            RewritePatternSet patterns(ctx);
            populateDecomposeNonCliffordPPRPatterns(patterns, decomposeMethod, avoidYMeasure);

            if (failed(applyPatternsGreedily(module, std::move(patterns)))) {
                return signalPassFailure();
            }
        }

        // Phase 5: Decompose Clifford PPRs into PPMs
        {
            RewritePatternSet patterns(ctx);
            populateDecomposeCliffordPPRPatterns(patterns, avoidYMeasure);

            if (failed(applyPatternsGreedily(module, std::move(patterns)))) {
                return signalPassFailure();
            }
        }
    }
};

} // namespace qec

std::unique_ptr<Pass> createPPMCompilationPass() { return std::make_unique<PPMCompilationPass>(); }

} // namespace catalyst
