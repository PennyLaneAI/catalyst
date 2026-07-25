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

#define DEBUG_TYPE "ppr-to-ppm"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Complex/IR/Complex.h"
#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/DialectConversion.h"

#include "Catalyst/IR/CatalystDialect.h"
#include "PBC/IR/PBCDialect.h"
#include "PBC/IR/PBCOps.h"
#include "PBC/Transforms/Patterns.h"
#include "Quantum/IR/QuantumOps.h"
#include "Quantum/Transforms/Passes.h"

using namespace llvm;
using namespace mlir;
using namespace catalyst;
using namespace catalyst::pbc;

namespace catalyst {
namespace pbc {

#define GEN_PASS_DECL_PPRTOPPMPASS
#define GEN_PASS_DEF_PPRTOPPMPASS
#include "PBC/Transforms/Passes.h.inc"

struct PPRToPPMPass : public impl::PPRToPPMPassBase<PPRToPPMPass> {
    using PPRToPPMPassBase::PPRToPPMPassBase;

    void runOnOperation() final
    {
        auto ctx = &getContext();
        auto module = getOperation();

        OpPassManager pm("builtin.module");
        pm.addPass(quantum::createAdjointLoweringPass());
        if (failed(runPipeline(pm, module))) {
            return signalPassFailure();
        }

        ConversionTarget target(*ctx);
        target.addLegalDialect<pbc::PBCDialect>();
        target.addLegalDialect<mlir::arith::ArithDialect>();
        target.addLegalDialect<mlir::scf::SCFDialect>();
        target.addDynamicallyLegalOp<pbc::PPRotationOp>([](pbc::PPRotationOp op) {
            return !op.hasPiOverFourRotation() && (!op.isNonClifford() || op.getCondition());
        });
        target.addDynamicallyLegalDialect<quantum::QuantumDialect>([](Operation *op) {
            return isa<quantum::AllocOp, quantum::AllocQubitOp, quantum::DeallocOp,
                       quantum::DeallocQubitOp, quantum::ExtractOp, quantum::InsertOp,
                       quantum::GlobalPhaseOp>(op);
        });

        RewritePatternSet patterns(ctx);
        populateDecomposeNonCliffordPPRPatterns(patterns, decomposeMethod, avoidYMeasure);
        populateDecomposeCliffordPPRPatterns(patterns, avoidYMeasure);

        if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
            return signalPassFailure();
        }
    }
};

} // namespace pbc
} // namespace catalyst
