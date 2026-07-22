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

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "PBC/Transforms/Patterns.h"
#include "Quantum/IR/QuantumOps.h"

using namespace llvm;
using namespace mlir;
using namespace catalyst;
using namespace catalyst::pbc;

namespace catalyst {
namespace pbc {

#define GEN_PASS_DECL_PPRTOPPMPASS
#define GEN_PASS_DEF_PPRTOPPMPASS
#include "PBC/Transforms/Passes.h.inc"

using namespace catalyst::quantum;

namespace {

/// Move ``quantum.dealloc_qb`` operations to after their last use in each function.
/// This ensures magic-state qubits remain mapped when ``ppr-to-ppm`` inserts late PPM users.
void sinkQuantumDeallocs(ModuleOp module)
{
    module.walk([&](func::FuncOp func) {
        SmallVector<DeallocQubitOp> deallocOps;
        func.walk([&](DeallocQubitOp deallocOp) { deallocOps.push_back(deallocOp); });

        for (DeallocQubitOp deallocOp : deallocOps) {
            mlir::Value qubit = deallocOp.getQubit();
            Operation *lastUser = nullptr;
            func.walk([&](Operation *op) {
                if (op == deallocOp) {
                    return;
                }
                for (mlir::Value operand : op->getOperands()) {
                    if (operand == qubit) {
                        lastUser = op;
                    }
                }
            });

            if (lastUser && lastUser != deallocOp->getPrevNode()) {
                deallocOp->moveAfter(lastUser);
            }
        }
    });
}

} // namespace

struct PPRToPPMPass : public impl::PPRToPPMPassBase<PPRToPPMPass> {
    using PPRToPPMPassBase::PPRToPPMPassBase;

    void runOnOperation() final
    {
        auto ctx = &getContext();
        auto module = cast<ModuleOp>(getOperation());

        RewritePatternSet non_clifford_patterns(ctx);
        populateDecomposeNonCliffordPPRPatterns(non_clifford_patterns, decomposeMethod,
                                                avoidYMeasure);

        if (failed(applyPatternsGreedily(module, std::move(non_clifford_patterns)))) {
            return signalPassFailure();
        }

        // Decompose Clifford PPRs into PPMs
        RewritePatternSet clifford_patterns(ctx);
        populateDecomposeCliffordPPRPatterns(clifford_patterns, avoidYMeasure);

        if (failed(applyPatternsGreedily(module, std::move(clifford_patterns)))) {
            return signalPassFailure();
        }

        sinkQuantumDeallocs(module);
    }
};

} // namespace pbc
} // namespace catalyst
