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

#include "llvm/ADT/StringSet.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "Quantum/IR/QuantumOps.h"

using namespace mlir;
using namespace catalyst::quantum;

namespace {

static const mlir::StringSet<> hermitianOps = {"Hadamard", "PauliX", "PauliY", "PauliZ", "CNOT",
                                               "CY",       "CZ",     "SWAP",   "Toffoli"};
static const mlir::StringSet<> rotationsOps = {"RX",  "RY",  "RZ",  "PhaseShift",
                                               "CRX", "CRY", "CRZ", "ControlledPhaseShift"};

// Canonicalize Adjoint on quantum.custom gates after adjoint-lowering.
// For Hermitian gates, the adjoint flag is set to false.
// For rotations, the parameters are negated.
struct CustomOpAdjointCanonicalizePattern : public OpRewritePattern<CustomOp> {
    using OpRewritePattern<CustomOp>::OpRewritePattern;

    // Canonicalize Adjoint on quantum.custom gates
    // moves LogicalResult CustomOp::canonicalize from QuantumOps into its own pass.
    LogicalResult matchAndRewrite(CustomOp op, PatternRewriter &rewriter) const override {
        if (op.getAdjoint()) {
            auto name = op.getGateName();
            if (hermitianOps.contains(name)) {
                op.setAdjoint(false);
                return success();
            } else if (rotationsOps.contains(name)) {
                auto params = op.getParams();
                SmallVector<Value> paramsNeg;
                for (auto param : params) {
                    auto paramNeg = mlir::arith::NegFOp::create(rewriter, op.getLoc(), param);
                    paramsNeg.push_back(paramNeg);
                }

                rewriter.replaceOpWithNewOp<CustomOp>(
                    op, op.getOutQubits().getTypes(), op.getOutCtrlQubits().getTypes(), paramsNeg,
                    op.getInQubits(), name, false, op.getInCtrlQubits(), op.getInCtrlValues());

                return success();
            }
            return failure();
        }
        return failure();
    }
};

} // namespace

namespace catalyst {
namespace quantum {

#define GEN_PASS_DEF_ADJOINTLOWERINGCANONICALIZATION
#include "Quantum/Transforms/Passes.h.inc"

// Populate the patterns for the AdjointLoweringCanonicalization pass.
// Allows reference in ions_decompositions.cpp and merge_rotation.cpp
void populateAdjointLoweringCanonicalizationPatterns(RewritePatternSet &patterns) {
    patterns.add<CustomOpAdjointCanonicalizePattern>(patterns.getContext());
}

struct AdjointLoweringCanonicalization
    : impl::AdjointLoweringCanonicalizationBase<AdjointLoweringCanonicalization> {
    using AdjointLoweringCanonicalizationBase::AdjointLoweringCanonicalizationBase;

    void runOnOperation() final {
        RewritePatternSet patterns(&getContext());
        populateAdjointLoweringCanonicalizationPatterns(patterns);

        if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
            return signalPassFailure();
        }
    }
};

} // namespace quantum
} // namespace catalyst
