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

// Canonicalize Adjoint on quantum.custom gates
// For Hermitian gates, the adjoint flag is set to false.
// For rotations, the parameters are negated.
struct CustopOpResolveGateLevelAdjointPattern : public OpRewritePattern<CustomOp> {
    using OpRewritePattern<CustomOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(CustomOp op, PatternRewriter &rewriter) const override {
        if (!op.getAdjoint()) {
            return failure();
        }

        auto name = op.getGateName();
        if (hermitianOps.contains(name)) {
            rewriter.modifyOpInPlace(op, [&] { op.setAdjoint(false); });
            return success();
        } else if (rotationsOps.contains(name)) {
            SmallVector<Value> paramsNeg;
            for (Value param : op.getParams()) {
                auto paramNeg = mlir::arith::NegFOp::create(rewriter, op.getLoc(), param);
                paramsNeg.push_back(paramNeg);
            }

            rewriter.modifyOpInPlace(op, [&] {
                op.getParamsMutable().assign(paramsNeg);
                op.setAdjoint(false);
            });
            return success();
        }
        return failure();
    }
};

// Creates a new pattern to match MultiRZOp with adjoint flag set to true and canonicalize it.
struct MultiRZOpResolveGateLevelAdjointPattern : public OpRewritePattern<MultiRZOp> {
    using OpRewritePattern<MultiRZOp>::OpRewritePattern;
    LogicalResult matchAndRewrite(MultiRZOp op, PatternRewriter &rewriter) const override {
        if (!op.getAdjoint()) {
            return failure();
        }

        auto paramNeg = mlir::arith::NegFOp::create(rewriter, op.getLoc(), op.getTheta());

        rewriter.modifyOpInPlace(op, [&] {
            op.getThetaMutable().assign(paramNeg);
            op.setAdjoint(false);
        });
        return success();
    }
};

// Creates a new pattern to match PCPhaseOp with adjoint flag set to true and canonicalize it.
struct PCPhaseOpResolveGateLevelAdjointPattern : public OpRewritePattern<PCPhaseOp> {
    using OpRewritePattern<PCPhaseOp>::OpRewritePattern;
    LogicalResult matchAndRewrite(PCPhaseOp op, PatternRewriter &rewriter) const override {
        if (!op.getAdjoint()) {
            return failure();
        }

        auto paramNeg = mlir::arith::NegFOp::create(rewriter, op.getLoc(), op.getTheta());

        rewriter.modifyOpInPlace(op, [&] {
            op.getThetaMutable().assign(paramNeg);
            op.setAdjoint(false);
        });
        return success();
    }
};

// Creates a new pattern to match PauliRotOp with adjoint flag set to true and canonicalize it.
struct PauliRotOpResolveGateLevelAdjointPattern : public OpRewritePattern<PauliRotOp> {
    using OpRewritePattern<PauliRotOp>::OpRewritePattern;
    LogicalResult matchAndRewrite(PauliRotOp op, PatternRewriter &rewriter) const override {
        if (!op.getAdjoint()) {
            return failure();
        }

        auto paramNeg = mlir::arith::NegFOp::create(rewriter, op.getLoc(), op.getAngle());

        rewriter.modifyOpInPlace(op, [&] {
            op.getAngleMutable().assign(paramNeg);
            op.setAdjoint(false);
        });
        return success();
    }
};

// Creates a new pattern to match GlobalPhaseOp with adjoint flag set to true and canonicalize it.
struct GlobalPhaseOpResolveGateLevelAdjointPattern : public OpRewritePattern<GlobalPhaseOp> {
    using OpRewritePattern<GlobalPhaseOp>::OpRewritePattern;
    LogicalResult matchAndRewrite(GlobalPhaseOp op, PatternRewriter &rewriter) const override {
        if (!op.getAdjoint()) {
            return failure();
        }

        auto paramNeg = mlir::arith::NegFOp::create(rewriter, op.getLoc(), op.getAngle());

        rewriter.modifyOpInPlace(op, [&] {
            op.getAngleMutable().assign(paramNeg);
            op.setAdjoint(false);
        });
        return success();
    }
};

} // namespace

namespace catalyst {
namespace quantum {

#define GEN_PASS_DEF_RESOLVEGATELEVELADJOINT
#include "Quantum/Transforms/Passes.h.inc"

// Populate the patterns for the ResolveGateLevelAdjoint pass.
// Allows reference in ions_decompositions.cpp and merge_rotation.cpp
void populateResolveGateLevelAdjointPatterns(RewritePatternSet &patterns) {
    patterns.add<CustopOpResolveGateLevelAdjointPattern, MultiRZOpResolveGateLevelAdjointPattern,
                 PCPhaseOpResolveGateLevelAdjointPattern, PauliRotOpResolveGateLevelAdjointPattern,
                 GlobalPhaseOpResolveGateLevelAdjointPattern>(patterns.getContext());
}

struct ResolveGateLevelAdjoint : impl::ResolveGateLevelAdjointBase<ResolveGateLevelAdjoint> {
    using ResolveGateLevelAdjointBase::ResolveGateLevelAdjointBase;

    void runOnOperation() final {
        RewritePatternSet patterns(&getContext());
        populateResolveGateLevelAdjointPatterns(patterns);

        if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
            return signalPassFailure();
        }
    }
};

} // namespace quantum
} // namespace catalyst
