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

#define DEBUG_TYPE "resolve-gate-level-adjoint"

#include "llvm/ADT/StringSet.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "QRef/IR/QRefOps.h"
#include "Quantum/IR/QuantumOps.h"
#include "Quantum/Transforms/Patterns.h"

using namespace mlir;

namespace {

static const mlir::StringSet<> hermitianOps = {"Hadamard", "PauliX", "PauliY", "PauliZ", "CNOT",
                                               "CY",       "CZ",     "SWAP",   "Toffoli"};
static const mlir::StringSet<> rotationsOps = {"RX",  "RY",  "RZ",  "PhaseShift",
                                               "CRX", "CRY", "CRZ", "ControlledPhaseShift"};

// Canonicalize adjoint on quantum.custom and qref.custom gates.
// For Hermitian gates, the adjoint flag is set to false.
// For rotations, the parameters are negated.
template <typename CustomOpTy>
struct CustomOpResolveGateLevelAdjointPattern : public OpRewritePattern<CustomOpTy> {
    using OpRewritePattern<CustomOpTy>::OpRewritePattern;

    LogicalResult matchAndRewrite(CustomOpTy op, PatternRewriter &rewriter) const override {
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
template <typename MultiRZOpTy>
struct MultiRZOpResolveGateLevelAdjointPattern : public OpRewritePattern<MultiRZOpTy> {
    using OpRewritePattern<MultiRZOpTy>::OpRewritePattern;
    LogicalResult matchAndRewrite(MultiRZOpTy op, PatternRewriter &rewriter) const override {
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
template <typename PCPhaseOpTy>
struct PCPhaseOpResolveGateLevelAdjointPattern : public OpRewritePattern<PCPhaseOpTy> {
    using OpRewritePattern<PCPhaseOpTy>::OpRewritePattern;
    LogicalResult matchAndRewrite(PCPhaseOpTy op, PatternRewriter &rewriter) const override {
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
template <typename PauliRotOpTy>
struct PauliRotOpResolveGateLevelAdjointPattern : public OpRewritePattern<PauliRotOpTy> {
    using OpRewritePattern<PauliRotOpTy>::OpRewritePattern;
    LogicalResult matchAndRewrite(PauliRotOpTy op, PatternRewriter &rewriter) const override {
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
template <typename GlobalPhaseOpTy>
struct GlobalPhaseOpResolveGateLevelAdjointPattern : public OpRewritePattern<GlobalPhaseOpTy> {
    using OpRewritePattern<GlobalPhaseOpTy>::OpRewritePattern;
    LogicalResult matchAndRewrite(GlobalPhaseOpTy op, PatternRewriter &rewriter) const override {
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

void populateResolveGateLevelAdjointPatterns(RewritePatternSet &patterns) {
    patterns.add<CustomOpResolveGateLevelAdjointPattern<quantum::CustomOp>,
                 CustomOpResolveGateLevelAdjointPattern<qref::CustomOp>,
                 MultiRZOpResolveGateLevelAdjointPattern<quantum::MultiRZOp>,
                 MultiRZOpResolveGateLevelAdjointPattern<qref::MultiRZOp>,
                 PCPhaseOpResolveGateLevelAdjointPattern<quantum::PCPhaseOp>,
                 PCPhaseOpResolveGateLevelAdjointPattern<qref::PCPhaseOp>,
                 PauliRotOpResolveGateLevelAdjointPattern<quantum::PauliRotOp>,
                 PauliRotOpResolveGateLevelAdjointPattern<qref::PauliRotOp>,
                 GlobalPhaseOpResolveGateLevelAdjointPattern<quantum::GlobalPhaseOp>,
                 GlobalPhaseOpResolveGateLevelAdjointPattern<qref::GlobalPhaseOp>>(
        patterns.getContext());
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
