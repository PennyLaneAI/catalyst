// Copyright 2023-2025 Xanadu Quantum Technologies Inc.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#define DEBUG_TYPE "merge-rotation"

#include "llvm/ADT/StringSet.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "Catalyst/IR/CatalystDialect.h"
#include "PBC/IR/PBCOps.h"
#include "Quantum/IR/QuantumOps.h"
#include "Quantum/Transforms/Patterns.h"

using namespace llvm;
using namespace mlir;
using namespace catalyst::quantum;

namespace catalyst {
namespace quantum {

#define GEN_PASS_DEF_MERGEROTATIONSPASS
#define GEN_PASS_DEF_RESOLVEADJOINTSPASS
#include "Quantum/Transforms/Passes.h.inc"

static const mlir::StringSet<> hermitianOps = {"Hadamard", "PauliX", "PauliY", "PauliZ", "CNOT",
                                               "CY",       "CZ",     "SWAP",   "Toffoli"};
static const mlir::StringSet<> rotationsOps = {"RX",  "RY",  "RZ",  "PhaseShift",
                                               "CRX", "CRY", "CRZ", "ControlledPhaseShift"};

struct ResolveCustomAdjointRewritePattern : public OpRewritePattern<CustomOp> {
    using OpRewritePattern::OpRewritePattern;

    LogicalResult matchAndRewrite(CustomOp op, PatternRewriter &rewriter) const override
    {
        if (!op.getAdjoint()) {
            return failure();
        }

        auto name = op.getGateName();
        if (hermitianOps.contains(name)) {
            op.setAdjoint(false);
            return success();
        }
        if (!rotationsOps.contains(name)) {
            return failure();
        }

        SmallVector<Value> paramsNeg;
        for (auto param : op.getParams()) {
            paramsNeg.push_back(arith::NegFOp::create(rewriter, op.getLoc(), param));
        }

        rewriter.replaceOpWithNewOp<CustomOp>(
            op, op.getOutQubits().getTypes(), op.getOutCtrlQubits().getTypes(), paramsNeg,
            op.getInQubits(), name, false, op.getInCtrlQubits(), op.getInCtrlValues());
        return success();
    }
};

struct ResolveMultiRZAdjointRewritePattern : public OpRewritePattern<MultiRZOp> {
    using OpRewritePattern::OpRewritePattern;

    LogicalResult matchAndRewrite(MultiRZOp op, PatternRewriter &rewriter) const override
    {
        if (!op.getAdjoint()) {
            return failure();
        }

        auto paramNeg = arith::NegFOp::create(rewriter, op.getLoc(), op.getTheta());
        rewriter.replaceOpWithNewOp<MultiRZOp>(
            op, op.getOutQubits().getTypes(), op.getOutCtrlQubits().getTypes(), paramNeg,
            op.getInQubits(), nullptr, op.getInCtrlQubits(), op.getInCtrlValues());
        return success();
    }
};

struct ResolvePCPhaseAdjointRewritePattern : public OpRewritePattern<PCPhaseOp> {
    using OpRewritePattern::OpRewritePattern;

    LogicalResult matchAndRewrite(PCPhaseOp op, PatternRewriter &rewriter) const override
    {
        if (!op.getAdjoint()) {
            return failure();
        }

        auto paramNeg = arith::NegFOp::create(rewriter, op.getLoc(), op.getTheta());
        rewriter.replaceOpWithNewOp<PCPhaseOp>(
            op, op.getOutQubits().getTypes(), op.getOutCtrlQubits().getTypes(), paramNeg,
            op.getDimAttr(), op.getInQubits(), nullptr, op.getInCtrlQubits(), op.getInCtrlValues());
        return success();
    }
};

static void populateResolveAdjointsPatterns(RewritePatternSet &patterns)
{
    patterns.add<ResolveCustomAdjointRewritePattern, ResolveMultiRZAdjointRewritePattern,
                 ResolvePCPhaseAdjointRewritePattern>(patterns.getContext());
}

struct ResolveAdjointsPass : impl::ResolveAdjointsPassBase<ResolveAdjointsPass> {
    using ResolveAdjointsPassBase::ResolveAdjointsPassBase;

    void runOnOperation() final
    {
        RewritePatternSet patterns(&getContext());
        populateResolveAdjointsPatterns(patterns);
        if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
            return signalPassFailure();
        }
    }
};

struct MergeRotationsPass : impl::MergeRotationsPassBase<MergeRotationsPass> {
    using MergeRotationsPassBase::MergeRotationsPassBase;

    void runOnOperation() final
    {
        LLVM_DEBUG(dbgs() << "merge rotation pass"
                          << "\n");

        Operation *module = getOperation();

        RewritePatternSet patternsCanonicalization(&getContext());

        populateResolveAdjointsPatterns(patternsCanonicalization);
        catalyst::pbc::PPRotationOp::getCanonicalizationPatterns(patternsCanonicalization,
                                                                 &getContext());
        catalyst::pbc::PPRotationArbitraryOp::getCanonicalizationPatterns(patternsCanonicalization,
                                                                          &getContext());
        if (failed(applyPatternsGreedily(module, std::move(patternsCanonicalization)))) {
            return signalPassFailure();
        }

        RewritePatternSet patterns(&getContext());
        populateLoopBoundaryPatterns(patterns, 1);
        populateMergeRotationsPatterns(patterns);
        if (failed(applyPatternsGreedily(module, std::move(patterns)))) {
            return signalPassFailure();
        }
    }
};

} // namespace quantum
} // namespace catalyst
