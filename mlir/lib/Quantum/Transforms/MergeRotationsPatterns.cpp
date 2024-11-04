// Copyright 2024 Xanadu Quantum Technologies Inc.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#define DEBUG_TYPE "merge-rotations"

#include "Quantum/IR/QuantumOps.h"
#include "Quantum/Transforms/Patterns.h"
#include "VerifyParentGateAnalysis.hpp"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Errc.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Arith/IR/Arith.h"

using llvm::dbgs;
using namespace mlir;
using namespace catalyst::quantum;

static const mlir::StringSet<> rotationsSet = {"RX",  "RY",  "RZ",  "PhaseShift",
                                               "CRX", "CRY", "CRZ", "ControlledPhaseShift",
                                               "qml.Rot", "qml.CRot"};

namespace {

struct MergeRotationsRewritePattern : public mlir::OpRewritePattern<CustomOp> {
    using mlir::OpRewritePattern<CustomOp>::OpRewritePattern;

    mlir::LogicalResult matchAndRewrite(CustomOp op, mlir::PatternRewriter &rewriter) const override
    {
        LLVM_DEBUG(dbgs() << "Simplifying the following operation:\n" << op << "\n");
        auto loc = op.getLoc();
        StringRef opGateName = op.getGateName();
        if (!rotationsSet.contains(opGateName))
            return failure();
        ValueRange inQubits = op.getInQubits();
        auto parentOp = dyn_cast_or_null<CustomOp>(inQubits[0].getDefiningOp());

        VerifyParentGateAndNameAnalysis vpga(op);
        if (!vpga.getVerifierResult()) {
            return failure();
        }

        if (opGateName == "qml.Rot" || opGateName == "qml.CRot") {
            LLVM_DEBUG(dbgs() << "Applying scalar formula for combined rotation operation:\n" << op << "\n");
            auto params = op.getParams();
            auto parentParams = parentOp.getParams();

            // Assuming params[0] = alpha1, params[1] = theta1, params[2] = beta1
            // and parentParams[0] = alpha2, parentParams[1] = theta2, parentParams[2] = beta2

            // Step 1: Calculate c1, c2, s1, s2
            auto c1 = rewriter.create<math::CosOp>(loc, params[1]);
            auto s1 = rewriter.create<math::SinOp>(loc, params[1]);
            auto c2 = rewriter.create<math::CosOp>(loc, parentParams[1]);
            auto s2 = rewriter.create<math::SinOp>(loc, parentParams[1]);

            // Step 2: Calculate cf
            auto c1Squared = rewriter.create<arith::MulFOp>(loc, c1, c1);
            auto c2Squared = rewriter.create<arith::MulFOp>(loc, c2, c2);
            auto s1Squared = rewriter.create<arith::MulFOp>(loc, s1, s1);
            auto s2Squared = rewriter.create<arith::MulFOp>(loc, s2, s2);
            auto cosAlphaDiff = rewriter.create<math::CosOp>(loc, rewriter.create<arith::SubFOp>(loc, params[0], parentParams[0]));

            auto term1 = rewriter.create<arith::MulFOp>(loc, c1Squared, c2Squared);
            auto term2 = rewriter.create<arith::MulFOp>(loc, s1Squared, s2Squared);
            auto product = rewriter.create<arith::MulFOp>(loc, c1, c2);
            product = rewriter.create<arith::MulFOp>(loc, product, s1);
            product = rewriter.create<arith::MulFOp>(loc, product, s2);
            auto two = rewriter.create<arith::ConstantOp>(loc, rewriter.getF64FloatAttr(2.0));
            auto term3 = rewriter.create<arith::MulFOp>(loc, two, rewriter.create<arith::MulFOp>(loc, product, cosAlphaDiff));

            auto cfSquare = rewriter.create<arith::SubFOp>(loc, rewriter.create<arith::AddFOp>(loc, term1, term2), term3);
            auto cf = rewriter.create<math::SqrtOp>(loc, cfSquare);

            // Step 3: Calculate theta_f = 2 * arccos(|cf|)
            auto absCf = rewriter.create<math::AbsFOp>(loc, cf);
            auto acosCf = rewriter.create<math::AcosOp>(loc, absCf);
            auto thetaF = rewriter.create<arith::MulFOp>(loc, two, acosCf);

            // Step 4: Calculate alpha_f
            auto alphaSum = rewriter.create<arith::AddFOp>(loc, params[0], parentParams[0]);
            auto betaDiff = rewriter.create<arith::SubFOp>(loc, parentParams[2], params[2]);
            auto sinAlphaSum = rewriter.create<math::SinOp>(loc, alphaSum);
            auto cosBetaDiff = rewriter.create<math::CosOp>(loc, betaDiff);

            auto term1_alpha = rewriter.create<arith::MulFOp>(loc, rewriter.create<arith::MulFOp>(loc, c1, s2), sinAlphaSum);
            auto term2_alpha = rewriter.create<arith::MulFOp>(loc, rewriter.create<arith::MulFOp>(loc, s1, s2), cosBetaDiff);
            auto numerator_alpha = rewriter.create<arith::SubFOp>(loc, rewriter.create<arith::NegFOp>(loc, term1_alpha), term2_alpha);

            auto cosAlphaSum = rewriter.create<math::CosOp>(loc, alphaSum);
            auto denominator_alpha = rewriter.create<arith::SubFOp>(loc, rewriter.create<arith::MulFOp>(loc, rewriter.create<arith::MulFOp>(loc, c1, c2), cosAlphaSum), term2_alpha);

            auto alphaF = rewriter.create<arith::NegFOp>(loc, rewriter.create<math::AtanOp>(loc, rewriter.create<arith::DivFOp>(loc, numerator_alpha, denominator_alpha)));

            // Step 5: Calculate beta_f
            auto betaSum = rewriter.create<arith::AddFOp>(loc, params[2], parentParams[2]);
            auto alphaDiffReversed = rewriter.create<arith::SubFOp>(loc, parentParams[0], params[0]);
            auto sinBetaSum = rewriter.create<math::SinOp>(loc, betaSum);
            auto cosAlphaDiffReversed = rewriter.create<math::CosOp>(loc, alphaDiffReversed);

            auto term1_beta = rewriter.create<arith::MulFOp>(loc, rewriter.create<arith::MulFOp>(loc, c1, s2), sinBetaSum);
            auto term2_beta = rewriter.create<arith::MulFOp>(loc, rewriter.create<arith::MulFOp>(loc, s1, s2), cosAlphaDiffReversed);
            auto numerator_beta = rewriter.create<arith::AddFOp>(loc, rewriter.create<arith::NegFOp>(loc, term1_beta), term2_beta);

            auto denominator_beta = denominator_alpha; // Reuse from alpha calculation if applicable
            auto betaF = rewriter.create<arith::NegFOp>(loc, rewriter.create<math::AtanOp>(loc, rewriter.create<arith::DivFOp>(loc, numerator_beta, denominator_beta)));

            // Step 6: Output angles (phi_f, theta_f, omega_f)
            // Assign phi_f = alphaF, theta_f = thetaF, omega_f = betaF as the final values
            SmallVector<mlir::Value> combinedAngles = {alphaF, thetaF, betaF};
            auto outQubitsTypes = op.getOutQubits().getTypes();
            auto outCtrlQubitsTypes = op.getOutCtrlQubits().getTypes();
            auto inQubits = op.getInQubits();
            auto inCtrlQubits = op.getInCtrlQubits();
            auto inCtrlValues = op.getInCtrlValues();
            rewriter.replaceOpWithNewOp<CustomOp>(op, outQubitsTypes, outCtrlQubitsTypes, combinedAngles, inQubits, opGateName, nullptr, inCtrlQubits, inCtrlValues);

            return success();
        } 
        else {
            TypeRange outQubitsTypes = op.getOutQubits().getTypes();
            TypeRange outQubitsCtrlTypes = op.getOutCtrlQubits().getTypes();
            ValueRange parentInQubits = parentOp.getInQubits();
            ValueRange parentInCtrlQubits = parentOp.getInCtrlQubits();
            ValueRange parentInCtrlValues = parentOp.getInCtrlValues();
            auto parentParams = parentOp.getParams();
            auto params = op.getParams();
            SmallVector<mlir::Value> sumParams;
            for (auto [param, parentParam] : llvm::zip(params, parentParams)) {
                mlir::Value sumParam =
                    rewriter.create<arith::AddFOp>(loc, parentParam, param).getResult();
                sumParams.push_back(sumParam);
            };
            auto mergeOp = rewriter.create<CustomOp>(loc, outQubitsTypes, outQubitsCtrlTypes, sumParams,
                                                    parentInQubits, opGateName, nullptr,
                                                    parentInCtrlQubits, parentInCtrlValues);

            op.replaceAllUsesWith(mergeOp);

            return success();
        }
    }
};

struct MergeMultiRZRewritePattern : public mlir::OpRewritePattern<MultiRZOp> {
    using mlir::OpRewritePattern<MultiRZOp>::OpRewritePattern;

    mlir::LogicalResult matchAndRewrite(MultiRZOp op,
                                        mlir::PatternRewriter &rewriter) const override
    {
        LLVM_DEBUG(dbgs() << "Simplifying the following operation:\n" << op << "\n");
        auto loc = op.getLoc();

        VerifyParentGateAnalysis<MultiRZOp> vpga(op);
        if (!vpga.getVerifierResult()) {
            return failure();
        }

        ValueRange inQubits = op.getInQubits();
        auto parentOp = dyn_cast_or_null<MultiRZOp>(inQubits[0].getDefiningOp());
        if (!parentOp)
            return failure();

        TypeRange outQubitsTypes = op.getOutQubits().getTypes();
        TypeRange outQubitsCtrlTypes = op.getOutCtrlQubits().getTypes();
        ValueRange parentInQubits = parentOp.getInQubits();
        ValueRange parentInCtrlQubits = parentOp.getInCtrlQubits();
        ValueRange parentInCtrlValues = parentOp.getInCtrlValues();

        auto parentTheta = parentOp.getTheta();
        auto theta = op.getTheta();

        mlir::Value sumParam = rewriter.create<arith::AddFOp>(loc, parentTheta, theta).getResult();

        auto mergeOp = rewriter.create<MultiRZOp>(loc, outQubitsTypes, outQubitsCtrlTypes, sumParam,
                                                  parentInQubits, nullptr, parentInCtrlQubits,
                                                  parentInCtrlValues);
        op.replaceAllUsesWith(mergeOp);
        return success();
    }
};
} // namespace

namespace catalyst {
namespace quantum {

void populateMergeRotationsPatterns(RewritePatternSet &patterns)
{
    patterns.add<MergeRotationsRewritePattern>(patterns.getContext(), 1);
    patterns.add<MergeMultiRZRewritePattern>(patterns.getContext(), 1);
}

} // namespace quantum
} // namespace catalyst
