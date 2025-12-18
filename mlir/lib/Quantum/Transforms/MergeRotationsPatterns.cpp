// Copyright 2024-2025 Xanadu Quantum Technologies Inc.

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

#include <array>
#include <cassert> // assert

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Errc.h"

#include "QEC/IR/QECOps.h"
#include "Quantum/IR/QuantumOps.h"
#include "Quantum/Transforms/Patterns.h"
#include "VerifyParentGateAnalysis.hpp"

using llvm::dbgs;
using namespace mlir;
using namespace catalyst::quantum;
using namespace catalyst::qec;

static const StringSet<> fixedRotationsAndPhaseShiftsSet = {
    "RX", "RY", "RZ", "PhaseShift", "CRX", "CRY", "CRZ", "ControlledPhaseShift"};
static const StringSet<> arbitraryRotationsSet = {"Rot", "CRot"};

namespace {

// convertOpParamsToValues: helper function for extracting CustomOp parameters as Values
SmallVector<Value> convertOpParamsToValues(CustomOp &op, PatternRewriter &rewriter)
{
    SmallVector<Value> values;
    auto params = op.getParams();
    for (auto param : params) {
        values.push_back(param);
    }
    return values;
}

// getStaticValuesOrNothing: helper function for extracting Rot or CRot parameters as:
// - doubles, in case they are constant
// - std::nullopt, otherwise
std::array<std::optional<double>, 3> getStaticValuesOrNothing(const SmallVector<Value> values)
{
    assert(values.size() == 3 && "found Rot or CRot operation should have exactly 3 parameters");
    auto staticValues = std::array<std::optional<double>, 3>{};
    for (auto [index, value] : llvm::enumerate(values)) {
        if (auto constOp = value.getDefiningOp();
            constOp && constOp->hasTrait<OpTrait::ConstantLike>()) {
            if (auto floatAttr = constOp->getAttrOfType<FloatAttr>("value")) {
                staticValues[index] = floatAttr.getValueAsDouble();
            }
        }
    }
    return staticValues;
}

template <typename ParentOpType, typename OpType>
struct MergeRotationsRewritePattern : public OpRewritePattern<OpType> {
    // Merge rotation patterns where at least one operand is non-static.
    // The result is a non-static CustomOp, as at least one operand is not known at compile time.
    using OpRewritePattern<OpType>::OpRewritePattern;

    // Fixed single rotations and phase shifts can be merged just by adding the angle parameters
    LogicalResult matchAndRewriteFixedRotationOrPhaseShift(OpType op,
                                                           PatternRewriter &rewriter) const
    {
        ValueRange inQubits = op.getInQubits();
        auto parentOp = llvm::cast<ParentOpType>(inQubits[0].getDefiningOp());

        TypeRange outQubitsTypes = op.getOutQubits().getTypes();
        TypeRange outQubitsCtrlTypes = op.getOutCtrlQubits().getTypes();
        ValueRange parentInQubits = parentOp.getInQubits();
        ValueRange parentInCtrlQubits = parentOp.getInCtrlQubits();
        ValueRange parentInCtrlValues = parentOp.getInCtrlValues();

        // Extract parameters of the op and its parent,
        // promoting the parameters to Values if necessary
        auto parentParams = convertOpParamsToValues(parentOp, rewriter);
        auto params = convertOpParamsToValues(op, rewriter);

        auto loc = op.getLoc();
        SmallVector<Value> sumParams;
        for (auto [param, parentParam] : llvm::zip(params, parentParams)) {
            Value sumParam = rewriter.create<arith::AddFOp>(loc, parentParam, param).getResult();
            sumParams.push_back(sumParam);
        }
        auto mergeOp = rewriter.create<CustomOp>(loc, outQubitsTypes, outQubitsCtrlTypes, sumParams,
                                                 parentInQubits, op.getGateName(), false,
                                                 parentInCtrlQubits, parentInCtrlValues);

        rewriter.replaceOp(op, mergeOp);
        rewriter.eraseOp(parentOp);

        return success();
    }

    // Arbitrary single rotations require more complex maths to be merged
    LogicalResult matchAndRewriteArbitraryRotation(OpType op, PatternRewriter &rewriter) const
    {
        ValueRange inQubits = op.getInQubits();
        auto parentOp = llvm::cast<ParentOpType>(inQubits[0].getDefiningOp());

        TypeRange outQubitsTypes = op.getOutQubits().getTypes();
        TypeRange outQubitsCtrlTypes = op.getOutCtrlQubits().getTypes();
        ValueRange parentInQubits = parentOp.getInQubits();
        ValueRange parentInCtrlQubits = parentOp.getInCtrlQubits();
        ValueRange parentInCtrlValues = parentOp.getInCtrlValues();

        // Extract parameters of the op and its parent,
        // promoting the parameters to Values if necessary
        auto parentParams = convertOpParamsToValues(parentOp, rewriter);
        auto params = convertOpParamsToValues(op, rewriter);

        // Parent params are ϕ1, θ1, and ω1
        // Params are ϕ2, θ2, and ω2
        Value phi1 = parentParams[0];
        Value theta1 = parentParams[1];
        Value omega1 = parentParams[2];
        Value phi2 = params[0];
        Value theta2 = params[1];
        Value omega2 = params[2];

        auto [phi1Opt, theta1Opt, omega1Opt] = getStaticValuesOrNothing(parentParams);
        auto [phi2Opt, theta2Opt, omega2Opt] = getStaticValuesOrNothing(params);

        Value phiF;
        Value thetaF;
        Value omegaF;

        // TODO: should we use an epsilon for comparing doubles here?
        bool omega1IsZero = omega1Opt.has_value() && omega1Opt.value() == 0.0;
        bool phi2IsZero = phi2Opt.has_value() && phi2Opt.value() == 0.0;
        bool theta1IsZero = theta1Opt.has_value() && theta1Opt.value() == 0.0;
        bool theta2IsZero = theta2Opt.has_value() && theta2Opt.value() == 0.0;

        auto loc = op.getLoc();

        // Special cases:
        //
        // 1. if (ω1 == 0 && ϕ2 == 0) { ϕF = ϕ1; θF = θ1 + θ2; ωF = ω2; }
        // 2a. if (θ1 == 0 && θ2 == 0) { ϕF = ϕ1 + ϕ2 + ω1 + ω2; θF = 0; ωF = 0; }
        // 2b. if (θ1 == 0) { ϕF = ϕ1 + ϕ2 + ω1; θF = θ2; ωF = ω2; }
        // 2c. if (θ2 == 0) { ϕF = ϕ1; θF = θ1; ωF = ω1 + ω2 + ϕ2; }
        auto zeroConst = rewriter.create<arith::ConstantOp>(loc, rewriter.getF64FloatAttr(0.0));
        if (omega1IsZero && phi2IsZero) {
            phiF = phi1;
            thetaF = rewriter.create<arith::AddFOp>(loc, theta1, theta2);
            omegaF = omega2;
        }
        else if (theta1IsZero && theta2IsZero) {
            phiF =
                rewriter.create<arith::AddFOp>(loc, rewriter.create<arith::AddFOp>(loc, phi1, phi2),
                                               rewriter.create<arith::AddFOp>(loc, omega1, omega2));
            thetaF = zeroConst;
            omegaF = zeroConst;
        }
        else if (theta1IsZero) {
            phiF = rewriter.create<arith::AddFOp>(
                loc, rewriter.create<arith::AddFOp>(loc, phi1, phi2), omega1);
            thetaF = theta2;
            omegaF = omega2;
        }
        else if (theta2IsZero) {
            phiF = phi1;
            thetaF = theta1;
            omegaF = rewriter.create<arith::AddFOp>(
                loc, rewriter.create<arith::AddFOp>(loc, omega1, omega2), phi2);
        }
        else {
            auto halfConst = rewriter.create<arith::ConstantOp>(loc, rewriter.getF64FloatAttr(0.5));
            auto twoConst = rewriter.create<arith::ConstantOp>(loc, rewriter.getF64FloatAttr(2.0));

            // α1 = (ϕ1 + ω1)/2, α2 = (ϕ2 + ω2)/2
            // β1 = (ϕ1 - ω1)/2, β2 = (ϕ2 - ω2)/2
            auto alpha1 = rewriter.create<arith::MulFOp>(
                loc, rewriter.create<arith::AddFOp>(loc, phi1, omega1), halfConst);
            auto alpha2 = rewriter.create<arith::MulFOp>(
                loc, rewriter.create<arith::AddFOp>(loc, phi2, omega2), halfConst);
            auto beta1 = rewriter.create<arith::MulFOp>(
                loc, rewriter.create<arith::SubFOp>(loc, phi1, omega1), halfConst);
            auto beta2 = rewriter.create<arith::MulFOp>(
                loc, rewriter.create<arith::SubFOp>(loc, phi2, omega2), halfConst);

            // c1 = cos(θ1/2), c2 = cos(θ2/2)
            // s1 = sin(θ1/2), s2 = sin(θ2/2)
            auto theta1Half = rewriter.create<arith::MulFOp>(loc, theta1, halfConst);
            auto c1 = rewriter.create<math::CosOp>(loc, theta1Half);
            auto s1 = rewriter.create<math::SinOp>(loc, theta1Half);
            auto theta2Half = rewriter.create<arith::MulFOp>(loc, theta2, halfConst);
            auto c2 = rewriter.create<math::CosOp>(loc, theta2Half);
            auto s2 = rewriter.create<math::SinOp>(loc, theta2Half);

            // cF = sqrt(c1^2 * c2^2 +
            //           s1^2 * s2^2 -
            //           2 * c1 * c2 * s1 * s2 * cos(ω1 + ϕ2))
            auto c1TimesC2 = rewriter.create<arith::MulFOp>(loc, c1, c2);
            auto s1TimesS2 = rewriter.create<arith::MulFOp>(loc, s1, s2);
            auto firstAddend =
                rewriter.create<arith::MulFOp>(loc, rewriter.create<arith::MulFOp>(loc, c1, c1),
                                               rewriter.create<arith::MulFOp>(loc, c2, c2));
            auto secondAddend =
                rewriter.create<arith::MulFOp>(loc, rewriter.create<arith::MulFOp>(loc, s1, s1),
                                               rewriter.create<arith::MulFOp>(loc, s2, s2));
            auto thirdAddend = rewriter.create<arith::NegFOp>(
                loc, rewriter.create<arith::MulFOp>(
                         loc, twoConst,
                         rewriter.create<arith::MulFOp>(
                             loc, c1TimesC2,
                             rewriter.create<arith::MulFOp>(
                                 loc, s1TimesS2,
                                 rewriter.create<math::CosOp>(
                                     loc, rewriter.create<arith::AddFOp>(loc, omega1, phi2))))));
            auto cF = rewriter.create<math::SqrtOp>(
                loc, rewriter.create<arith::AddFOp>(
                         loc, firstAddend,
                         rewriter.create<arith::AddFOp>(loc, secondAddend, thirdAddend)));

            // TODO: can we check these problematic scenarios for differentiability by code?
            // Problematic scenarios for differentiability:
            //
            // 1. if (cF == 0) { /* sqrt not differentiable at 0 */ return failure(); }
            // 2. if (cF == 1) { /* acos not differentiable at 1 */ return failure(); }

            // θF = 2 * acos(cF)
            auto acosCF = rewriter.create<math::AcosOp>(loc, cF);
            thetaF = rewriter.create<arith::MulFOp>(loc, twoConst, acosCF);

            // αF = - atan((- c1 * c2 * sin(α1 + α2) - s1 * s2 * sin(β2 - β1)) /
            //             (  c1 * c2 * cos(α1 + α2) - s1 * s2 * cos(β2 - β1)))
            auto alpha1PlusAlpha2 = rewriter.create<arith::AddFOp>(loc, alpha1, alpha2);
            auto beta2MinusBeta1 = rewriter.create<arith::SubFOp>(loc, beta2, beta1);
            auto term1 = rewriter.create<arith::NegFOp>(
                loc, rewriter.create<arith::MulFOp>(
                         loc, c1TimesC2, rewriter.create<math::SinOp>(loc, alpha1PlusAlpha2)));
            auto term2 = rewriter.create<arith::NegFOp>(
                loc, rewriter.create<arith::MulFOp>(
                         loc, s1TimesS2, rewriter.create<math::SinOp>(loc, beta2MinusBeta1)));
            auto term3 = rewriter.create<arith::MulFOp>(
                loc, c1TimesC2, rewriter.create<math::CosOp>(loc, alpha1PlusAlpha2));
            auto term4 = rewriter.create<arith::NegFOp>(
                loc, rewriter.create<arith::MulFOp>(
                         loc, s1TimesS2, rewriter.create<math::CosOp>(loc, beta2MinusBeta1)));
            auto alphaF = rewriter.create<arith::NegFOp>(
                loc, rewriter.create<math::AtanOp>(
                         loc, rewriter.create<arith::DivFOp>(
                                  loc, rewriter.create<arith::AddFOp>(loc, term1, term2),
                                  rewriter.create<arith::AddFOp>(loc, term3, term4))));

            // βF = - atan((- c1 * s2 * sin(α1 + β2) + s1 * c2 * sin(α2 - β1)) /
            //             (  c1 * s2 * cos(α1 + β2) + s1 * c2 * cos(α2 - β1)))
            auto c1TimesS2 = rewriter.create<arith::MulFOp>(loc, c1, s2);
            auto s1TimesC2 = rewriter.create<arith::MulFOp>(loc, s1, c2);
            auto alpha1PlusBeta2 = rewriter.create<arith::AddFOp>(loc, alpha1, beta2);
            auto alpha2MinusBeta1 = rewriter.create<arith::SubFOp>(loc, alpha2, beta1);
            auto term5 = rewriter.create<arith::NegFOp>(
                loc, rewriter.create<arith::MulFOp>(
                         loc, c1TimesS2, rewriter.create<math::SinOp>(loc, alpha1PlusBeta2)));
            auto term6 = rewriter.create<arith::MulFOp>(
                loc, s1TimesC2, rewriter.create<math::SinOp>(loc, alpha2MinusBeta1));
            auto term7 = rewriter.create<arith::MulFOp>(
                loc, c1TimesS2, rewriter.create<math::CosOp>(loc, alpha1PlusBeta2));
            auto term8 = rewriter.create<arith::MulFOp>(
                loc, s1TimesC2, rewriter.create<math::CosOp>(loc, alpha2MinusBeta1));
            auto betaF = rewriter.create<arith::NegFOp>(
                loc, rewriter.create<math::AtanOp>(
                         loc, rewriter.create<arith::DivFOp>(
                                  loc, rewriter.create<arith::AddFOp>(loc, term5, term6),
                                  rewriter.create<arith::AddFOp>(loc, term7, term8))));

            // ϕF = αF + βF
            phiF = rewriter.create<arith::AddFOp>(loc, alphaF, betaF);

            // ωF = αF - βF
            omegaF = rewriter.create<arith::SubFOp>(loc, alphaF, betaF);
        }

        auto sumParams = SmallVector<Value>{phiF, thetaF, omegaF};
        auto mergeOp = rewriter.create<CustomOp>(loc, outQubitsTypes, outQubitsCtrlTypes, sumParams,
                                                 parentInQubits, op.getGateName(), false,
                                                 parentInCtrlQubits, parentInCtrlValues);

        rewriter.replaceOp(op, mergeOp);
        rewriter.eraseOp(parentOp);

        return success();
    }

    LogicalResult matchAndRewrite(OpType op, PatternRewriter &rewriter) const override
    {
        LLVM_DEBUG(dbgs() << "Simplifying the following operation:\n" << op << "\n");

        StringRef opGateName = op.getGateName();
        if (!fixedRotationsAndPhaseShiftsSet.contains(opGateName) &&
            !arbitraryRotationsSet.contains(opGateName)) {
            return failure();
        }

        VerifyHeterogeneousParentGateAndNameAnalysis<OpType, ParentOpType> vpga(op);
        if (!vpga.getVerifierResult()) {
            return failure();
        }

        if (fixedRotationsAndPhaseShiftsSet.contains(opGateName)) {
            return matchAndRewriteFixedRotationOrPhaseShift(op, rewriter);
        }
        return matchAndRewriteArbitraryRotation(op, rewriter);
    }
};

struct MergePPRRewritePattern : public OpRewritePattern<PPRotationOp> {
    using OpRewritePattern::OpRewritePattern;

    LogicalResult matchAndRewrite(PPRotationOp op, PatternRewriter &rewriter) const override
    {
        // NOTE: a bit unorthodox, but we find the *second* PPR in a pair, since it's easier to
        // look backwards by checking inQubits.getDefiningOp
        ValueRange opInQubits = op.getInQubits();

        Operation *definingOp = opInQubits[0].getDefiningOp();
        if (!definingOp) {
            return failure();
        }

        auto parentOp = dyn_cast<PPRotationOp>(definingOp);
        if (!parentOp) {
            return failure();
        }

        // verify that parentOp agrees on all qubits, not just the first
        for (Value qubit : opInQubits) {
            if (qubit.getDefiningOp() != parentOp) {
                return failure();
            }
        }
        ValueRange parentOpOutQubits = parentOp.getOutQubits();
        if (parentOpOutQubits.size() != opInQubits.size()) {
            return failure();
        }

        // When two rotations have permuted Pauli strings, we can still merge them, we just need to
        // correctly re-map the inputs. This map stores the index of a qubit in parentOp's out
        // qubits at the index it appears in op's in qubits.
        SmallVector<unsigned> inverse_permutation;
        for (auto qubit : opInQubits) {
            inverse_permutation.push_back(cast<OpResult>(qubit).getResultNumber());
        }

        // check Pauli + qubit pairings
        ArrayAttr opPauliProduct = op.getPauliProduct();
        ArrayAttr parentOpPauliProduct = parentOp.getPauliProduct();
        for (size_t i = 0; i < opInQubits.size(); i++) {
            if (opPauliProduct[i] != parentOpPauliProduct[inverse_permutation[i]]) {
                return failure();
            }
        }

        // check same conditionals
        Value opCondition = op.getCondition();
        if (opCondition != parentOp.getCondition()) {
            return failure();
        }

        int16_t opRotation = static_cast<int16_t>(op.getRotationKind());
        int16_t parentOpRotation = static_cast<int16_t>(parentOp.getRotationKind());
        
        if (std::abs(opRotation) != std::abs(parentOpRotation)) {
            return failure();
        }

        int16_t newAngle = opRotation / 2;

        // remove identity operations
        ValueRange parentOpInQubits = parentOp.getInQubits();
        if (opRotation == -parentOpRotation || std::abs(newAngle) == 1) {
            rewriter.replaceOp(op, parentOpInQubits);
            rewriter.eraseOp(parentOp);

            return success();
        }


        Location loc = op.getLoc();

        // We need to construct the Pauli string + inQubits for new op. The simplest way to ensure
        // that permuted PPRs can merge correctly is to maintain output qubits order and permute
        // input qubits
        SmallVector<mlir::Value> newInQubits;
        for (size_t i = 0; i < parentOpInQubits.size(); i++) {
            newInQubits.push_back(parentOpInQubits[inverse_permutation[i]]);
        }

        auto mergeOp = rewriter.create<PPRotationOp>(loc, parentOpOutQubits.getTypes(), opPauliProduct, newAngle, newInQubits, opCondition);

        rewriter.replaceOp(op, mergeOp);
        rewriter.eraseOp(parentOp);

        return success();
    }
};

struct MergePPRArbitraryRewritePattern : public OpRewritePattern<PPRotationArbitraryOp> {
    using OpRewritePattern::OpRewritePattern;

    LogicalResult matchAndRewrite(PPRotationArbitraryOp op,
                                  PatternRewriter &rewriter) const override
    {
        ValueRange opInQubits = op.getInQubits();

        Operation *definingOp = opInQubits[0].getDefiningOp();
        if (!definingOp) {
            return failure();
        }

        auto parentOp = dyn_cast<PPRotationArbitraryOp>(definingOp);
        if (!parentOp) {
            return failure();
        }

        // verify that parentOp is parent of all qubits
        for (mlir::Value qubit : opInQubits) {
            if (qubit.getDefiningOp() != parentOp) {
                return failure();
            }
        }
        ValueRange parentOpOutQubits = parentOp.getOutQubits();
        if (parentOpOutQubits.size() != opInQubits.size()) {
            return failure();
        }

        // When two rotations have permuted Pauli strings, we can still merge them, we just need to
        // correctly re-map the inputs. This map stores the index of a qubit in parentOp's out
        // qubits at the index it appears in op's in qubits.
        SmallVector<unsigned> inverse_permutation;
        for (auto qubit : opInQubits) {
            inverse_permutation.push_back(cast<OpResult>(qubit).getResultNumber());
        }

        // check Pauli + qubit pairings
        ArrayAttr opPauliProduct = op.getPauliProduct();
        ArrayAttr parentOpPauliProduct = parentOp.getPauliProduct();
        for (size_t i = 0; i < opInQubits.size(); i++) {
            if (opPauliProduct[i] != parentOpPauliProduct[inverse_permutation[i]]) {
                return failure();
            }
        }

        // check same conditionals
        mlir::Value opCondition = op.getCondition();
        if (opCondition != parentOp.getCondition()) {
            return failure();
        }

        Location loc = op.getLoc();

        mlir::Value opRotation = op.getArbitraryAngle();
        mlir::Value parentOpRotation = parentOp.getArbitraryAngle();
        auto newAngleOp =
            rewriter.create<arith::AddFOp>(loc, opRotation, parentOpRotation).getResult();

        // We need to construct the Pauli string + inQubits for new op. The simplest way to ensure
        // that permuted PPRs can merge correctly is to maintain output qubits order and permute
        // input qubits
        ValueRange parentOpInQubits = parentOp.getInQubits();
        SmallVector<mlir::Value> newInQubits;
        for (size_t i = 0; i < parentOpInQubits.size(); i++) {
            newInQubits.push_back(parentOpInQubits[inverse_permutation[i]]);
        }

        auto mergeOp = rewriter.create<PPRotationArbitraryOp>(loc, parentOpOutQubits.getTypes(),
                                                              opPauliProduct, newAngleOp,
                                                              newInQubits, opCondition);

        // replace and erase old ops
        rewriter.replaceOp(op, mergeOp);
        rewriter.eraseOp(parentOp);

        return success();
    }
};

struct MergeMultiRZRewritePattern : public OpRewritePattern<MultiRZOp> {
    using OpRewritePattern<MultiRZOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(MultiRZOp op, PatternRewriter &rewriter) const override
    {
        LLVM_DEBUG(dbgs() << "Simplifying the following operation:\n" << op << "\n");
        auto loc = op.getLoc();

        VerifyParentGateAnalysis<MultiRZOp> vpga(op);
        if (!vpga.getVerifierResult()) {
            return failure();
        }

        ValueRange inQubits = op.getInQubits();
        auto parentOp = llvm::cast<MultiRZOp>(inQubits[0].getDefiningOp());

        TypeRange outQubitsTypes = op.getOutQubits().getTypes();
        TypeRange outQubitsCtrlTypes = op.getOutCtrlQubits().getTypes();
        ValueRange parentInQubits = parentOp.getInQubits();
        ValueRange parentInCtrlQubits = parentOp.getInCtrlQubits();
        ValueRange parentInCtrlValues = parentOp.getInCtrlValues();

        auto parentTheta = parentOp.getTheta();
        auto theta = op.getTheta();

        Value sumParam = rewriter.create<arith::AddFOp>(loc, parentTheta, theta).getResult();

        auto mergeOp = rewriter.create<MultiRZOp>(loc, outQubitsTypes, outQubitsCtrlTypes, sumParam,
                                                  parentInQubits, nullptr, parentInCtrlQubits,
                                                  parentInCtrlValues);
        rewriter.replaceOp(op, mergeOp);
        rewriter.eraseOp(parentOp);

        return success();
    }
};
} // namespace

namespace catalyst {
namespace quantum {

void populateMergeRotationsPatterns(RewritePatternSet &patterns)
{
    patterns.add<MergeRotationsRewritePattern<CustomOp, CustomOp>>(patterns.getContext(), 1);
    patterns.add<MergeMultiRZRewritePattern>(patterns.getContext(), 1);
    patterns.add<MergePPRRewritePattern>(patterns.getContext(), 1);
    patterns.add<MergePPRArbitraryRewritePattern>(patterns.getContext(), 1);
}

} // namespace quantum
} // namespace catalyst
