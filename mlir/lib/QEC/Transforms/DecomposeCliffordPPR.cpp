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

#define DEBUG_TYPE "decompose_clifford_ppr"

#include <mlir/Dialect/Arith/IR/Arith.h> // for arith::XOrIOp and arith::ConstantOp
#include <mlir/IR/Builders.h>

#include "Quantum/IR/QuantumOps.h"

#include "QEC/IR/QECDialect.h"
#include "QEC/Transforms/Patterns.h"
#include "QEC/Utils/PauliStringWrapper.h"
#include "QEC/Utils/Utility.h"

using namespace mlir;
using namespace catalyst::qec;
using namespace catalyst::quantum;

namespace {

// Determine type of measurement best on two options:
// 1. avoidPauliYMeasure == true: Use |Y⟩ as axillary qubit and measure P⊗Z
// 2. avoidPauliYMeasure == false: Use |0⟩ as axillary qubit and measure -P⊗Y
std::pair<StringRef, uint16_t> determinePauliAndRotationSignOfMeasurement(bool avoidPauliYMeasure)
{
    if (avoidPauliYMeasure) {
        return std::make_pair("Z", 1);
    }
    return std::make_pair("Y", -1);
}

/// Decompose the PPR (pi/4) into PPR and PPMs operations via flattening method
/// as described in Figure 11(b) in the paper: https://arxiv.org/abs/1808.02892
///
/// ─────┌───┐───────
/// ─────│ P │(π/4)──
/// ─────└───┘───────
///
/// into
///
/// ─────┌───┐────────┌───────┐──
/// ─────|-P |────────| P(π/2)|──
/// ─────|   |────────└───╦───┘──
///      |   ╠════════════╣
///      |   |     ┌───┐  ║
/// |0⟩──| Y |─────| X ╠══╝
///      └───┘     └───┘
/// If we prepare |Y⟩ as axillary qubit, then we can use P⊗Z as the measurement operator
/// on first operation instead of -P⊗Y.
PPRotationOp decompose_pi_over_four_flattening(bool avoidPauliYMeasure, PPRotationOp op,
                                               TypedValue<IntegerType> measResult,
                                               PatternRewriter &rewriter)
{
    auto loc = op.getLoc();

    // Fabricate axillary qubit |Y⟩ (plus_i) or |0⟩ (zero)
    auto axillaryType = avoidPauliYMeasure ? LogicalInitKind::plus_i : LogicalInitKind::zero;
    auto magicFabrication = rewriter.create<FabricateOp>(loc, axillaryType);

    auto [pauliForAxillaryQubit, rotationSign] =
        determinePauliAndRotationSignOfMeasurement(avoidPauliYMeasure);

    // Extract qubits and insert axillary qubit
    SmallVector<Value> m1InQubits = op.getInQubits();
    m1InQubits.emplace_back(magicFabrication.getOutQubits().front());

    // Extract P and insert Pauli for axillary qubit
    SmallVector<StringRef> pauliP = extractPauliString(op);
    pauliP.emplace_back(pauliForAxillaryQubit);

    auto ppmPZ =
        rewriter.create<PPMeasurementOp>(loc, pauliP, rotationSign, m1InQubits, measResult);

    SmallVector<StringRef> pauliX = {"X"};
    auto ppmX =
        rewriter.create<PPMeasurementOp>(loc, pauliX, ppmPZ.getOutQubits().back(), measResult);

    auto cond = rewriter.create<arith::XOrIOp>(loc, ppmPZ.getMres(), ppmX.getMres());

    SmallVector<Value> outPZQubits = ppmPZ.getOutQubits();
    outPZQubits.pop_back();
    pauliP.pop_back();

    const uint16_t PI_DENOMINATOR = 2; // For rotation of P(PI/2)
    auto pprPI2 =
        rewriter.create<PPRotationOp>(loc, pauliP, PI_DENOMINATOR, outPZQubits, cond.getResult());

    rewriter.replaceOp(op, pprPI2.getOutQubits());
    return pprPI2;
}

struct DecomposeCliffordPPR : public OpRewritePattern<PPRotationOp> {
    using OpRewritePattern::OpRewritePattern;

    bool avoidPauliYMeasure;

    DecomposeCliffordPPR(MLIRContext *context, bool avoidPauliYMeasure, PatternBenefit benefit = 1)
        : OpRewritePattern(context, benefit), avoidPauliYMeasure(avoidPauliYMeasure)
    {
    }

    LogicalResult matchAndRewrite(PPRotationOp op, PatternRewriter &rewriter) const override
    {
        if (op.isClifford()) {
            decompose_pi_over_four_flattening(avoidPauliYMeasure, op, op.getCondition(), rewriter);
            return success();
        }
        return failure();
    }
};
} // namespace

namespace catalyst {
namespace qec {

void populateDecomposeCliffordPPRPatterns(RewritePatternSet &patterns, bool avoidPauliYMeasure)
{
    patterns.add<DecomposeCliffordPPR>(patterns.getContext(), avoidPauliYMeasure, 1);
}

} // namespace qec
} // namespace catalyst
