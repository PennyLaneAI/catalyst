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

/// Decompose the PPR (pi/4) into PPR and PPMs operations via flattening method
/// as described in Figure 11(b) in the paper: https://arxiv.org/abs/1808.02892
///
/// ─────┌───┐───────
/// ─────│ P │(π/4)──
/// ─────└───┘───────
///
/// into
///
/// ─────┌───┐────┌───────┐─────┌───────┐──
/// ─────|-P |────| P(π/4)|─────| P(π/2)|──
/// ─────|   |────└───╦───┘─────└───╦───┘──
///      |   ╠════════╝             ║
///      |   |        ┌───┐         ║
/// |0⟩──| Y |────────| X ╠═════════╝
///      └───┘        └───┘
/// If we prepare |Y⟩ as axillary qubit, then we can use P⊗Z as the measurement operator
/// on first operation instead of -P⊗Y.
PPRotationOp decompose_pi_over_four_flattening(LogicalInitKind ancillaType, PPRotationOp op,
                                               TypedValue<IntegerType> measResult,
                                               PatternRewriter &rewriter)
{
    auto loc = op.getLoc();

    AllocOp axillaryQubitReg = buildAllocQreg(loc, 1, rewriter);
    ExtractOp zOp = buildExtractOp(loc, axillaryQubitReg, 0, rewriter);
    auto magic = rewriter.create<PrepareStateOp>(loc, ancillaType, zOp->getResults());

    SmallVector<Value> m1InQubits = op.getInQubits();
    m1InQubits.emplace_back(magic.getOutQubits().front());
    SmallVector<StringRef> pauliP = extractPauliString(op);

    uint16_t rotation_sign = 1;
    // if |Y⟩ is used as axillary qubit
    if (ancillaType == LogicalInitKind::plus_i) {
        pauliP.emplace_back("Z");
    }
    else if (ancillaType == LogicalInitKind::zero) {
        pauliP.emplace_back("Y");
        rotation_sign = -1;
    }
    else {
        assert(false && "Only |Y⟩ or |0⟩ can be used as axillary qubit for pi/4 decomposition");
    }

    auto ppmPZ =
        rewriter.create<PPMeasurementOp>(loc, pauliP, rotation_sign, m1InQubits, measResult);

    SmallVector<StringRef> pauliX = {"X"};
    auto ppmX =
        rewriter.create<PPMeasurementOp>(loc, pauliX, ppmPZ.getOutQubits().back(), measResult);

    auto cond = rewriter.create<arith::XOrIOp>(loc, ppmPZ.getMres(), ppmX.getMres());

    SmallVector<Value> outPZQubits = ppmPZ.getOutQubits();
    outPZQubits.pop_back();
    pauliP.pop_back();
    auto pprPI2 = rewriter.create<PPRotationOp>(loc, pauliP, 2, outPZQubits, cond.getResult());

    rewriter.replaceOp(op, pprPI2.getOutQubits());
    return pprPI2;
}

struct DecomposeCliffordPPR : public OpRewritePattern<PPRotationOp> {
    using OpRewritePattern::OpRewritePattern;

    LogicalInitKind ancillaType;
    DecomposeCliffordPPR(MLIRContext *context, LogicalInitKind ancillaType,
                         PatternBenefit benefit = 1)
        : OpRewritePattern(context, benefit), ancillaType(ancillaType)
    {
    }

    LogicalResult matchAndRewrite(PPRotationOp op, PatternRewriter &rewriter) const override
    {
        if (op.isClifford()) {
            decompose_pi_over_four_flattening(ancillaType, op, op.getCondition(), rewriter);
            return success();
        }
        return failure();
    }
};
} // namespace

namespace catalyst {
namespace qec {

void populateDecomposeCliffordPPRPatterns(RewritePatternSet &patterns, LogicalInitKind ancillaType)
{
    patterns.add<DecomposeCliffordPPR>(patterns.getContext(), ancillaType, 1);
}

} // namespace qec
} // namespace catalyst
