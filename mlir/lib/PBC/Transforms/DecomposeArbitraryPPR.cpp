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

#include "mlir/Transforms/DialectConversion.h"

#include "PBC/IR/PBCOps.h"
#include "PBC/Transforms/Patterns.h"
#include "PBC/Utils/PauliStringWrapper.h"
#include "Quantum/IR/QuantumOps.h"

using namespace mlir;
using namespace catalyst;
using namespace catalyst::quantum;
using namespace catalyst::pbc;

namespace {

/// Decompose arbitrary PPR operation as described in Figure 13(a)
// in the paper: https://arxiv.org/abs/2211.15465
///
/// This works for all arbitrary PPR operations except for the case where it is single-qubit
/// arbitrary PPR operation (i.e. ppr.arbitrary ["Z"](phi) %q0).
///
/// ─────┌───┐───────
/// ─────│ P │(phi)──
/// ─────└───┘───────
///
/// into
///
/// ─────┌───┐───────────────────────┌───────┐
/// ─────| P |───────────────────────| P(π/2)|
/// ─────|   |───────────────────────└───╦───┘
///      |   ╠═══════╗                   ║
///      |   |   ┌───╩───┐  ┌───────┐  ┌─╩─┐
/// |+⟩──| Z |───| X(π/2)|──| Z(phi)|──| X |
///      └───┘   └───────┘  └───────┘  └───┘
/// PZ, and X are PPMs, while X(phi), Z(phi), and P(pi/2) are PPRs.
LogicalResult convertArbitraryPPRToArbitraryZ(PPRotationArbitraryOp &op,
                                              PatternRewriter &rewriter) {
    auto loc = op.getLoc();

    /// |+⟩──
    auto qubitType = QubitType::get(rewriter.getContext());
    auto plus = LogicalInitKind::plus;
    auto plusQubit = PrepareStateOp::create(rewriter, loc, {qubitType}, plus);

    // ┌───┐──
    // | P |──
    // |   ╠══
    // | Z |──
    // └───┘
    SmallVector<StringRef> PZ = extractPauliString(op);
    PZ.emplace_back("Z");
    SmallVector<Value> inQubits = op.getInQubits();
    inQubits.emplace_back(plusQubit.getOutQubits().front());
    auto ppmPZ = PPMeasurementOp::create(rewriter, loc, PZ, inQubits);

    // ════╗
    // ┌───╩───┐
    // | X(π/2)|──
    // └───────┘
    SmallVector<StringRef> X = {"X"};
    const int8_t PI2 = 2; // For rotation of P(PI/2)
    auto inQubit = ppmPZ.getOutQubits().back();
    auto pprX = PPRotationOp::create(rewriter, loc, X, PI2, inQubit, ppmPZ.getMres());

    // ┌───────┐
    // | Z(phi)|──
    // └───────┘
    SmallVector<StringRef> Z = {"Z"};
    auto phi = op.getArbitraryAngle();
    auto pprZ = PPRotationArbitraryOp::create(rewriter, loc, Z, phi, pprX.getOutQubits());

    // ┌─╩─┐
    // | X |──
    // └───┘
    auto ppmX = PPMeasurementOp::create(rewriter, loc, X, pprZ.getOutQubits());

    // ┌───────┐──
    // | P(π/2)|──
    // └───╦───┘──
    //     ║
    SmallVector<Value> outPZQubits = ppmPZ.getOutQubits();
    outPZQubits.pop_back();
    auto P = op.getPauliProduct();
    auto pprP = PPRotationOp::create(rewriter, loc, P, PI2, outPZQubits, ppmX.getMres());

    rewriter.replaceOp(op, pprP.getOutQubits());

    // Deallocate the axillary qubits |+⟩
    DeallocQubitOp::create(rewriter, loc, ppmX.getOutQubits().back());
    return success();
}

struct DecomposeArbitraryPPR : public OpRewritePattern<PPRotationArbitraryOp> {
    using OpRewritePattern<PPRotationArbitraryOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(PPRotationArbitraryOp op,
                                  PatternRewriter &rewriter) const override {
        if (op.getPauliProduct() == rewriter.getStrArrayAttr({"Z"})) {
            return failure();
        }
        return convertArbitraryPPRToArbitraryZ(op, rewriter);
    }
};

} // namespace

namespace catalyst {
namespace pbc {

void populateDecomposeArbitraryPPRPatterns(RewritePatternSet &patterns) {
    patterns.add<DecomposeArbitraryPPR>(patterns.getContext());
}

} // namespace pbc
} // namespace catalyst
