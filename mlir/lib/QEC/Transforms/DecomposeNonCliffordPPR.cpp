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

#define DEBUG_TYPE "decompose-non-clifford-ppr"

#include <mlir/Dialect/Arith/IR/Arith.h> // for arith::XOrIOp and arith::ConstantOp
#include <mlir/IR/Builders.h>

#include "Quantum/IR/QuantumOps.h"

#include "QEC/IR/QECDialect.h"
#include "QEC/Transforms/PPRDecomposeUtils.h"
#include "QEC/Transforms/Patterns.h"
#include "QEC/Utils/PauliStringWrapper.h"

using namespace mlir;
using namespace catalyst::qec;
using namespace catalyst::quantum;

namespace {

// Return the magic state or complex conjugate of the magic state
LogicalInitKind getMagicState(QECOpInterface op)
{
    int16_t rotationKind = static_cast<int16_t>(op.getRotationKind());
    if (rotationKind > 0) {
        return LogicalInitKind::magic;
    }
    return LogicalInitKind::magic_conj;
}

/// Decompose the Non-Clifford (pi/8) PPR into PPR and PPMs operations via auto corrected method
/// as described in Figure 17(b) in the paper: https://arxiv.org/abs/1808.02892
///
/// ─────┌───┐───────
/// ─────│ P │(π/8)──
/// ─────└───┘───────
///
/// into
///
/// ─────┌───┐────────────────┌───────┐──
/// ─────| P |────────────────| P(π/2)|──
/// ─────|   |────────────────└───╦───┘──
///      |   ╠═══════════╗        ║
///      |   |  ┌───┐    ║  ┌───┐ ║
/// |m⟩──| Z |──| Z |────║──| X ╠═╣
///      └───┘  |   |    ║  └───┘ ║
///             |   ╠═════════════╝
///             |   |    ║  ┌───┐
/// |0⟩─────────| Y |────╚══╣X/Z|
///             └───┘       └───┘
/// All the operations in second diagram are PPM except for the last PPR P(π/2)
/// For P(-π/8) we need to use complex conjugate|m̅⟩ as the magic state.
///
/// Rather than performing a Pauli-Y measurement for Clifford rotations (sometimes more costly),
/// a |Y⟩ state is used instead, `avoidPauliYMeasure` is used to control this.
///
/// Details:
/// - If P⊗Z measurement yields -1 then apply X, otherwise apply Z
///   * The measurement results are stored as i1 values, -1 is true and 1 is false
/// - If Z⊗Y and X measurement yield different result, then apply P(π/2) on the input qubits
void decompose_auto_corrected_pi_over_eight(bool avoidPauliYMeasure, PPRotationOp op,
                                            PatternRewriter &rewriter)
{
    auto loc = op.getLoc();

    // Initialize |0⟩ (zero) or Fabricate |Y⟩ (plus_i)
    auto axillaryQubit = initializeZeroOrPlusI(avoidPauliYMeasure, loc, rewriter);
    auto magic = rewriter.create<FabricateOp>(loc, getMagicState(op));

    auto [pauliForAxillaryQubit, rotationSign] =
        determinePauliAndRotationSignOfMeasurement(avoidPauliYMeasure);

    SmallVector<StringRef> pauliP = extractPauliString(op);
    SmallVector<Value> inQubits = op.getInQubits(); // [input qubits]

    // PPM (P⊗Z) on input qubits and |m⟩
    // extend the pi/8 P with Z -> P⊗Z
    SmallVector<StringRef> extPauliP = pauliP;
    extPauliP.emplace_back("Z");                    // extend Z for the axillary qubit
    inQubits.emplace_back(magic.getOutQubits()[0]); // [input qubits, |m⟩]
    auto ppmPZ = rewriter.create<PPMeasurementOp>(loc, extPauliP, inQubits); // [input qubits, |m⟩]

    // PPM (Z⊗Y/0) on qubits |m⟩ and |Y⟩or|0⟩
    SmallVector<Value> axillaryQubits = {ppmPZ.getOutQubits().back(), axillaryQubit};
    SmallVector<StringRef> pauliZY = {"Z", pauliForAxillaryQubit}; // [Z, Y/Z]
    auto ppmZY = rewriter.create<PPMeasurementOp>(loc, pauliZY, rotationSign, axillaryQubits,
                                                  nullptr); // [|m⟩, |Y⟩/|0⟩]

    // PPM (X) on qubit |m⟩
    SmallVector<StringRef> pauliX = {"X"};
    auto ppmX = rewriter.create<PPMeasurementOp>(loc, pauliX, ppmZY.getOutQubits().front()); // |m⟩

    // PPM (X/Z) based on the result of PPM (P⊗Z) on qubit |0⟩
    SmallVector<StringRef> pauliZ = {"Z"};
    auto ppmXZ = rewriter.create<SelectPPMeasurementOp>(loc, ppmPZ.getMres(), pauliX, pauliZ,
                                                        ppmZY.getOutQubits().back()); // |0⟩

    // XOR of the results of PPM (P⊗Z) and PPM (X)
    auto condOp = rewriter.create<arith::XOrIOp>(loc, ppmZY.getMres(), ppmX.getMres());

    // PPR P(π/2) based on the result of XOR on input qubits
    SmallVector<Value> outPZQubits = ppmPZ.getOutQubits();
    outPZQubits.pop_back();
    auto pprPI2 = rewriter.create<PPRotationOp>(loc, pauliP, 2, outPZQubits, condOp.getResult());

    // Deallocate the axillary qubits
    rewriter.create<DeallocQubitOp>(loc, ppmXZ.getOutQubits().back()); // |0⟩
    rewriter.create<DeallocQubitOp>(loc, ppmX.getOutQubits().back());  // |m⟩

    rewriter.replaceOp(op, pprPI2.getOutQubits());
}

/// Decompose the Non-Clifford (pi/8) PPR into PPR and PPMs operations via inject magic state method
/// as described in Figure 7 in the paper: https://arxiv.org/abs/1808.02892
///
/// ─────┌───┐───────
/// ─────│ P │(π/8)──
/// ─────└───┘───────
///
/// into
///
/// ─────┌───┐────┌───────┐─────┌───────┐──
/// ─────| P |────| P(π/4)|─────| P(π/2)|──
/// ─────|   |────└───╦───┘─────└───╦───┘──
///      |   ╠════════╝             ║
///      |   |        ┌───┐         ║
/// |m⟩──| Z |────────| X ╠═════════╝
///      └───┘        └───┘
/// All the operations in second diagram are PPM except for the last PPR P(π/2) and P(π/4)
/// For P(-π/8) we need to use complex conjugate|m̅⟩ as the magic state.
///
/// Details:
/// - If P⊗Z measurement yields -1 then apply P(π/4)
///   * The measurement results are stored as i1 values, -1 is true and 1 is false
/// - If X measurement yields -1 then apply P(π/2)
void decompose_inject_magic_state_pi_over_eight(PPRotationOp op, PatternRewriter &rewriter)
{
    auto loc = op.getLoc();

    // Fabricate the magic state |m⟩
    auto magic = rewriter.create<FabricateOp>(loc, getMagicState(op));

    SmallVector<StringRef> pauliP = extractPauliString(op); // [P]
    SmallVector<Value> inQubits = op.getInQubits();         // [input qubits]

    // PPM (P⊗Z) on input qubits and |m⟩
    SmallVector<StringRef> extendedPauliP = pauliP;
    extendedPauliP.emplace_back("Z");               // extend Z for the axillary qubit -> [P, Z]
    inQubits.emplace_back(magic.getOutQubits()[0]); // [input qubits, |m⟩]
    auto ppmPZ = rewriter.create<PPMeasurementOp>(loc, extendedPauliP, inQubits);

    // PPR P(π/4) on input qubits if PPM (P⊗Z) yields -1
    SmallVector<Value> outPZQubits = ppmPZ.getOutQubits(); // [input qubits, |m⟩]
    outPZQubits.pop_back();                                // [input qubits]
    const uint16_t PI_DENOMINATOR = 4;                     // For rotation of P(PI/4)
    auto pprPI4 =
        rewriter.create<PPRotationOp>(loc, pauliP, PI_DENOMINATOR, outPZQubits, ppmPZ.getMres());

    // PPM (X) on |m⟩
    SmallVector<StringRef> pauliX = {"X"};
    auto ppmX = rewriter.create<PPMeasurementOp>(loc, pauliX, ppmPZ.getOutQubits().back());

    // PPR P(π/2) on input qubits if PPM (X) yields -1
    auto pprPI2 =
        rewriter.create<PPRotationOp>(loc, pauliP, 2, pprPI4.getOutQubits(), ppmX.getMres());

    // Deallocate the axillary qubit
    rewriter.create<DeallocQubitOp>(loc, pprPI2.getOutQubits().back());

    rewriter.replaceOp(op, pprPI2.getOutQubits());
}

struct DecomposeNonCliffordPPR : public OpRewritePattern<PPRotationOp> {
    using OpRewritePattern::OpRewritePattern;

    DecomposeMethod method;
    bool avoidPauliYMeasure;

    DecomposeNonCliffordPPR(MLIRContext *context, DecomposeMethod method, bool avoidPauliYMeasure,
                            PatternBenefit benefit = 1)
        : OpRewritePattern(context, benefit), method(method), avoidPauliYMeasure(avoidPauliYMeasure)
    {
    }

    LogicalResult matchAndRewrite(PPRotationOp op, PatternRewriter &rewriter) const override
    {
        if (op.isNonClifford() && !op.getCondition()) {
            switch (method) {
            case DecomposeMethod::AutoCorrected:
                decompose_auto_corrected_pi_over_eight(avoidPauliYMeasure, op, rewriter);
                break;
            case DecomposeMethod::CliffordCorrected:
                decompose_inject_magic_state_pi_over_eight(op, rewriter);
                break;
            }
            return success();
        }
        return failure();
    }
};
} // namespace

namespace catalyst {
namespace qec {

void populateDecomposeNonCliffordPPRPatterns(RewritePatternSet &patterns,
                                             DecomposeMethod decomposeMethod,
                                             bool avoidPauliYMeasure)
{
    patterns.add<DecomposeNonCliffordPPR>(patterns.getContext(), decomposeMethod,
                                          avoidPauliYMeasure, 1);
}

} // namespace qec
} // namespace catalyst
