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
#include "mlir/Dialect/Arith/IR/Arith.h" // for arith::XOrIOp and arith::ConstantOp
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

#include "Quantum/IR/QuantumOps.h"

#include "QEC/IR/QECOps.h"
#include "QEC/Transforms/PPRDecomposeUtils.h"
#include "QEC/Transforms/Patterns.h"
#include "QEC/Utils/PauliStringWrapper.h"

using namespace mlir;
using namespace catalyst::qec;
using namespace catalyst::quantum;

namespace {

// Return the magic state or complex conjugate of the magic state
LogicalInitKind getMagicState(PPRotationOp op)
{
    int16_t rotationKind = static_cast<int16_t>(op.getRotationKind());
    if (rotationKind > 0) {
        return LogicalInitKind::magic;
    }
    return LogicalInitKind::magic_conj;
}

/// Decompose the Non-Clifford (pi/8) PPR into PPR and PPMs operations via pauli corrected method
/// as described in Figure 13(a) in the paper: https://arxiv.org/pdf/2211.15465
///
/// ─────┌───┐───────
/// ─────│ P │(π/8)──
/// ─────└───┘───────
///
/// into
///
/// ─────┌───┐────────┌───────┐
/// ─────| P |────────| P(π/2)|
/// ─────|   |────────└───╦───┘
///      |   ╠═════╗      ║
///      |   |     ║  ┌───╩───┐
/// |m⟩──| Z |─────╚══╣ Y / X |
///      └───┘        └───────┘
/// All the operations in second diagram are PPM except for the last PPR P(π/2)
/// For P(-π/8) we need to use flip the ordering in which the X and Y measurements are applied.
///
/// Rather than performing a possible Pauli-Y measurement (sometimes more costly),
/// a |Y⟩ state can be used instead, `avoidPauliYMeasure` is used to control this. (see Figure 13(b)
/// in the paper)
///
/// Details:
/// - If P⊗Z measurement yields -1 then apply Y measurement, otherwise apply X measurement
///   * The measurement results are stored as i1 values.
///   * Measuring -1 corresponds to storing `true = 1` and 1 corresponds to storing `false = 0`.
///   - If the X or Y measurement yields -1, apply P(π/2) on the input qubits
void decomposePauliCorrectedPiOverEight(bool avoidPauliYMeasure, PPRotationOp op,
                                        PatternRewriter &rewriter)
{
    auto loc = op.getLoc();
    // We always initialize the magic state here, not the conjugate.
    auto magic = rewriter.create<FabricateOp>(loc, LogicalInitKind::magic);

    SmallVector<StringRef> pauliP = extractPauliString(op); // [P]
    SmallVector<Value> inQubits = op.getInQubits();         // [input qubits]

    // PPM (P⊗Z) on input qubits and |m⟩
    SmallVector<StringRef> extendedPauliP = pauliP;
    extendedPauliP.emplace_back("Z");                   // extend Z for the axillary qubit -> [P, Z]
    inQubits.emplace_back(magic.getOutQubits().back()); // [input qubits, |m⟩]

    int16_t rotationKind = static_cast<int16_t>(op.getRotationKind());
    uint16_t rotationSign = 1;
    if (rotationKind < 0) {
        rotationSign = -1;
    }
    auto ppmPZ = rewriter.create<PPMeasurementOp>(loc, extendedPauliP, rotationSign, inQubits);

    auto ppmPZRes = ppmPZ.getMres();
    if (avoidPauliYMeasure) {
        auto YBuilder = [&](OpBuilder &builder, Location loc) {
            // Initialize |Y⟩ state
            auto yQubit = rewriter.create<FabricateOp>(loc, LogicalInitKind::plus_i);
            // PPM (Z⊗Y) on qubits |m⟩ and |Y⟩
            SmallVector<Value> axillaryQubits = {ppmPZ.getOutQubits().back(),
                                                 yQubit.getOutQubits().back()};
            SmallVector<StringRef> pauliZZ = {"Z", "Z"}; // [Z, Z]
            auto ppmZZ = rewriter.create<PPMeasurementOp>(loc, pauliZZ, axillaryQubits);
            SmallVector<StringRef> pauliXX = {"X", "X"}; // [X, X]
            auto ppmXX = rewriter.create<PPMeasurementOp>(loc, pauliXX, ppmZZ.getOutQubits());
            SmallVector<Value> outPZQubits = ppmPZ.getOutQubits(); // [input qubits, |m⟩]
            outPZQubits.pop_back();                                // [input qubits]
            auto pprPI2 =
                rewriter.create<PPRotationOp>(loc, pauliP, 2, outPZQubits, ppmXX.getMres());
            for (auto q : axillaryQubits)
                rewriter.create<DeallocQubitOp>(loc, q);
            rewriter.create<scf::YieldOp>(loc, pprPI2.getOutQubits());
        };

        auto XBuilder = [&](OpBuilder &builder, Location loc) {
            // PPM (X) on qubit |m⟩
            SmallVector<StringRef> pauliX = {"X"};
            auto ppmX = rewriter.create<PPMeasurementOp>(loc, pauliX, ppmPZ.getOutQubits().back());
            SmallVector<Value> outPZQubits = ppmPZ.getOutQubits(); // [input qubits, |m⟩]
            outPZQubits.pop_back();                                // [input qubits]
            auto pprPI2 =
                rewriter.create<PPRotationOp>(loc, pauliP, 2, outPZQubits, ppmX.getMres());
            rewriter.create<DeallocQubitOp>(loc,
                                            ppmX.getOutQubits().back()); // Deallocate |m⟩ qubit
            rewriter.create<scf::YieldOp>(loc, pprPI2.getOutQubits());
        };

        scf::IfOp ifOp;
        if (rotationKind > 0) {
            ifOp = rewriter.create<scf::IfOp>(loc, ppmPZRes, YBuilder, XBuilder);
        }
        else {
            ifOp = rewriter.create<scf::IfOp>(loc, ppmPZRes, XBuilder, YBuilder);
        }
        rewriter.replaceOp(op, ifOp);
    }
    else {
        SmallVector<StringRef> pauliX = {"X"};
        SmallVector<StringRef> pauliY = {"Y"};
        auto ppmXY = rewriter.create<SelectPPMeasurementOp>(
            loc, ppmPZRes, rotationKind > 0 ? pauliY : pauliX, rotationKind > 0 ? pauliX : pauliY,
            ppmPZ.getOutQubits().back());
        // PPR P(π/2) on input qubits if PPM (X or Y) yields -1
        SmallVector<Value> outPZQubits = ppmPZ.getOutQubits(); // [input qubits, |m⟩]
        outPZQubits.pop_back();                                // [input qubits]
        auto pprPI2 = rewriter.create<PPRotationOp>(loc, pauliP, 2, outPZQubits, ppmXY.getMres());
        rewriter.create<DeallocQubitOp>(loc, ppmXY.getOutQubits().back());
        rewriter.replaceOp(op, pprPI2.getOutQubits());
    }
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
///   * The measurement results are stored as i1 values.
///   * Measuring -1 corresponds to storing `true = 1` and 1 corresponds to storing `false = 0`.
///   - If Z⊗Y and X measurement yield different result, then apply P(π/2) on the input qubits
///
/// FIXME: The result from the circuit above is non-deterministic. Test and reimplement Guillermo's
/// decomposition.
/// ─────┌───┐─────────────────────────┌───────┐──
/// ─────| P |─────────────────────────| P(π/2)|──
/// ─────|   |─────────────────────────└───╦───┘──
///      |   ╠════════════════════╗        ║
///      |   |  ┌───┐             ║  ┌───┐ ║
/// |m⟩──| Z |──| Z |─────────────║──| X ╠═╣
///      └───┘  |   |             ║  └───┘ ║
///             |   ╠═════╗(+1)   ║        ║
///             |   | ┌───╩───┐   ║  ┌───┐ ║
/// |0⟩─────────| Y |─| Z(π/2)|───╚══╣X/Z╠═╝
///             └───┘ └───────┘      └───┘
void decomposeAutoCorrectedPiOverEight(bool avoidPauliYMeasure, PPRotationOp op,
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
///   * The measurement results are stored as i1 values.
///   * Measuring -1 corresponds to storing `true = 1` and 1 corresponds to storing `false = 0`.
/// - If X measurement yields -1 then apply P(π/2)
/// FIXME: The expected value output is non-deterministic -- presumably caused by global phase.
void decomposeInjectMagicStatePiOverEight(PPRotationOp op, PatternRewriter &rewriter)
{
    auto loc = op.getLoc();

    // Fabricate the magic state |m⟩
    auto magic = rewriter.create<FabricateOp>(loc, getMagicState(op));

    SmallVector<StringRef> pauliP = extractPauliString(op); // [P = n qubit]
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
    rewriter.create<DeallocQubitOp>(loc, ppmX.getOutQubits().back());

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
                decomposeAutoCorrectedPiOverEight(avoidPauliYMeasure, op, rewriter);
                break;
            case DecomposeMethod::CliffordCorrected:
                decomposeInjectMagicStatePiOverEight(op, rewriter);
                break;
            case DecomposeMethod::PauliCorrected:
                decomposePauliCorrectedPiOverEight(avoidPauliYMeasure, op, rewriter);
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
