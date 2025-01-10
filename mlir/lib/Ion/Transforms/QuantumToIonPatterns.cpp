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

#include <algorithm>
#include <cassert>
#include <optional>

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Value.h"
#include "mlir/Transforms/DialectConversion.h"

#include "Ion/IR/IonOps.h"
#include "Ion/Transforms/Patterns.h"
#include "Ion/Transforms/oqd_database_managers.hpp"
#include "Quantum/IR/QuantumOps.h"

using namespace mlir;
using namespace catalyst::ion;
using namespace catalyst::quantum;

namespace catalyst {
namespace ion {

enum LevelTransition {
    // Encoding of level transitions for a pulse
    // For example, "DOWN_E" means the transition from downstate to estate
    DOWN_E = 0,
    UP_E = 1,
};

/**
 * @brief Walk back the qubit SSA until we reach an ExtractOp that has an idxAttr set, or until we
 *        reach the root of the SSA.
 *
 *        TODO (backlog): This function will likely not scale well as circuit depth increases.
 *        Is there a more efficient way to get the index of the qubit SSA?
 *
 * @param gate The gate that uses the qubit SSA.
 * @param position The position of the qubit SSA in the gate.
 * @return The index of the qubit SSA if it is known, otherwise std::nullopt.
 */
std::optional<int64_t> walkBackQubitSSA(quantum::CustomOp gate, int64_t position)
{
    // TODO (backlog): make that function able to cross control flow op (for, while, ...)
    auto qubit = gate.getInQubits()[position];
    auto definingOp = qubit.getDefiningOp();

    if (auto extractOp = dyn_cast<quantum::ExtractOp>(definingOp)) {
        if (extractOp.getIdxAttr().has_value()) {
            return extractOp.getIdxAttr().value();
        }
        return std::nullopt;
    }
    else {
        // TODO (backlog): if a pass on one operation fails, there may be a mixture of quantum ops
        // and parallel protocol ops in the IR. Since MLIR will continue processing other ops after
        // a failure, we may end up in a situation in which we attempt to cast a parallel protocol
        // op to a quantum op, which will raise an incompatible-type assertion error. It would be
        // best if we can immediately and gracefully exit after an emitError() call before this
        // happens, or modify this function to handle this case.
        auto customOp = cast<quantum::CustomOp>(definingOp);
        auto outQubits = customOp.getOutQubits();
        int64_t index = 0;
        for (const auto &outQubit : outQubits) {
            if (qubit == outQubit) {
                position = index;
                break;
            }
            ++index;
        }
        return walkBackQubitSSA(customOp, position);
    }
}

/**
 * @brief Returns the index of the two-qubit combination in the set of all possible combinations of
 *        two qubits in an n-qubit system.
 *
 *        In a system with n qubits, there are (n choose 2) = n*(n-1)/2 possible two-qubit
 *        combinations. Each combination is represented by a unique index. For example, in a five-
 *        qubit system, where the qubits are indexed as [0, 1, 2, 3, 4] the possible two-qubit
 *        combinations and their indices are:
 *
 *            0: (0, 1)
 *            1: (0, 2)
 *            2: (0, 3)
 *            3: (0, 4)
 *            4: (1, 2)
 *            5: (1, 3)
 *            6: (1, 4)
 *            7: (2, 3)
 *            8: (2, 4)
 *            9: (3, 4)
 *
 *        The formula to compute the index is:
 *
 *            index = sum_{j=1}^{i1} (n - j) + i2 - (i1 + 1)
 *                  = (i1 * n) - (i1 * (i1 + 1) / 2) + (i2 - i1 - 1)
 *
 * @param nQubits Number of qubits in the system.
 * @param idx1 Index of the first qubit.
 * @param idx2 Index of the second qubit.
 * @return int64_t
 */
int64_t getTwoQubitCombinationIndex(int64_t nQubits, int64_t idx1, int64_t idx2)
{
    assert(nQubits >= 2 && "At least two qubits must be present in the system.");

    assert(idx1 >= 0 && idx1 < nQubits && "First qubit index is out of range.");
    assert(idx2 >= 0 && idx2 < nQubits && "Second qubit index is out of range.");

    assert(idx1 != idx2 && "The two qubit indicies cannot be the same.");

    if (idx1 > idx2) {
        std::swap(idx1, idx2);
    };

    return (idx1 * nQubits) - (idx1 * (idx1 + 1) / 2) + (idx2 - idx1 - 1);
}

/**
 * @brief Computes the pulse duration given the rotation angle and the Rabi frequency.
 *
 *        The pulse duration t is given by:
 *
 *            t = angle / rabi.
 *
 *        This function returns the pulse duration as an mlir::Value by creating an arith::DivFOp.
 *        In order to do so, it must also create an arith::ConstantOp for the Rabi frequency.
 *
 * @param rewriter MLIR PatternRewriter
 * @param loc      MLIR Location
 * @param angle    Rotation angle as mlir::Value
 * @param rabi     Rabi frequency
 * @return mlir::Value The pulse duration.
 */
mlir::Value computePulseDuration(mlir::PatternRewriter &rewriter, mlir::Location &loc,
                                 const mlir::Value &angle, double rabi)
{
    TypedAttr rabiAttr = rewriter.getF64FloatAttr(rabi);
    mlir::Value rabiValue = rewriter.create<arith::ConstantOp>(loc, rabiAttr).getResult();
    return rewriter.create<arith::DivFOp>(loc, angle, rabiValue).getResult();
}

mlir::LogicalResult oneQubitGateToPulse(CustomOp op, mlir::PatternRewriter &rewriter, double phase1,
                                        double phase2, const std::vector<Beam> &beams1)
{
    auto qubitIndex = walkBackQubitSSA(op, 0);
    if (qubitIndex.has_value()) {
        // Set the optional transition index now
        auto qubitIndexValue = qubitIndex.value();
        Beam beam = beams1[qubitIndexValue];

        auto beam0toEAttr = BeamAttr::get(
            op.getContext(), rewriter.getI64IntegerAttr(LevelTransition::DOWN_E),
            rewriter.getF64FloatAttr(beam.rabi), rewriter.getF64FloatAttr(beam.detuning),
            rewriter.getI64VectorAttr(beam.polarization),
            rewriter.getI64VectorAttr(beam.wavevector));
        auto beam1toEAttr = BeamAttr::get(
            op.getContext(), rewriter.getI64IntegerAttr(LevelTransition::UP_E),
            rewriter.getF64FloatAttr(beam.rabi), rewriter.getF64FloatAttr(beam.detuning),
            rewriter.getI64VectorAttr(beam.polarization),
            rewriter.getI64VectorAttr(beam.wavevector));

        // TODO (backlog): Pull the math formula from database and apply it in MLIR (but right now
        // it is not in the database)
        // Potentially Rabi and Detuning become SSA values and not attributes.

        auto loc = op.getLoc();
        auto qubits = op.getInQubits();

        auto ppOp = rewriter.create<ion::ParallelProtocolOp>(
            loc, qubits, [&](OpBuilder &builder, Location loc, ValueRange qubits) {
                mlir::FloatAttr phase1Attr = builder.getF64FloatAttr(phase1);
                mlir::FloatAttr phase2Attr = builder.getF64FloatAttr(phase2);
                auto angle = op.getParams().front();
                auto time = computePulseDuration(rewriter, loc, angle, beam.rabi);
                auto qubit = qubits.front();
                builder.create<ion::PulseOp>(loc, time, qubit, beam0toEAttr, phase1Attr);
                builder.create<ion::PulseOp>(loc, time, qubit, beam1toEAttr, phase2Attr);
                builder.create<ion::YieldOp>(loc);
            });
        rewriter.replaceOp(op, ppOp);
        return success();
    }
    else {
        op.emitError() << "Impossible to determine the original qubit because of dynamism.";
        return failure();
    }
};

mlir::LogicalResult MSGateToPulse(CustomOp op, mlir::PatternRewriter &rewriter,
                                  const std::vector<Beam> &beams2,
                                  const std::vector<PhononMode> &phonons)
{
    auto qnode = op->getParentOfType<func::FuncOp>();

    auto qubitIndex0 = walkBackQubitSSA(op, 0);
    auto qubitIndex1 = walkBackQubitSSA(op, 1);

    if (qubitIndex0.has_value() && qubitIndex1.has_value()) {
        quantum::AllocOp allocOp;
        qnode.walk([&](quantum::AllocOp op) {
            allocOp = op;
            return WalkResult::interrupt();
        });
        auto qubitIndex0Value = qubitIndex0.value();
        auto qubitIndex1Value = qubitIndex1.value();
        auto nQubits = allocOp.getNqubitsAttr();
        if (nQubits.has_value()) {
            if (static_cast<size_t>(qubitIndex0Value) >= phonons.size()) {
                op.emitError() << "Missing phonon parameters for qubit " << qubitIndex0Value
                               << " used as input to MS gate; there are only " << phonons.size()
                               << " phonon parameters in the database."
                               << " Ensure that the database contains all necessary parameters for "
                                  "the circuit.";
                return failure();
            }

            if (static_cast<size_t>(qubitIndex1Value) >= phonons.size()) {
                op.emitError() << "Missing phonon parameters for qubit " << qubitIndex1Value
                               << " used as input to MS gate; there are only " << phonons.size()
                               << " phonon parameters in the database."
                               << " Ensure that the database contains all necessary parameters for "
                                  "the circuit.";
                return failure();
            }

            // Assume that each ion has 3 phonons (x, y, z)
            const Phonon &phonon0ComX = phonons[qubitIndex0Value].COM_x;
            const Phonon &phonon1ComX = phonons[qubitIndex1Value].COM_x;

            auto twoQubitComboIndex =
                getTwoQubitCombinationIndex(nQubits.value(), qubitIndex0Value, qubitIndex1Value);

            if (static_cast<size_t>(twoQubitComboIndex) >= beams2.size()) {
                op.emitError()
                    << "Missing two-qubit beam parameters for qubits "
                    << "(" << qubitIndex0Value << ", " << qubitIndex1Value << ") "
                    << "used as input to MS gate. Expected beam parameters for two-qubit "
                    << "combinatorial index " << twoQubitComboIndex << " but there are only "
                    << beams2.size() << " beam parameters in the database."
                    << " Ensure that the database contains all necessary parameters for the "
                       "circuit.";
                return failure();
            }

            const Beam &beam = beams2[twoQubitComboIndex];

            auto loc = op.getLoc();
            auto qubits = op.getInQubits();

            // Helper function to flip the sign of each element in a vector,
            // e.g. [a, b, c] -> [-a, -b, -c]
            auto flipSign = [](const std::vector<int64_t> &v) -> std::vector<int64_t> {
                std::vector<int64_t> result(v.size());
                std::transform(v.begin(), v.end(), result.begin(), [](int64_t x) { return -x; });
                return result;
            };

            auto ppOp = rewriter.create<ion::ParallelProtocolOp>(
                loc, qubits, [&](OpBuilder &builder, Location loc, ValueRange qubits) {
                    mlir::FloatAttr phase0Attr = builder.getF64FloatAttr(0.0);
                    auto angle = op.getParams().front();
                    auto time = computePulseDuration(rewriter, loc, angle, beam.rabi);
                    auto qubit0 = qubits.front();
                    auto qubit1 = qubits.back();

                    // Note that the each beamAttr below is different! The respective formulas are
                    // taken from the Ion dialect specification document.

                    // TODO: Pull the math formula from database and apply it in MLIR once OQD
                    // provides it.
                    // Rabi and phase may become SSA values and not attributes.

                    // Pulse1(
                    //     transition=Transition(level1=0,level2=e),
                    //     rabi=float from calibration db(~100 KHz-MHz),
                    //     detuning=float from calibration db,
                    //     phase=0,
                    //     polarization=Int array from from calibration db,
                    //     wavevector=Int array from from calibration db,
                    //     target=qubit0,
                    //     time=ms_angle/rabi
                    // )
                    auto beam1Attr = BeamAttr::get(
                        op.getContext(), rewriter.getI64IntegerAttr(LevelTransition::DOWN_E),
                        rewriter.getF64FloatAttr(beam.rabi),
                        rewriter.getF64FloatAttr(beam.detuning),
                        rewriter.getI64VectorAttr(beam.polarization),
                        rewriter.getI64VectorAttr(beam.wavevector));
                    builder.create<ion::PulseOp>(loc, time, qubit0, beam1Attr, phase0Attr);

                    // Pulse2(
                    //     transition=Transition(level1=1,level2=e),
                    //     rabi=float from calibration db(~100 KHz-MHz),
                    //     detuning=Delta(from database) + omega_0 (COMx phonon frequency) + mu(from
                    //       database),
                    //     phase=0,
                    //     polarization=Int array from from calibration db,
                    //     wavevector=-Int array from from calibration db,
                    //     target=qubit0,
                    //     time=ms_angle/rabi
                    // )

                    // TODO: Also need delta and mu (waiting on OQD to provide them)
                    auto beam2Attr = BeamAttr::get(
                        op.getContext(), rewriter.getI64IntegerAttr(LevelTransition::UP_E),
                        rewriter.getF64FloatAttr(beam.rabi),
                        // TODO: fill in formula with delta and mu once available
                        rewriter.getF64FloatAttr(beam.detuning + phonon0ComX.energy),
                        rewriter.getI64VectorAttr(beam.polarization),
                        rewriter.getI64VectorAttr(flipSign(beam.wavevector)));
                    builder.create<ion::PulseOp>(loc, time, qubit0, beam2Attr, phase0Attr);

                    // Pulse3(
                    //     transition=Transition(level1=1,level2=e),
                    //     rabi=float from calibration db(~100 KHz-MHz),
                    //     detuning=Delta(from database) - omega_0 - mu(from database),
                    //     phase=0,
                    //     polarization=Int array from from calibration db,
                    //     wavevector=-Int array from from calibration db,
                    //     target=qubit0,
                    //     time=ms_angle/rabi
                    // )

                    // TODO: Also need delta and mu (waiting on OQD to provide them)
                    auto beam3Attr = BeamAttr::get(
                        op.getContext(), rewriter.getI64IntegerAttr(LevelTransition::UP_E),
                        rewriter.getF64FloatAttr(beam.rabi),
                        // TODO: fill in formula with delta and mu once available
                        rewriter.getF64FloatAttr(beam.detuning - phonon0ComX.energy),
                        rewriter.getI64VectorAttr(beam.polarization),
                        rewriter.getI64VectorAttr(flipSign(beam.wavevector)));
                    builder.create<ion::PulseOp>(loc, time, qubit0, beam3Attr, phase0Attr);

                    // Pulse4(
                    //     transition=Transition(level1=0,level2=e),
                    //     rabi=float from calibration db(~100 KHz-MHz),
                    //     detuning=float from calibration db,
                    //     phase=0,
                    //     polarization=Int array from from calibration db,
                    //     wavevector=Int array from from calibration db,
                    //     target=qubit1,
                    //     time=ms_angle/rabi
                    // )

                    auto beam4Attr = BeamAttr::get(
                        op.getContext(), rewriter.getI64IntegerAttr(LevelTransition::DOWN_E),
                        rewriter.getF64FloatAttr(beam.rabi),
                        rewriter.getF64FloatAttr(beam.detuning),
                        rewriter.getI64VectorAttr(beam.polarization),
                        rewriter.getI64VectorAttr(beam.wavevector));
                    builder.create<ion::PulseOp>(loc, time, qubit1, beam4Attr, phase0Attr);

                    // Pulse5(
                    //     transition=Transition(level1=1,level2=e),
                    //     rabi=float from calibration db(~100 KHz-MHz),
                    //     detuning=Delta(from database) + omega_0 (COMx phonon frequency) + mu(from
                    //       database),
                    //     phase=0,
                    //     polarization=Int array from from calibration db,
                    //     wavevector=-Int array from from calibration db,
                    //     target=qubit1,
                    //     time=ms_angle/rabi
                    // )

                    // TODO: Also need delta and mu (waiting on OQD to provide them)
                    auto beam5Attr = BeamAttr::get(
                        op.getContext(), rewriter.getI64IntegerAttr(LevelTransition::UP_E),
                        rewriter.getF64FloatAttr(beam.rabi),
                        // TODO: fill in formula with delta and mu once available
                        rewriter.getF64FloatAttr(beam.detuning + phonon1ComX.energy),
                        rewriter.getI64VectorAttr(beam.polarization),
                        rewriter.getI64VectorAttr(flipSign(beam.wavevector)));
                    builder.create<ion::PulseOp>(loc, time, qubit1, beam5Attr, phase0Attr);

                    // Pulse6(
                    //     transition=Transition(level1=1,level2=e),
                    //     rabi=float from calibration db(~100 KHz-MHz),
                    //     detuning=Delta(from database) - omega_0 (COMx phonon frequency) - mu(from
                    //       database),
                    //     phase=0,
                    //     polarization=Int array from from calibration db,
                    //     wavevector=-Int array from from calibration db,
                    //     target=qubit1,
                    //     time=ms_angle/rabi
                    // )

                    // TODO: Also need delta and mu (waiting on OQD to provide them)
                    auto beam6Attr = BeamAttr::get(
                        op.getContext(), rewriter.getI64IntegerAttr(LevelTransition::UP_E),
                        rewriter.getF64FloatAttr(beam.rabi),
                        // TODO: fill in formula with delta and mu once available
                        rewriter.getF64FloatAttr(beam.detuning - phonon1ComX.energy),
                        rewriter.getI64VectorAttr(beam.polarization),
                        rewriter.getI64VectorAttr(flipSign(beam.wavevector)));
                    builder.create<ion::PulseOp>(loc, time, qubit1, beam6Attr, phase0Attr);

                    builder.create<ion::YieldOp>(loc);
                });
            rewriter.replaceOp(op, ppOp);
            return success();
        }
        else {
            op.emitError()
                << "Impossible to determine the number of qubits because the value is dynamic";
            return failure();
        }
    }
    else {
        op.emitError() << "Impossible to determine the original qubit because the value is dynamic";
        return failure();
    }
};

struct QuantumToIonRewritePattern : public mlir::OpConversionPattern<CustomOp> {
    using mlir::OpConversionPattern<CustomOp>::OpConversionPattern;

    std::vector<Beam> beams1;
    std::vector<Beam> beams2;
    std::vector<PhononMode> phonons;

    QuantumToIonRewritePattern(mlir::MLIRContext *ctx, const OQDDatabaseManager &dataManager)
        : mlir::OpConversionPattern<CustomOp>::OpConversionPattern(ctx)
    {
        beams1 = dataManager.getBeams1Params();
        beams2 = dataManager.getBeams2Params();
        phonons = dataManager.getPhononParams();
    }

    LogicalResult matchAndRewrite(CustomOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override

    {
        // Assume ions are in the same funcop as the operations
        // RX case -> PP(P1, P2)
        if (op.getGateName() == "RX") {
            auto result = oneQubitGateToPulse(op, rewriter, 0.0, 0.0, beams1);
            return result;
        }
        // RY case -> PP(P1, P2)
        else if (op.getGateName() == "RY") {
            auto result = oneQubitGateToPulse(op, rewriter, 0.0, llvm::numbers::pi, beams1);
            return result;
        }
        // MS case -> PP(P1, P2, P3, P4, P5, P6)
        else if (op.getGateName() == "MS") {
            auto result = MSGateToPulse(op, rewriter, beams2, phonons);
            return result;
        }
        return failure();
    }
};

void populateQuantumToIonPatterns(RewritePatternSet &patterns,
                                  const OQDDatabaseManager &dataManager)
{
    patterns.add<QuantumToIonRewritePattern>(patterns.getContext(), dataManager);
}

} // namespace ion
} // namespace catalyst
