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

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"

#include "Ion/IR/IonOps.h"
#include "Ion/Transforms/Patterns.h"
#include "Quantum/IR/QuantumOps.h"

#include "oqd_database_managers.hpp"

using namespace mlir;
using namespace catalyst::ion;
using namespace catalyst::quantum;

namespace catalyst {
namespace ion {

enum LevelTransition {
    // Encoding of level transtions for a pulse
    // For example, "DOWN_E" means the transition from downstate to estate
    DOWN_E = 0,
    UP_E = 1,
};

std::optional<int64_t> walkBackQubitSSA(quantum::CustomOp gate, int64_t position)
{
    // TODO: make that function able to cross control flow op (for, while, ...)
    auto qubit = gate.getInQubits()[position];
    auto definingOp = qubit.getDefiningOp();

    if (auto extractOp = dyn_cast<quantum::ExtractOp>(definingOp)) {
        if (extractOp.getIdxAttr().has_value()) {
            return extractOp.getIdxAttr().value();
        }
        return std::nullopt;
    }
    else {
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

mlir::LogicalResult oneQubitGateToPulse(CustomOp op, mlir::PatternRewriter &rewriter, double phase1,
                                        double phase2)
{
    std::vector<Beam> beams1 = getBeams1Params();

    auto qubitIndex = walkBackQubitSSA(op, 0);
    if (qubitIndex.has_value()) {
        // Set the optional transition index now
        auto qubitIndexValue = qubitIndex.value();
        Beam beam = beams1[qubitIndexValue];

        // TODO: assumption for indices 0: 0->e, 1: 1->e
        auto beam0toEAttr = BeamAttr::get(
            op.getContext(),
            /*transition_index=*/rewriter.getI64IntegerAttr(LevelTransition::DOWN_E),
            rewriter.getF64FloatAttr(beam.rabi), rewriter.getF64FloatAttr(beam.detuning),
            rewriter.getI64VectorAttr(beam.polarization),
            rewriter.getI64VectorAttr(beam.wavevector));
        auto beam1toEAttr = BeamAttr::get(
            op.getContext(), /*transition_index=*/rewriter.getI64IntegerAttr(LevelTransition::UP_E),
            rewriter.getF64FloatAttr(beam.rabi), rewriter.getF64FloatAttr(beam.detuning),
            rewriter.getI64VectorAttr(beam.polarization),
            rewriter.getI64VectorAttr(beam.wavevector));

        // TODO: Pull the math formula from database and apply it in MLIR (but right now it is not
        // in the database) Potentially Rabi and Detuning become SSA values and not attributes.

        auto loc = op.getLoc();
        auto qubits = op.getInQubits();

        auto ppOp = rewriter.create<ion::ParallelProtocolOp>(
            loc, qubits, [&](OpBuilder &builder, Location loc, ValueRange qubits) {
                mlir::FloatAttr phase1Attr = builder.getF64FloatAttr(phase1);
                mlir::FloatAttr phase2Attr = builder.getF64FloatAttr(phase2);
                // TODO: Add the relationship between angle and time
                auto time = op.getParams().front();
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

mlir::LogicalResult MSGateToPulse(CustomOp op, mlir::PatternRewriter &rewriter)
{
    auto qnode = op->getParentOfType<func::FuncOp>();
    ion::SystemOp ionSystem;
    qnode.walk([&](ion::SystemOp op) {
        ionSystem = op;
        return WalkResult::interrupt();
    });
    auto qubitIndex0 = walkBackQubitSSA(op, 0);
    auto qubitIndex1 = walkBackQubitSSA(op, 1);
    if (qubitIndex0.has_value() && qubitIndex1.has_value()) {
        // TODO: double check the nex assumption, there is (n**2, 2) (combinatorial) =
        // n**2(n**2-1)/2 two qubits They are 3 phonons per ion (x, y , z)

        quantum::AllocOp allocOp;
        qnode.walk([&](quantum::AllocOp op) {
            allocOp = op;
            return WalkResult::interrupt();
        });
        auto qubitIndex0Value = qubitIndex0.value();
        auto qubitIndex1Value = qubitIndex1.value();
        auto nQubits = allocOp.getNqubitsAttr();
        if (nQubits.has_value()) {
            // Triangular indices: (a, b) -> n(n-1)/2
            // e.g.
            // (0, 1) -> 0
            // (0, 2) -> 1
            // (0, 3) -> 2
            // (0, 4) -> 3
            // (1, 2) -> 4
            // (1, 3) -> 5
            // (1, 4) -> 6
            // (2, 3) -> 7
            // (2, 4) -> 8
            // (3, 4) -> 9
            // Symmetric
            if (qubitIndex0Value > qubitIndex1Value) {
                std::swap(qubitIndex0Value, qubitIndex1Value);
            };
            // TODO: double check this formula
            auto indexInteraction = (qubitIndex0Value * (nQubits.value() - 1) -
                                     qubitIndex0Value * (qubitIndex0Value + 1)) /
                                        2 +
                                    (qubitIndex1Value - qubitIndex0Value - 1);

            // TODO: assumption is that each ion has 3 phonons (x, y, z)
            auto phonon0ComX = ionSystem.getPhonons()[3 * qubitIndex0Value];
            auto phonon1ComX = ionSystem.getPhonons()[3 * qubitIndex1Value];

            PhononAttr phonon0ComXAttr = cast<PhononAttr>(phonon0ComX);
            PhononAttr phonon1ComXAttr = cast<PhononAttr>(phonon1ComX);

            auto beam = ionSystem.getBeams2()[indexInteraction];
            BeamAttr beamAttr = cast<BeamAttr>(beam);

            auto loc = op.getLoc();
            auto qubits = op.getInQubits();

            auto ppOp = rewriter.create<ion::ParallelProtocolOp>(
                loc, qubits, [&](OpBuilder &builder, Location loc, ValueRange qubits) {
                    mlir::FloatAttr phase0Attr = builder.getF64FloatAttr(0.0);
                    // TODO: Add the relationship between angle and time
                    auto time = op.getParams().front();
                    auto qubit0 = qubits.front();
                    auto qubit1 = qubits.back();

                    // TODO: Double check the formula in Ion dialect document (each beamAttr is
                    // different) see below.
                    // TODO: Pull the math formula from database and apply it in MLIR
                    // Rabi and phase becomes SSA values and not attributes.

                    // Pulse1(
                    //     transition=Transition(level1=0,level2=e),
                    //     rabi= float from calibration db(~100 KHz-MHz),
                    //     detuning=float from calibration db,
                    //     phase=0,
                    //     polarization=Int array from from calibration db,
                    //     wavevector=Int array from from calibration db,
                    //     target=qubit0
                    //     time=rabi/ms_angle (double check the formula)
                    // )
                    auto beam0Attr =
                        BeamAttr::get(op.getContext(), rewriter.getI64IntegerAttr(0),
                                      beamAttr.getRabi(), beamAttr.getDetuning(),
                                      beamAttr.getPolarization(), beamAttr.getWavevector());
                    builder.create<ion::PulseOp>(loc, time, qubit0, beam0Attr, phase0Attr);
                    // Pulse2(
                    //     transition=Transition(level1=1,level2=e),
                    //     rabi= float from calibration db(~100 KHz-MHz),
                    //     detuning=Delta(from database) + omega_0 (COMx phonon frequency) + mu
                    //     (from database), phase=0, polarization=Int array from from calibration
                    //     db, wavevector=-Int array from from calibration db, target=qubit0
                    //     time=rabi/ms_angle (double check the formula)
                    // )

                    // TODO: where to find delta and mu?
                    // TODO: wave vector change sign
                    auto beam1Attr =
                        BeamAttr::get(op.getContext(), rewriter.getI64IntegerAttr(1),
                                      beamAttr.getRabi(), phonon0ComXAttr.getEnergy(),
                                      beamAttr.getPolarization(), beamAttr.getWavevector());
                    builder.create<ion::PulseOp>(loc, time, qubit0, beam1Attr, phase0Attr);
                    // Pulse3(
                    //     transition=Transition(level1=1,level2=e),
                    //     rabi= float from calibration db(~100 KHz-MHz),
                    //     detuning=Delta(from database) - omega_0 - mu(from database)
                    //     phase=0,
                    //     polarization=Int array from from calibration db,
                    //     wavevector=-Int array from from calibration db,
                    //     target=qubit0
                    //     time=rabi/ms_angle (double check the formula)
                    // )

                    // TODO: where to find delta and mu?
                    // TODO: phonon0ComXAttr change sign
                    // TODO: wave vector change sign
                    auto beam2Attr =
                        BeamAttr::get(op.getContext(), rewriter.getI64IntegerAttr(1),
                                      beamAttr.getRabi(), phonon0ComXAttr.getEnergy(),
                                      beamAttr.getPolarization(), beamAttr.getWavevector());
                    builder.create<ion::PulseOp>(loc, time, qubit0, beam2Attr, phase0Attr);

                    // Pulse4(
                    //     transition=Transition(level1=0,level2=e),
                    //     rabi= float from calibration db(~100 KHz-MHz),
                    //     detuning=float from calibration db,
                    //     phase=0,
                    //     polarization=Int array from from calibration db,
                    //     wavevector=Int array from from calibration db,
                    //     target=qubit1
                    //     time=rabi/ms_angle (double check the formula)
                    // )

                    auto beam3Attr =
                        BeamAttr::get(op.getContext(), rewriter.getI64IntegerAttr(0),
                                      beamAttr.getRabi(), beamAttr.getDetuning(),
                                      beamAttr.getPolarization(), beamAttr.getWavevector());
                    builder.create<ion::PulseOp>(loc, time, qubit1, beam3Attr, phase0Attr);

                    // Pulse5(
                    //     transition=Transition(level1=1,level2=e),
                    //     rabi= float from calibration db(~100 KHz-MHz),
                    //     detuning=Delta(from database) + omega_0 (COMx phonon frequency) + mu
                    //     (from database), phase=0, polarization=Int array from from calibration
                    //     db, wavevector=-Int array from from calibration db, target=qubit1
                    //     time=rabi/ms_angle (double check the formula)
                    // )

                    // TODO: where to find delta and mu?
                    // TODO: wave vector change sign
                    // TODO: phonon1ComXAttr change sign
                    auto beam4Attr =
                        BeamAttr::get(op.getContext(), rewriter.getI64IntegerAttr(1),
                                      beamAttr.getRabi(), phonon1ComXAttr.getEnergy(),
                                      beamAttr.getPolarization(), beamAttr.getWavevector());
                    builder.create<ion::PulseOp>(loc, time, qubit1, beam4Attr, phase0Attr);

                    // Pulse6(
                    //     transition=Transition(level1=1,level2=e),
                    //     rabi= float from calibration db(~100 KHz-MHz),
                    //     detuning=Delta(from database) - omega_0 (COMx phonon frequency) - mu(from
                    //     database) phase=0, polarization=Int array from from calibration db,
                    //     wavevector=-Int array from from calibration db,
                    //     target=qubit1
                    //     time=rabi/ms_angle (double check the formula)
                    // )
                    // )

                    // TODO: where to find delta and mu?
                    // TODO: wave vector change sign
                    // TODO: phonon1ComXAttr change sign
                    auto beam5Attr =
                        BeamAttr::get(op.getContext(), rewriter.getI64IntegerAttr(1),
                                      beamAttr.getRabi(), phonon1ComXAttr.getEnergy(),
                                      beamAttr.getPolarization(), beamAttr.getWavevector());
                    builder.create<ion::PulseOp>(loc, time, qubit1, beam5Attr, phase0Attr);

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

struct QuantumToIonRewritePattern : public mlir::OpRewritePattern<CustomOp> {
    using mlir::OpRewritePattern<CustomOp>::OpRewritePattern;

    mlir::LogicalResult matchAndRewrite(CustomOp op, mlir::PatternRewriter &rewriter) const override
    {
        // TODO: Assumption 1 Ion are in the same funcop as the operations
        // RX case -> PP(P1, P2)
        if (op.getGateName() == "RX") {
            auto result = oneQubitGateToPulse(op, rewriter, 0.0, 0.0);
            return result;
        }
        // RY case -> PP(P1, P2)
        else if (op.getGateName() == "RY") {
            auto result = oneQubitGateToPulse(op, rewriter, 0.0, llvm::numbers::pi);
            return result;
        }
        // MS case -> PP(P1, P2, P3, P4, P5, P6)
        else if (op.getGateName() == "MS") {
            auto result = MSGateToPulse(op, rewriter);
            return result;
        }
        return failure();
    }
};

void populateQuantumToIonPatterns(RewritePatternSet &patterns)
{
    patterns.add<QuantumToIonRewritePattern>(patterns.getContext());
}

} // namespace ion
} // namespace catalyst
