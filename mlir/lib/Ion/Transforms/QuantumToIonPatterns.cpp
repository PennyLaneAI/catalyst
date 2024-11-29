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

#include "Ion/IR/IonOps.h"
#include "Quantum/IR/QuantumOps.h"
#include "mlir/IR/PatternMatch.h"

#include "Ion/Transforms/Patterns.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include <iostream>

using namespace mlir;
using namespace catalyst::ion;
using namespace catalyst::quantum;

namespace catalyst {
namespace ion {

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
    auto qnode = op->getParentOfType<func::FuncOp>();
    ion::IonOp ion;
    qnode.walk([&](ion::IonOp op) {
        ion = op;
        return WalkResult::interrupt();
    });
    auto qubitIndex = walkBackQubitSSA(op, 0);
    if (qubitIndex.has_value()) {
        // Here we assume that they are 2*n_qubits beam for 1 qubit gate each pair is (transition
        // 0->e
        // and transition 1->e)
        auto qubitIndexValue = qubitIndex.value();
        auto beam0toE = ion.getBeams()[qubitIndexValue];
        BeamAttr beam0toEAttr = cast<BeamAttr>(beam0toE);
        auto beam1toE = ion.getBeams()[qubitIndexValue + 1];
        BeamAttr beam1toEAttr = cast<BeamAttr>(beam1toE);

        // TODO: Pull the math formula from database and apply it in MLIR
        // Rabi and phase becomes values and not attributes.

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

// void MSGateToPulse(CustomOp op, mlir::PatternRewriter &rewriter)
// {
//     auto qnode = op->getParentOfType<func::FuncOp>();
//     ion::IonOp ion;
//     qnode.walk([&](ion::IonOp op) {
//         ion = op;
//         return WalkResult::interrupt();
//     });
//     auto qubitIndex = walkBackQubitSSA(op, 0);
//     auto beam = ion.getBeams2Qubit()[qubitIndex];
//     BeamAttr beamAttr = cast<BeamAttr>(beam);

//     // TODO: Pull the math formula from database and apply it in MLIR
//     // Rabi and phase becomes SSA values and not attributes.

//     auto loc = op.getLoc();
//     auto qubits = op.getInQubits();

//     auto ppOp = rewriter.create<ion::ParallelProtocolOp>(
//         loc, qubits, [&](OpBuilder &builder, Location loc, ValueRange qubits) {
//             mlir::FloatAttr phase0Attr = builder.getF64FloatAttr(0.0);
//             // TODO: Add the relationship between angle and time
//             auto time = op.getParams().front();
//             auto qubit0 = qubits.front();
//             auto qubit1 = qubits.back();
//             builder.create<ion::PulseOp>(loc, time, qubit0, beam0Attr, phase0Attr);
//             builder.create<ion::PulseOp>(loc, time, qubit0, beam1Attr, phase0Attr);
//             builder.create<ion::PulseOp>(loc, time, qubit0, beam2Attr, phase0Attr);
//             builder.create<ion::PulseOp>(loc, time, qubit1, beam3Attr, phase0Attr);
//             builder.create<ion::PulseOp>(loc, time, qubit1, beam4Attr, phase0Attr);
//             builder.create<ion::PulseOp>(loc, time, qubit1, beam5Attr, phase0Attr);
//             builder.create<ion::YieldOp>(loc);
//         });
//     rewriter.replaceOp(op, ppOp);
// };

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
            // MSGateToPulse(op, rewriter);
            return success();
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
