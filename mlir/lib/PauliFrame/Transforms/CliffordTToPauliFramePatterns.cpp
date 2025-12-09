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

#define DEBUG_TYPE "to-pauli-frame"

#include <concepts>

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/PatternMatch.h"
#include <llvm/Support/Casting.h>
#include <llvm/Support/Debug.h>
#include <llvm/Support/LogicalResult.h>
#include <llvm/Support/raw_ostream.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/Value.h>
#include <mlir/IR/ValueRange.h>

#include "PauliFrame/IR/PauliFrameOps.h"
#include "PauliFrame/Transforms/Patterns.h"
#include "Quantum/IR/QuantumDialect.h"
#include "Quantum/IR/QuantumOps.h"

using namespace mlir;

namespace {

using namespace catalyst::pauli_frame;
using namespace catalyst::quantum;

//===----------------------------------------------------------------------===//
// Helper functions
//===----------------------------------------------------------------------===//

template <typename T>
concept has_observable = requires(T obj) {
    // 1. The expression obj.getObs() must be valid
    // 2. The type returned by obj.getObs() must be exactly TypedValue<ObservableType>
    { obj.getObs() } -> std::same_as<TypedValue<ObservableType>>;
};

enum class GateEnum { I, X, Y, Z, H, S, T, CNOT, Unknown };

// Hash gate name to GateEnum
GateEnum hashGate(CustomOp op)
{
    auto gateName = op.getGateName();
    if (gateName == "Identity" || gateName == "I")
        return GateEnum::I;
    else if (gateName == "PauliX" || gateName == "X")
        return GateEnum::X;
    else if (gateName == "PauliY" || gateName == "Y")
        return GateEnum::Y;
    else if (gateName == "PauliZ" || gateName == "Z")
        return GateEnum::Z;
    else if (gateName == "H" || gateName == "Hadamard")
        return GateEnum::H;
    else if (gateName == "S")
        return GateEnum::S;
    else if (gateName == "T")
        return GateEnum::T;
    else if (gateName == "CNOT")
        return GateEnum::CNOT;
    else
        return GateEnum::Unknown;
}

// Insert the ops that physically apply the Pauli X and Z gates and a flush op.
// Applies the gates in the order X -> Z and returns the output qubit of the Z gate.
OpResult insertPauliOpsAfterFlush(PatternRewriter &rewriter, Location loc, FlushOp flushOp)
{
    auto pauliXIfOp = rewriter.create<scf::IfOp>(
        loc, flushOp.getXParity(),
        [&](OpBuilder &builder, Location loc) { // then
            auto pauliX = rewriter.create<CustomOp>(loc, "X", flushOp.getOutQubit());
            builder.create<scf::YieldOp>(loc, pauliX.getOutQubits());
        },
        [&](OpBuilder &builder, Location loc) { // else
            builder.create<scf::YieldOp>(loc, flushOp.getOutQubit());
        });

    auto pauliXOutQubit = pauliXIfOp->getResult(0);

    auto pauliZIfOp = rewriter.create<scf::IfOp>(
        loc, flushOp.getZParity(),
        [&](OpBuilder &builder, Location loc) { // then
            auto pauliZ = rewriter.create<CustomOp>(loc, "Z", pauliXOutQubit);
            builder.create<scf::YieldOp>(loc, pauliZ.getOutQubits());
        },
        [&](OpBuilder &builder, Location loc) { // else
            builder.create<scf::YieldOp>(loc, pauliXOutQubit);
        });

    return pauliZIfOp->getResult(0);
}

//===----------------------------------------------------------------------===//
// Gate-conversion functions
//===----------------------------------------------------------------------===//

/**
 * @brief Helper function to the Clifford+T -> PauliFrame pattern for Pauli gates (I, X, Y, Z).
 *
 * Performs the following rewrite, for example, from:
 *
 *  %0 = ... : !quantum.bit
 *  %1 = quantum.custom "X"() %0 : !quantum.bit  // or "I", "Y", "Z"
 *  %2 = <op_that_consumes_qubit> %1 : ...
 *
 * to:
 *
 *   %0 = ... : !quantum.bit
 *   %1 = pauli_frame.update[true, false] %0 : !quantum.bit  // and similar for other Pauli gates
 *   %2 = <op_that_consumes_qubit> %1 : ...
 */
LogicalResult convertPauliGate(CustomOp op, PatternRewriter &rewriter, bool x_parity, bool z_parity)
{
    auto loc = op->getLoc();
    auto outQubitTypes = op.getOutQubits().getTypes();
    auto inQubits = op.getInQubits();

    UpdateOp updateOp =
        rewriter.create<UpdateOp>(loc, outQubitTypes, rewriter.getBoolAttr(x_parity),
                                  rewriter.getBoolAttr(z_parity), inQubits);

    rewriter.replaceOp(op, updateOp.getOutQubits());
    return success();
}

/**
 * @brief Helper function to the Clifford+T -> PauliFrame pattern for Clifford gates (H, S, CNOT).
 *
 * Performs the following rewrite, for example, from:
 *
 *   %0 = ... : !quantum.bit
 *   %1 = quantum.custom "Hadamard"() %0 : !quantum.bit
 *   %2 = <op_that_consumes_qubit> %1 : ...
 *
 * to:
 *
 *   %0 = ... : !quantum.bit
 *   %1 = pauli_frame.update_with_clifford[Hadamard] %0 : !quantum.bit
 *   %2 = quantum.custom "Hadamard"() %1 : !quantum.bit
 *   %3 = <op_that_consumes_qubit> %2 : ...
 *
 * Note that since H = H†, CNOT = CNOT†, and since the Pauli conjugation relations for S and S† are
 * equivalent up to a global phase, we need not consider the adjoint parameter of the quantum gate.
 */
LogicalResult convertCliffordGate(CustomOp op, PatternRewriter &rewriter, CliffordGate gate)
{
    auto loc = op->getLoc();
    auto outQubitTypes = op.getOutQubits().getTypes();
    auto inQubits = op.getInQubits();

    UpdateWithCliffordOp updateOp =
        rewriter.create<UpdateWithCliffordOp>(loc, outQubitTypes, gate, inQubits);

    op->setOperands(updateOp->getResults());
    return success();
}

/**
 * @brief Helper function to the Clifford+T -> PauliFrame pattern for non-Clifford gates (T).
 *
 * Performs the following rewrite, for example, from:
 *
 *   %0 = ... : !quantum.bit
 *   %1 = quantum.custom "T"() %0 : !quantum.bit
 *   %2 = <op_that_consumes_qubit> %1 : ...
 *
 * to:
 *
 *   %0 = ... : !quantum.bit
 *   %x_parity, %z_parity, %out_qubit = pauli_frame.flush %0 : i1, i1, !quantum.bit
 *   %1 = scf.if %x_parity -> (!quantum.bit) {
 *     %out_qubits_0 = quantum.custom "X"() %out_qubit : !quantum.bit
 *     scf.yield %out_qubits_0 : !quantum.bit
 *   } else {
 *     scf.yield %out_qubit : !quantum.bit
 *   }
 *   %2 = scf.if %z_parity -> (!quantum.bit) {
 *     %out_qubits_0 = quantum.custom "Z"() %1 : !quantum.bit
 *     scf.yield %out_qubits_0 : !quantum.bit
 *   } else {
 *     scf.yield %1 : !quantum.bit
 *   }
 *   %3 = quantum.custom "T"() %2 : !quantum.bit
 *   %4 = <op_that_consumes_qubit> %3 : ...
 */
LogicalResult convertNonCliffordGate(CustomOp op, PatternRewriter &rewriter)
{
    auto loc = op->getLoc();
    auto outQubitTypes = op.getOutQubits().getTypes();

    if (outQubitTypes.size() > 1) {
        op->emitError("Only single-qubit non-Clifford gates are supported");
        return failure();
    }

    auto outQubitType = outQubitTypes[0];
    auto inQubits = op.getInQubits();

    FlushOp flushOp = rewriter.create<FlushOp>(loc, rewriter.getI1Type(), rewriter.getI1Type(),
                                               outQubitType, inQubits[0]);

    auto pauliZOutQubit = insertPauliOpsAfterFlush(rewriter, loc, flushOp);

    op->setOperands(pauliZOutQubit);

    return success();
}

//===----------------------------------------------------------------------===//
// Clifford+T to Pauli Frame Patterns
//===----------------------------------------------------------------------===//

/**
 * @brief Rewrite pattern for Clifford+T ops -> PauliFrame
 */
struct CliffordTToPauliFramePattern : public OpRewritePattern<CustomOp> {
    using OpRewritePattern::OpRewritePattern;

    LogicalResult matchAndRewrite(CustomOp op, PatternRewriter &rewriter) const override
    {
        auto op_enum = hashGate(op);
        switch (op_enum) {
        case GateEnum::I:
            return convertPauliGate(op, rewriter, false, false);
        case GateEnum::X:
            return convertPauliGate(op, rewriter, true, false);
        case GateEnum::Y:
            return convertPauliGate(op, rewriter, true, true);
        case GateEnum::Z:
            return convertPauliGate(op, rewriter, false, true);
        case GateEnum::H:
            return convertCliffordGate(op, rewriter, CliffordGate::Hadamard);
        case GateEnum::S:
            return convertCliffordGate(op, rewriter, CliffordGate::S);
        case GateEnum::CNOT:
            return convertCliffordGate(op, rewriter, CliffordGate::CNOT);
        case GateEnum::T:
            return convertNonCliffordGate(op, rewriter);
        case GateEnum::Unknown: {
            op->emitError(
                "Unsupported gate. Supported gates: I, X, Y, Z, H, S, S†, T, T†, and CNOT");
            return failure();
        }
        }
        return success();
    }
};

/**
 * @brief Rewrite pattern for Pauli record initialization of a single qubit
 *
 * The Pauli records are initialized by inserting `pauli_frame.init` ops immediately after each
 * single-qubit allocation op, `quantum.alloc_qb`, as follows, from:
 *
 *   %0 = quantum.alloc_qb : !quantum.bit
 *   %1 = <op_that_consumes_qubit> %0 : ...
 *
 * to:
 *
 *   %0 = quantum.alloc_qb : !quantum.bit
 *   %1 = pauli_frame.init %0
 *   %2 = <op_that_consumes_qubit> %1 : ...
 */
struct InitPauliRecordQbitPattern : public OpRewritePattern<AllocQubitOp> {
    using OpRewritePattern::OpRewritePattern;

    LogicalResult matchAndRewrite(AllocQubitOp op, PatternRewriter &rewriter) const override
    {
        auto loc = op->getLoc();
        auto qubit = op.getQubit();

        rewriter.setInsertionPointAfter(op);
        InitOp initOp = rewriter.create<InitOp>(loc, qubit.getType(), qubit);

        qubit.replaceAllUsesExcept(initOp.getOutQubits()[0], initOp);
        return success();
    }
};

/**
 * @brief Rewrite pattern for Pauli record initialization of a quantum register
 *
 * The Pauli records are initialized by inserting `pauli_frame.init_qreg` ops immediately after each
 * register allocation op, `quantum.alloc`, as follows, from:
 *
 *   %0 = quantum.alloc( 1) : !quantum.reg
 *   %1 = <op_that_consumes_qreg> %0 : ...
 *
 * to:
 *
 *   %0 = quantum.alloc( 1) : !quantum.reg
 *   %1 = pauli_frame.init_qreg %0 : !quantum.reg
 *   %2 = <op_that_consumes_qreg> %1 : ...
 */
struct InitPauliRecordQregPattern : public OpRewritePattern<AllocOp> {
    using OpRewritePattern::OpRewritePattern;

    LogicalResult matchAndRewrite(AllocOp op, PatternRewriter &rewriter) const override
    {
        auto loc = op->getLoc();
        auto qreg = op.getQreg();

        rewriter.setInsertionPointAfter(op);
        InitQregOp initQregOp = rewriter.create<InitQregOp>(loc, qreg.getType(), qreg);

        qreg.replaceAllUsesExcept(initQregOp.getOutQreg(), initQregOp);
        return success();
    }
};

/**
 * @brief Rewrite pattern for measurement corrections
 *
 * Measurement results are corrected by inserting `pauli_frame.correct_measurement` ops immediately
 * after each computational-basis mid-circuit measurement op, `quantum.measure`, as follows, from:
 *
 *   %0 = ... : !quantum.bit
 *   %mres, %1 = quantum.measure %0 : i1, !quantum.bit
 *
 * to:
 *
 *   %0 = ... : !quantum.bit
 *   %mres, %1 = quantum.measure %0 : i1, !quantum.bit
 *   %mres_1, %2 = pauli_frame.correct_measurement %mres, %1 : i1, !quantum.bit
 */
struct CorrectMeasurementPattern : public OpRewritePattern<MeasureOp> {
    using OpRewritePattern::OpRewritePattern;

    LogicalResult matchAndRewrite(MeasureOp op, PatternRewriter &rewriter) const override
    {
        auto loc = op->getLoc();
        auto mres = op.getMres();
        auto outQubit = op.getOutQubit();

        rewriter.setInsertionPointAfter(op);
        CorrectMeasurementOp correctMeasOp = rewriter.create<CorrectMeasurementOp>(
            loc, mres.getType(), outQubit.getType(), mres, outQubit);

        mres.replaceAllUsesExcept(correctMeasOp.getOutMres(), correctMeasOp);
        outQubit.replaceAllUsesExcept(correctMeasOp.getOutQubit(), correctMeasOp);
        return success();
    }
};

template <has_observable MeasurementProcessOp>
struct FlushBeforeMeasurementProcessPattern : public OpRewritePattern<MeasurementProcessOp> {
    using OpRewritePattern<MeasurementProcessOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(MeasurementProcessOp op, PatternRewriter &rewriter) const override
    {
        auto loc = op->getLoc();

        const auto obs = op.getObs();
        if (!obs) {
            op.emitError() << "Failed to flush Pauli record before terminal measurement process";
            return failure();
        }

        const auto obsOp = obs.getDefiningOp();

        auto compBasisOp = dyn_cast<ComputationalBasisOp>(obsOp);
        if (!compBasisOp) {
            obsOp->emitError() << "Only computational-basis observables are currently supported\n";
            return failure();
        }

        // The flush op will be inserted before the compbasis op
        rewriter.setInsertionPoint(compBasisOp);

        const OperandRange qubits = compBasisOp.getQubits();

        for (auto [idx, qubit] : llvm::enumerate(qubits)) {
            auto flushOp = rewriter.create<FlushOp>(loc, rewriter.getI1Type(), rewriter.getI1Type(),
                                                    qubit.getType(), qubit);
            auto pauliZOutQubit = insertPauliOpsAfterFlush(rewriter, loc, flushOp);
            compBasisOp.setOperand(idx, pauliZOutQubit);
        }
        return success();
    }
};

} // namespace

namespace catalyst {
namespace pauli_frame {

void populateCliffordTToPauliFramePatterns(RewritePatternSet &patterns)
{
    patterns.add<CliffordTToPauliFramePattern>(patterns.getContext());
    patterns.add<InitPauliRecordQbitPattern>(patterns.getContext());
    patterns.add<InitPauliRecordQregPattern>(patterns.getContext());
    patterns.add<CorrectMeasurementPattern>(patterns.getContext());
    patterns.add<FlushBeforeMeasurementProcessPattern<SampleOp>>(patterns.getContext());
    patterns.add<FlushBeforeMeasurementProcessPattern<CountsOp>>(patterns.getContext());
    // patterns.add<FlushBeforeMeasurementProcessPattern<ExpvalOp>>(patterns.getContext());    //
    // FIXME patterns.add<FlushBeforeMeasurementProcessPattern<VarianceOp>>(patterns.getContext());
    // // FIXME
    patterns.add<FlushBeforeMeasurementProcessPattern<ProbsOp>>(patterns.getContext());
}

} // namespace pauli_frame
} // namespace catalyst
