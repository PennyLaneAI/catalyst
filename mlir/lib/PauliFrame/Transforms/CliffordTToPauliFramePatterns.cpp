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
#include <optional>
#include <string>

#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/LogicalResult.h"
#include "llvm/Support/raw_ostream.h"

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"

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

/**
 * Concept for operations that have an observable operand (`obs`). This is generally used for
 * operations representing measurement processes. This concept encapsulates the following
 * requirements on type `T`:
 *
 *   1. The expression obj.getObs() must be valid
 *   2. The type returned by obj.getObs() must be exactly TypedValue<ObservableType>
 */
template <typename T>
concept OpWithObservable = requires(T obj) {
    { obj.getObs() } -> std::same_as<TypedValue<ObservableType>>;
};

// The supported Clifford+T gates
enum class GateEnum { I, X, Y, Z, H, S, T, CNOT, Unknown };

// Hash gate name to GateEnum
GateEnum hashGate(CustomOp op)
{
    return llvm::StringSwitch<GateEnum>(op.getGateName())
        .Cases({"Identity", "I"}, GateEnum::I)
        .Cases({"PauliX", "X"}, GateEnum::X)
        .Cases({"PauliY", "Y"}, GateEnum::Y)
        .Cases({"PauliZ", "Z"}, GateEnum::Z)
        .Cases({"H", "Hadamard"}, GateEnum::H)
        .Case("S", GateEnum::S)
        .Case("T", GateEnum::T)
        .Case("CNOT", GateEnum::CNOT)
        .Default(GateEnum::Unknown);
}

// Insert the ops that physically apply the Pauli X and Z gates and a flush op.
// Applies the gates in the order X -> Z and returns the output qubit of the Z gate.
OpResult insertPauliOpsAfterFlush(PatternRewriter &rewriter, Location loc, FlushOp flushOp)
{
    auto pauliXIfOp = scf::IfOp::create(
        rewriter, loc, flushOp.getXParity(),
        [&](OpBuilder &builder, Location loc) { // then
            auto pauliX = CustomOp::create(rewriter, loc, "X", flushOp.getOutQubit());
            scf::YieldOp::create(builder, loc, pauliX.getOutQubits());
        },
        [&](OpBuilder &builder, Location loc) { // else
            scf::YieldOp::create(builder, loc, flushOp.getOutQubit());
        });

    auto pauliXOutQubit = pauliXIfOp->getResult(0);

    auto pauliZIfOp = scf::IfOp::create(
        rewriter, loc, flushOp.getZParity(),
        [&](OpBuilder &builder, Location loc) { // then
            auto pauliZ = CustomOp::create(rewriter, loc, "Z", pauliXOutQubit);
            scf::YieldOp::create(builder, loc, pauliZ.getOutQubits());
        },
        [&](OpBuilder &builder, Location loc) { // else
            scf::YieldOp::create(builder, loc, pauliXOutQubit);
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
    LLVM_DEBUG(llvm::dbgs() << "Applying Pauli frame protocol to Pauli gate: " << op.getGateName()
                            << "\n");
    auto loc = op->getLoc();
    auto outQubitTypes = op.getOutQubits().getTypes();
    auto inQubits = op.getInQubits();

    UpdateOp updateOp =
        UpdateOp::create(rewriter, loc, outQubitTypes, rewriter.getBoolAttr(x_parity),
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
    LLVM_DEBUG(llvm::dbgs() << "Applying Pauli frame protocol to Clifford gate: "
                            << op.getGateName() << "\n");
    auto loc = op->getLoc();
    auto outQubitTypes = op.getOutQubits().getTypes();
    auto inQubits = op.getInQubits();

    UpdateWithCliffordOp updateOp =
        UpdateWithCliffordOp::create(rewriter, loc, outQubitTypes, gate, inQubits);

    rewriter.modifyOpInPlace(op, [&] { op->setOperands(updateOp->getResults()); });
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
    LLVM_DEBUG(llvm::dbgs() << "Applying Pauli frame protocol to non-Clifford gate: "
                            << op.getGateName() << "\n");
    auto loc = op->getLoc();
    auto outQubitTypes = op.getOutQubits().getTypes();

    assert(outQubitTypes.size() == 1 &&
           "only single-qubit non-Clifford gates are supported (i.e. T gates)");

    auto outQubitType = outQubitTypes[0];
    auto inQubits = op.getInQubits();

    FlushOp flushOp = FlushOp::create(rewriter, loc, rewriter.getI1Type(), rewriter.getI1Type(),
                                      outQubitType, inQubits[0]);

    auto pauliZOutQubit = insertPauliOpsAfterFlush(rewriter, loc, flushOp);

    rewriter.modifyOpInPlace(op, [&] { op->setOperands(pauliZOutQubit); });

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
            op->emitError() << "Unsupported gate: '" << op.getGateName() << "'. "
                            << "Only Clifford+T gates are supported for Pauli frame conversion: "
                            << "I, X, Y, Z, H, S, S†, T, T†, and CNOT";
        }
        }
        std::string msg =
            llvm::formatv("failed to apply Pauli frame tracking protocols: unsupported gate: {0}",
                          op.getGateName().data());
        llvm_unreachable(msg.c_str());
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
        LLVM_DEBUG(llvm::dbgs() << "Initializing Pauli record of qubit: " << qubit << "\n");

        rewriter.setInsertionPointAfter(op);
        InitOp initOp = InitOp::create(rewriter, loc, qubit.getType(), qubit);

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
        LLVM_DEBUG(llvm::dbgs() << "Initializing Pauli records of qubits in register: " << qreg
                                << "\n");

        rewriter.setInsertionPointAfter(op);
        InitQregOp initQregOp = InitQregOp::create(rewriter, loc, qreg.getType(), qreg);

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
        LLVM_DEBUG(
            llvm::dbgs() << "Applying Pauli frame protocol to correct measurement result of qubit: "
                         << outQubit << "\n");

        rewriter.setInsertionPointAfter(op);
        CorrectMeasurementOp correctMeasOp = CorrectMeasurementOp::create(
            rewriter, loc, mres.getType(), outQubit.getType(), mres, outQubit);

        mres.replaceAllUsesExcept(correctMeasOp.getOutMres(), correctMeasOp);
        outQubit.replaceAllUsesExcept(correctMeasOp.getOutQubit(), correctMeasOp);
        return success();
    }
};

template <OpWithObservable MeasurementProcessOp>
struct FlushBeforeMeasurementProcessPattern : public OpRewritePattern<MeasurementProcessOp> {
    using OpRewritePattern<MeasurementProcessOp>::OpRewritePattern;

    // Helper function that checks if a flush op has been applied to a qubit value, and if so,
    // return it. If no flush op has been applied, return a nullopt. This function does not check if
    // multiple flush ops have been applied to a qubit value; it only returns the first one found.
    // By construction, a flush op should generally NOT be applied multiple times to the same qubit
    // value.
    std::optional<FlushOp> getFlushOpAppliedToQubit(const Value qubit) const
    {
        LLVM_DEBUG(llvm::dbgs() << "Attempting to retrieve pauli_frame.flush op applied to qubit: "
                                << qubit << "\n");

        for (const auto &use : qubit.getUses()) {
            auto *owner = use.getOwner();
            LLVM_DEBUG(llvm::dbgs() << " -> visiting qubit user op: " << owner->getName() << "\n");

            if (auto flushOp = llvm::dyn_cast<FlushOp>(owner)) {
                LLVM_DEBUG(llvm::dbgs() << "   -> success; returning\n");
                return flushOp;
            }
        }
        return std::nullopt;
    }

    // After every flush op there should be two `scf.if` ops that apply Pauli X and Z gates, in that
    // order, conditional on the x- and z-parity bits returns by the flush op. This is a helper
    // function that returns the output qubit of the Pauli Z op, which constitutes the final return
    // value of the flush "block".
    Value getOutputQubitOfPauliZAfterFlush(FlushOp flushOp) const
    {
        LLVM_DEBUG(llvm::dbgs()
                   << "Attempting to retrieve output qubit of Pauli Z after pauli_frame.flush op: "
                   << flushOp << "\n");

        auto z_parity = flushOp.getZParity();
        assert(z_parity && "invalid pauli_frame.flush op: missing z-parity return value");

        // Loop over all uses of the z-parity value (there should typically only be one, the scf.if
        // that contains the Pauli Z op). Once found, return its output qubit value.
        for (const auto &use : z_parity.getUses()) {
            auto *owner = use.getOwner();
            LLVM_DEBUG(llvm::dbgs()
                       << " -> visiting z-parity bit user op: " << owner->getName() << "\n");

            if (auto pauliZIfOp = llvm::dyn_cast<scf::IfOp>(owner)) {
                auto pauliZOutQubit = pauliZIfOp.getResult(0);
                assert(llvm::isa<QubitType>(pauliZOutQubit.getType()) &&
                       "expected return value of scf.if op to have type quantum.bit");
                LLVM_DEBUG(llvm::dbgs() << "   -> success; returning Pauli Z output qubit\n");
                return pauliZOutQubit;
            }
        }
        llvm_unreachable("failed to find output qubit of Pauli Z op after pauli_frame.flush op");
    }

    LogicalResult matchAndRewrite(MeasurementProcessOp op, PatternRewriter &rewriter) const override
    {
        LLVM_DEBUG(llvm::dbgs() << "Applying Pauli frame protocol to flush Pauli record before "
                                   "terminal measurement process: "
                                << op << "\n");
        auto loc = op->getLoc();

        auto obs = op.getObs();
        assert(obs && "invalid measurement process op: missing observable operand");

        auto obsOp = obs.getDefiningOp();

        // The flush op will be inserted before the observable op
        rewriter.setInsertionPoint(obsOp);

        // Helper function to insert the flush operations per qubit operand of the observable op.
        // If `qubit` has already had a flush op applied to it, do not flush again. Instead, update
        // input qubit operand of observable op to use the output qubit after the flush "block".
        auto insertFlushOpPerQubitOrSkip = [&](unsigned int idx, const Value qubit) {
            std::optional<FlushOp> flushOp = getFlushOpAppliedToQubit(qubit);
            if (!flushOp) {
                auto flushOp = FlushOp::create(rewriter, loc, rewriter.getI1Type(),
                                               rewriter.getI1Type(), qubit.getType(), qubit);
                auto pauliZOutQubit = insertPauliOpsAfterFlush(rewriter, loc, flushOp);
                rewriter.modifyOpInPlace(obsOp, [&] { obsOp->setOperand(idx, pauliZOutQubit); });
            }
            else {
                // Get output qubit of Pauli Z op after the flush
                auto pauliZOutQubit = getOutputQubitOfPauliZAfterFlush(flushOp.value());
                rewriter.modifyOpInPlace(obsOp, [&] { obsOp->setOperand(idx, pauliZOutQubit); });
            }
        };

        if (auto compBasisOp = dyn_cast<ComputationalBasisOp>(obsOp)) {
            auto qubits = compBasisOp.getQubits();
            for (const auto &[idx, qubit] : llvm::enumerate(qubits)) {
                insertFlushOpPerQubitOrSkip(idx, qubit);
            }
        }
        else if (auto namedObsOp = dyn_cast<NamedObsOp>(obsOp)) {
            insertFlushOpPerQubitOrSkip(0, namedObsOp.getQubit());
        }
        else {
            obsOp->emitError() << "Unsupported observable op: " << obsOp->getName();
            std::string msg = llvm::formatv(
                "failed to apply Pauli frame tracking protocols: unsupported observable: {0}",
                obsOp->getName());
            llvm_unreachable(msg.c_str());
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
    patterns.add<FlushBeforeMeasurementProcessPattern<ExpvalOp>>(patterns.getContext());
    patterns.add<FlushBeforeMeasurementProcessPattern<VarianceOp>>(patterns.getContext());
    patterns.add<FlushBeforeMeasurementProcessPattern<SampleOp>>(patterns.getContext());
    patterns.add<FlushBeforeMeasurementProcessPattern<CountsOp>>(patterns.getContext());
    patterns.add<FlushBeforeMeasurementProcessPattern<ProbsOp>>(patterns.getContext());
}

} // namespace pauli_frame
} // namespace catalyst
