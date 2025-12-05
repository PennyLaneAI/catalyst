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

#include "Quantum/IR/QuantumDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"

// #include "Catalyst/Utils/EnsureFunctionDeclaration.h"
#include "PauliFrame/IR/PauliFrameOps.h"
#include "PauliFrame/Transforms/Patterns.h"
#include "Quantum/IR/QuantumOps.h"
#include <mlir/IR/Builders.h>
#include <mlir/IR/Location.h>

using namespace mlir;

namespace {

using namespace catalyst::pauli_frame;
using namespace catalyst::quantum;

//===----------------------------------------------------------------------===//
// Helper functions
//===----------------------------------------------------------------------===//

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

//===----------------------------------------------------------------------===//
// Gate-conversion functions
//===----------------------------------------------------------------------===//

// TODO
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

// TODO
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

// TODO
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

    auto pauliZOutQubit = pauliZIfOp->getResult(0);

    op->setOperands(pauliZOutQubit);

    return success();
}

//===----------------------------------------------------------------------===//
// Clifford+T to Pauli Frame Patterns
//===----------------------------------------------------------------------===//

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

} // namespace

namespace catalyst {
namespace pauli_frame {

void populateCliffordTToPauliFramePatterns(RewritePatternSet &patterns)
{
    patterns.add<CliffordTToPauliFramePattern>(patterns.getContext());
}

} // namespace pauli_frame
} // namespace catalyst
