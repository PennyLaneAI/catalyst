// Copyright 2026 Xanadu Quantum Technologies Inc.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#define DEBUG_TYPE "lower-qec-init-ops"

#include "QEC/IR/QECOps.h"
#include "QEC/Transforms/Patterns.h"
#include "Quantum/IR/QuantumOps.h"

using namespace mlir;
using namespace catalyst::qec;
using namespace catalyst::quantum;

namespace {

/// Create a single-qubit gate using CustomOp
/// Returns the output qubit from the gate
Value createGate(Location loc, PatternRewriter &rewriter, Value inQubit, StringRef gateName,
                 bool adjoint = false)
{
    auto outQubitType = inQubit.getType();
    auto gateOp = CustomOp::create(rewriter, loc,
                                   /*out_qubits=*/TypeRange{outQubitType},
                                   /*out_ctrl_qubits=*/TypeRange{},
                                   /*params=*/ValueRange{},
                                   /*in_qubits=*/ValueRange{inQubit},
                                   /*gate_name=*/gateName,
                                   /*adjoint=*/adjoint,
                                   /*in_ctrl_qubits=*/ValueRange{},
                                   /*in_ctrl_values=*/ValueRange{});
    return gateOp.getOutQubits().front();
}

/// Apply the gates required to prepare the given state from |0⟩
/// Returns the final qubit after all gates are applied
Value applyStatePreparationGates(Location loc, PatternRewriter &rewriter, Value qubit,
                                 LogicalInitKind initState)
{
    switch (initState) {
    case LogicalInitKind::zero: // |0⟩ - no gates needed
        return qubit;
    case LogicalInitKind::one: // |1⟩ = X|0⟩
        return createGate(loc, rewriter, qubit, "PauliX");
    case LogicalInitKind::plus: // |+⟩ = H|0⟩
        return createGate(loc, rewriter, qubit, "Hadamard");
    case LogicalInitKind::minus: // |−⟩ = ZH|0⟩
        qubit = createGate(loc, rewriter, qubit, "Hadamard");
        return createGate(loc, rewriter, qubit, "PauliZ");
    case LogicalInitKind::plus_i: // |+i⟩ = SH|0⟩
        qubit = createGate(loc, rewriter, qubit, "Hadamard");
        return createGate(loc, rewriter, qubit, "S");
    case LogicalInitKind::minus_i: // |−i⟩ = S†H|0⟩
        qubit = createGate(loc, rewriter, qubit, "Hadamard");
        return createGate(loc, rewriter, qubit, "S", /*adjoint=*/true);
    case LogicalInitKind::magic: // |m⟩ = TH|0⟩
        qubit = createGate(loc, rewriter, qubit, "Hadamard");
        return createGate(loc, rewriter, qubit, "T");
    case LogicalInitKind::magic_conj: // |m̅⟩ = T†H|0⟩
        qubit = createGate(loc, rewriter, qubit, "Hadamard");
        return createGate(loc, rewriter, qubit, "T", /*adjoint=*/true);
    }
    llvm_unreachable("Unknown LogicalInitKind");
}

/// Template pattern to lower QEC init ops (PrepareStateOp, FabricateOp) to alloc + gates
/// - PrepareStateOp: uses existing input qubits
/// - FabricateOp: allocates new qubits via AllocQubitOp
template <typename OpType> struct LowerQECInitOpPattern : public OpRewritePattern<OpType> {
    using OpRewritePattern<OpType>::OpRewritePattern;

    LogicalResult matchAndRewrite(OpType op, PatternRewriter &rewriter) const override
    {
        Location loc = op.getLoc();
        auto initState = op.getInitState();

        SmallVector<Value> resultQubits;

        size_t numQubits = op.getOutQubits().size();
        for (size_t i = 0; i < numQubits; ++i) {
            Value qubit;
            if constexpr (std::is_same_v<OpType, PrepareStateOp>) {
                // PrepareStateOp: use existing input qubits
                qubit = op.getInQubits()[i];
            }
            else {
                // FabricateOp: allocate a new qubit
                auto allocOp = AllocQubitOp::create(rewriter, loc);
                qubit = allocOp.getResult();
            }

            // Apply the appropriate gates to prepare the desired state
            Value resultQubit = applyStatePreparationGates(loc, rewriter, qubit, initState);
            resultQubits.push_back(resultQubit);
        }

        rewriter.replaceOp(op, resultQubits);
        return success();
    }
};

} // namespace

namespace catalyst {
namespace qec {

void populateLowerQECInitOpsPatterns(RewritePatternSet &patterns)
{
    patterns.add<LowerQECInitOpPattern<PrepareStateOp>>(patterns.getContext());
    patterns.add<LowerQECInitOpPattern<FabricateOp>>(patterns.getContext());
}

} // namespace qec
} // namespace catalyst
