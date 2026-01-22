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

#include <type_traits>

#include "llvm/Support/MathExtras.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Builders.h"

#include "QEC/IR/QECOps.h"
#include "QEC/Transforms/Patterns.h"
#include "Quantum/IR/QuantumOps.h"

using namespace mlir;
using namespace catalyst::qec;
using namespace catalyst::quantum;

namespace {

/// Get the state vector values for a given LogicalInitKind
/// Returns {re0, im0, re1, im1} for a single qubit state
std::tuple<double, double, double, double> getStateVectorValues(LogicalInitKind initState)
{
    constexpr double sqrt2_inv = 1.0 / llvm::numbers::sqrt2; // 1/√2

    switch (initState) {
    case LogicalInitKind::zero:
        return {1.0, 0.0, 0.0, 0.0};
    case LogicalInitKind::one:
        return {0.0, 0.0, 1.0, 0.0};
    case LogicalInitKind::plus:
        return {sqrt2_inv, 0.0, sqrt2_inv, 0.0};
    case LogicalInitKind::minus:
        return {sqrt2_inv, 0.0, -sqrt2_inv, 0.0};
    case LogicalInitKind::plus_i:
        return {sqrt2_inv, 0.0, 0.0, sqrt2_inv};
    case LogicalInitKind::minus_i:
        return {sqrt2_inv, 0.0, 0.0, -sqrt2_inv};
    case LogicalInitKind::magic: // T gate
        // |m⟩ = (|0⟩ + e^{iπ/4}|1⟩) / √2 = [1/√2, (1+i)/(2)]
        return {sqrt2_inv, 0.0, 0.5, 0.5};
    case LogicalInitKind::magic_conj: // T† gate
        // |m̅⟩ = (|0⟩ + e^{-iπ/4}|1⟩) / √2 = [1/√2, (1-i)/(2)]
        return {sqrt2_inv, 0.0, 0.5, -0.5};
    }
    llvm_unreachable("Unknown LogicalInitKind");
}

/// Create a dense tensor constant for a single qubit state vector
Value createStateVectorTensor(Location loc, PatternRewriter &rewriter, LogicalInitKind initState)
{
    auto ctx = rewriter.getContext();
    auto f64Type = Float64Type::get(ctx);
    auto complexType = ComplexType::get(f64Type);
    auto tensorType = RankedTensorType::get({2}, complexType);

    auto [re0, im0, re1, im1] = getStateVectorValues(initState);

    auto denseAttr = DenseElementsAttr::get(
        tensorType, ArrayRef<std::complex<double>>{std::complex<double>(re0, im0),
                                                   std::complex<double>(re1, im1)});

    return rewriter.create<arith::ConstantOp>(loc, denseAttr);
}

/// Template pattern to lower QEC init ops (PrepareStateOp, FabricateOp) to SetStateOp
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
                auto allocOp = rewriter.create<AllocQubitOp>(loc);
                qubit = allocOp.getResult();
            }

            Value stateVector = createStateVectorTensor(loc, rewriter, initState);
            auto setStateOp = rewriter.create<SetStateOp>(loc, qubit.getType(), stateVector, qubit);
            resultQubits.push_back(setStateOp.getOutQubits().front());
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
