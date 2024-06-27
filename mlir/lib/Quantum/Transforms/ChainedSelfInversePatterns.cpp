// Copyright 2023 Xanadu Quantum Technologies Inc.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#define DEBUG_TYPE "chained-self-inverse"

#include "llvm/Support/Debug.h"
#include "llvm/Support/Errc.h"

#include "Quantum/IR/QuantumOps.h"
#include "Quantum/Transforms/Patterns.h"

using llvm::dbgs;
using namespace mlir;
using namespace catalyst;
using namespace catalyst::quantum;

namespace {

// We check if the operation and it's parent are a pair of self-inverse operations.
// If so, replace op with it's "grandparent".
// e.g.
//    %1 = (a qubit value produced from somewhere)
//    %out_qubits_1 = quantum.custom "PauliZ"() %1 : !quantum.bit
//    %out_qubits_2 = quantum.custom "PauliZ"() %out_qubits_1 : !quantum.bit
//    %2 = (a user operation that uses %out_qubit_2)
// The Value %out_qubits_2 can be replaced by the Value %1 (which is its grandparent) in all uses

mlir::LogicalResult matchAndRewriteImpl(CustomOp op, mlir::PatternRewriter &rewriter,
                                        StringRef OpName)
{
    LLVM_DEBUG(dbgs() << "Simplifying the following " << OpName << " operation:\n" << op << "\n");
    if (op.getGateName().str() != OpName)
        return failure();

    ValueRange qbs = op.getInQubits();
    auto parentOp = dyn_cast<CustomOp>(qbs[0].getDefiningOp());

    if (parentOp == nullptr) {
        return failure();
    }

    if (parentOp.getGateName().str() != OpName) {
        return failure();
    }

    // for multiple qubit gates, the wires need to be in order
    // since the cancelled inverses need to have matching control/target wires
    // e.g. The following pair of neighbouring CNOTs should NOT be eliminated
    //    %out_qubits:2 = quantum.custom "CNOT"() %1, %2 : !quantum.bit, !quantum.bit
    //    %out_qubits_0:2 = quantum.custom "CNOT"() %out_qubits#1, %out_qubits#0 : !quantum.bit,
    //    !quantum.bit
    if (qbs.size() > 1) {
        for (size_t i = 0; i < qbs.size(); i++) {
            if ((parentOp.getOutQubits())[i] != qbs[i]) {
                return failure();
            }
        }
    }

    ValueRange simplifiedVal = parentOp.getInQubits();
    rewriter.replaceOp(op, simplifiedVal);
    return success();
}

struct ChainedPauliXOpRewritePattern : public mlir::OpRewritePattern<CustomOp> {
    using mlir::OpRewritePattern<CustomOp>::OpRewritePattern;

    mlir::LogicalResult matchAndRewrite(CustomOp op, mlir::PatternRewriter &rewriter) const override
    {
        return matchAndRewriteImpl(op, rewriter, "PauliX");
    }
};

struct ChainedPauliYOpRewritePattern : public mlir::OpRewritePattern<CustomOp> {
    using mlir::OpRewritePattern<CustomOp>::OpRewritePattern;

    mlir::LogicalResult matchAndRewrite(CustomOp op, mlir::PatternRewriter &rewriter) const override
    {
        return matchAndRewriteImpl(op, rewriter, "PauliY");
    }
};

struct ChainedPauliZOpRewritePattern : public mlir::OpRewritePattern<CustomOp> {
    using mlir::OpRewritePattern<CustomOp>::OpRewritePattern;

    mlir::LogicalResult matchAndRewrite(CustomOp op, mlir::PatternRewriter &rewriter) const override
    {
        return matchAndRewriteImpl(op, rewriter, "PauliZ");
    }
};

struct ChainedHadamardOpRewritePattern : public mlir::OpRewritePattern<CustomOp> {
    using mlir::OpRewritePattern<CustomOp>::OpRewritePattern;

    mlir::LogicalResult matchAndRewrite(CustomOp op, mlir::PatternRewriter &rewriter) const override
    {
        return matchAndRewriteImpl(op, rewriter, "Hadamard");
    }
};

struct ChainedCNOTOpRewritePattern : public mlir::OpRewritePattern<CustomOp> {
    using mlir::OpRewritePattern<CustomOp>::OpRewritePattern;

    mlir::LogicalResult matchAndRewrite(CustomOp op, mlir::PatternRewriter &rewriter) const override
    {
        return matchAndRewriteImpl(op, rewriter, "CNOT");
    }
};

struct ChainedCYOpRewritePattern : public mlir::OpRewritePattern<CustomOp> {
    using mlir::OpRewritePattern<CustomOp>::OpRewritePattern;

    mlir::LogicalResult matchAndRewrite(CustomOp op, mlir::PatternRewriter &rewriter) const override
    {
        return matchAndRewriteImpl(op, rewriter, "CY");
    }
};

struct ChainedCZOpRewritePattern : public mlir::OpRewritePattern<CustomOp> {
    using mlir::OpRewritePattern<CustomOp>::OpRewritePattern;

    mlir::LogicalResult matchAndRewrite(CustomOp op, mlir::PatternRewriter &rewriter) const override
    {
        return matchAndRewriteImpl(op, rewriter, "CZ");
    }
};

struct ChainedSWAPOpRewritePattern : public mlir::OpRewritePattern<CustomOp> {
    using mlir::OpRewritePattern<CustomOp>::OpRewritePattern;

    mlir::LogicalResult matchAndRewrite(CustomOp op, mlir::PatternRewriter &rewriter) const override
    {
        return matchAndRewriteImpl(op, rewriter, "SWAP");
    }
};

struct ChainedToffoliOpRewritePattern : public mlir::OpRewritePattern<CustomOp> {
    using mlir::OpRewritePattern<CustomOp>::OpRewritePattern;

    mlir::LogicalResult matchAndRewrite(CustomOp op, mlir::PatternRewriter &rewriter) const override
    {
        return matchAndRewriteImpl(op, rewriter, "Toffoli");
    }
};

} // namespace

namespace catalyst {
namespace quantum {

void populateSelfInversePatterns(RewritePatternSet &patterns)
{
    patterns.add<ChainedPauliXOpRewritePattern>(patterns.getContext(), 1);
    patterns.add<ChainedPauliYOpRewritePattern>(patterns.getContext(), 1);
    patterns.add<ChainedPauliZOpRewritePattern>(patterns.getContext(), 1);
    patterns.add<ChainedHadamardOpRewritePattern>(patterns.getContext(), 1);
    patterns.add<ChainedCNOTOpRewritePattern>(patterns.getContext(), 1);
    patterns.add<ChainedCYOpRewritePattern>(patterns.getContext(), 1);
    patterns.add<ChainedCZOpRewritePattern>(patterns.getContext(), 1);
    patterns.add<ChainedSWAPOpRewritePattern>(patterns.getContext(), 1);
    patterns.add<ChainedToffoliOpRewritePattern>(patterns.getContext(), 1);
}

} // namespace quantum
} // namespace catalyst
