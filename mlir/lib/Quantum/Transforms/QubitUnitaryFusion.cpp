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

#define DEBUG_TYPE "qubit-unitary-fusion"

#include "mlir/Dialect/Linalg/Ops.h"

#include "llvm/Support/Debug.h"
#include "llvm/Support/Errc.h"

#include "Quantum/IR/QuantumOps.h"
#include "Quantum/Transforms/Patterns.h"

using llvm::dbgs;
using namespace mlir;
using namespace catalyst;
using namespace catalyst::quantum;

namespace {

struct QubitUnitaryFusionOpRewritePattern : public mlir::OpRewritePattern<CustomOp> {
    using mlir::OpRewritePattern<CustomOp>::OpRewritePattern;

    mlir::LogicalResult matchAndRewrite(CustomOp op, mlir::PatternRewriter &rewriter) const override
    {
        LLVM_DEBUG(dbgs() << "Simplifying the following hadamard operation:\n" << op << "\n");
        if (op.getGateName().str() != "Hadamard")
            return failure();

        ValueRange qbs = op.getInQubits();
        auto parentHadamard = dyn_cast<CustomOp>(qbs[0].getDefiningOp());

        if (parentHadamard == nullptr)
            return failure();

        if (parentHadamard.getGateName().str() != "Hadamard")
            return failure();

        Value simplifiedVal = parentHadamard.getInQubits()[0];
        rewriter.replaceOp(op, simplifiedVal);
        return success();
    }
};

struct QubitUnitaryFusion : public OpRewritePattern<QubitUnitaryOp>
{
    using mlir::OpRewritePattern<CustomOp>::OpRewritePattern;

    LogicalResult match(QubitUnitaryOp op)
    {
        ValueRange qbs = op.getInQubits();
        Operation *parent = qbs[0].getDefiningOp();

        if (!isa<QubitUnitaryOp>(parent))
            return failure();

        QubitUnitaryOp parentOp = cast<QubitUnitaryOp>(parent);
        ValueRange parentQbs = parentOp.getOutQubits();

        if (qbs.size() != parentQbs.size())
            return failure();

        for (auto [qb1, qb2] : llvm::zip(qbs, parentQbs))
            if (qb1 != qb2)
                return failure();

        return success();
    }

    void rewrite(QubitUnitaryOp op, PatternRewriter &rewriter)
    {
        ValueRange qbs = op.getInQubits();
        QubitUnitaryOp parentOp = cast<QubitUnitaryOp>(qbs[0].getDefiningOp());

        Value m1 = op.getMatrix();
        Value m2 = parentOp.getMatrix();

        linalg::MatmulOp matmul = rewriter.create<linalg::MatmulOp>(op.getLoc(), {m1, m2}, {});
        Value res = matmul.getResult(0);

        rewriter.updateRootInPlace(op, [&] {
            op->setOperand(0, res);
        });
        rewriter.replaceOp(parentOp, parentOp.getResults());
    }
};

} // namespace

namespace catalyst {
namespace quantum {

void populateQubitUnitaryFusionPatterns(RewritePatternSet &patterns)
{
    patterns.add<QubitUnitaryFusionOpRewritePattern>(patterns.getContext(), 1);
}

} // namespace quantum
} // namespace catalyst
