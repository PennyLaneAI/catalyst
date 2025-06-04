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

#define DEBUG_TYPE "mergeunitarygates"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/Debug.h"

#include "Catalyst/IR/CatalystDialect.h"
#include "Quantum/IR/QuantumOps.h"
#include "Quantum/Transforms/Patterns.h"

using namespace llvm;
using namespace mlir;
using namespace catalyst::quantum;

namespace catalyst {
namespace quantum {

#define GEN_PASS_DEF_MERGEUNITARYGATESPASS
#define GEN_PASS_DECL_MERGEUNITARYGATESPASS
#include "Quantum/Transforms/Passes.h.inc"

struct QubitUnitaryFusion : public OpRewritePattern<QubitUnitaryOp> {
    using mlir::OpRewritePattern<QubitUnitaryOp>::OpRewritePattern;

    // The above boilerplate instructs the pattern to be applied
    // to all operations of type `QubitUnitaryOp` in the input mlir

    LogicalResult matchAndRewrite(QubitUnitaryOp op, PatternRewriter &rewriter) const override
    {
        // Pattern matching logic
        ValueRange qbs = op.getInQubits();
        Operation *parent = qbs[0].getDefiningOp();

        // Parent should be a QubitUnitaryOp
        if (!isa<QubitUnitaryOp>(parent)) {
            return failure();
        }

        QubitUnitaryOp parentOp = cast<QubitUnitaryOp>(parent);
        ValueRange parentQbs = parentOp.getOutQubits();

        // Parent's output qubits should be the current op's input qubits,
        // and the qubits need to be in the same order
        if (qbs.size() != parentQbs.size()) {
            return failure();
        }

        for (auto [qb1, qb2] : llvm::zip(qbs, parentQbs)) {
            if (qb1 != qb2) {
                return failure();
            }
        }

        // Rewrite logic

        // In the tablegen definition of `QubitUnitaryOp`, there is a
        // field called `$matrix`, storing the matrix for the unitary gate.
        // Tablegen automatically generates getters for all of the fields.
        mlir::Value m1 = op.getMatrix();
        mlir::Value m2 = parentOp.getMatrix();

        // Get the type of a 2x2 complex matrix
        // Note that both m1 and m2 have this type already
        mlir::Type MatrixType = m1.getType();

        // Create the matrix multiplication operation
        // The linalg.matmul op's semantics is:
        //   linalg.matmul({A, B}, {C})
        // performs C+=A*B
        // so we need to create a zero matrix of the desired type and shape first
        tensor::EmptyOp zeromat =
            rewriter.create<tensor::EmptyOp>(op.getLoc(), MatrixType, ValueRange{});

        // The first argument to the `create` need to be a `Location`
        // which can usually just be a `getLoc()` from any operation you have handy
        // The second argument needs to be (a list of) type(s) of the operation's output
        // The third argument needs to be (a list of) input value(s) to the operation
        linalg::MatmulOp matmul = rewriter.create<linalg::MatmulOp>(
            op.getLoc(), TypeRange{MatrixType}, ValueRange{m1, m2}, ValueRange{zeromat});

        // Some peculiarity for the matmul operation; no need to worry about it here
        matmul->setAttr("operandSegmentSizes", rewriter.getDenseI32ArrayAttr({2, 1}));

        // Replace the matrix for the parent unitary (which is the first unitary op)
        // with the product matrix
        // Note: we need to move the zero matrix
        // and the matmul before the parent unitary
        // so all of them are defined before being used by the parent unitary
        zeromat->moveBefore(parentOp);
        matmul->moveBefore(parentOp);
        mlir::Value res = matmul.getResult(0);
        rewriter.modifyOpInPlace(parentOp, [&] { parentOp->setOperand(0, res); });

        // The second unitary is not needed anymore
        // Whoever uses the second unitary, use the first one instead!
        op.replaceAllUsesWith(parentOp);

        return success();
    }
};

struct MergeUnitaryGatesPass : public impl::MergeUnitaryGatesPassBase<MergeUnitaryGatesPass> {
    using impl::MergeUnitaryGatesPassBase<MergeUnitaryGatesPass>::MergeUnitaryGatesPassBase;

    void runOnOperation() override
    {
        // Get the current operation being operated on.
        // Default is the top-level module operation
        Operation *module = getOperation();
        MLIRContext *ctx = &getContext();

        // Define the set of patterns to use.
        RewritePatternSet quantumPatterns(ctx);
        quantumPatterns.add<QubitUnitaryFusion>(ctx);

        // Apply patterns in an iterative and greedy manner.
        // This visits all operation in the module recursively and apply the pattern
        if (failed(applyPatternsGreedily(module, std::move(quantumPatterns)))) {
            return signalPassFailure();
        }
    }
};

} // namespace quantum

std::unique_ptr<Pass> createMergeUnitaryGatesPass()
{
    return std::make_unique<MergeUnitaryGatesPass>();
}

} // namespace catalyst
