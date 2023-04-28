// Copyright 2022-2023 Xanadu Quantum Technologies Inc.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <memory>
#include <vector>

#include "llvm/Support/Errc.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "Gradient/IR/GradientOps.h"

#include "Gradient/IR/GradientOps.h"
#include "Gradient/Transforms/Passes.h"
#include "Gradient/Transforms/Patterns.h"
#include "Quantum/IR/QuantumOps.h"

using namespace mlir;
using namespace catalyst::gradient;

namespace catalyst {
namespace gradient {

struct JVPLoweringPattern : public OpRewritePattern<JVPOp> {
    using OpRewritePattern<JVPOp>::OpRewritePattern;

    LogicalResult match(JVPOp op) const override;
    void rewrite(JVPOp op, PatternRewriter &rewriter) const override;
};

LogicalResult JVPLoweringPattern::match(JVPOp op) const
{
    llvm::errs() << "matched JVP op\n";
    return success();
}

void JVPLoweringPattern::rewrite(JVPOp op, PatternRewriter &rewriter) const
{
    Location loc = op.getLoc();
    llvm::errs() << "replacing JVP op\n";

    /* func::FuncOp callee = */
    /*     SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(op, op.getCalleeAttr()); */

    /* auto gradOp = rewriter.create<GradOp>(loc, */
    /*   op.getResultTypes(), */
    /*   op.getMethod(), */
    /*   op.getCallee(), */
    /*   op.getOperands(), */
    /*   op.getDiffArgIndices().value(), */
    /*   op.getFiniteDiffParam().value() */
    /*   ); */

    rewriter.replaceOpWithNewOp<GradOp>(op, 
      op.getResultTypes(),
      op.getMethod(),
      op.getCallee(),
      op.getOperands(),
      op.getDiffArgIndices().value(),
      op.getFiniteDiffParam().value()
    );

    llvm::errs() << "replaced JVP\n";
}



struct JVPLoweringPass
    : public PassWrapper<JVPLoweringPass, OperationPass<ModuleOp>> {

    JVPLoweringPass() {}

    StringRef getArgument() const override { return "lower-jvp-vjp"; }

    StringRef getDescription() const override { return "Lower JVP operations to grad operations."; }

    void getDependentDialects(DialectRegistry &registry) const override
    {
        registry.insert<bufferization::BufferizationDialect>();
        registry.insert<memref::MemRefDialect>();
    }

    void runOnOperation() final
    {
        llvm::errs() << "JVP lowering called\n";

        ModuleOp op = getOperation();

        RewritePatternSet patterns(&getContext());
        patterns.add<JVPLoweringPattern>(patterns.getContext(), 1);

        if (failed(applyPatternsAndFoldGreedily(op, std::move(patterns)))) {
            llvm::errs() << "JVP lowering failed\n";
            return signalPassFailure();
        }
        llvm::errs() << "JVP lowering succeeded\n";
    }
};

} // namespace gradient

std::unique_ptr<Pass> createJVPLoweringPass()
{
    return std::make_unique<gradient::JVPLoweringPass>();
}

} // namespace catalyst
