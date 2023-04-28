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
#include <algorithm>

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

    size_t op_halfsize = (op.operand_end() - op.operand_begin()) / 2;
    auto func_operands = OperandRange(op.operand_begin(), op.operand_begin() + op_halfsize);
    auto tang_operands = OperandRange(op.operand_begin() + op_halfsize, op.operand_end());

    auto res_halfsize = (op.result_type_end() - op.result_type_begin()) / 2;
    auto func_result_types = ValueTypeRange<ResultRange>(op.result_type_begin(), op.result_type_begin() + res_halfsize);

    std::string fnName = op.getCallee().str();
    FunctionType fnType = rewriter.getFunctionType(op.getOperandTypes(), func_result_types);
    StringAttr visibility = rewriter.getStringAttr("private");
    auto funcOp = rewriter.create<func::FuncOp>(loc, fnName, fnType, visibility);
    auto fcallOp = rewriter.create<func::CallOp>(loc, funcOp, func_operands);

    llvm::errs() << "calling " << fnName << " \n";

    /* auto gradOp = rewriter.create<GradOp>(loc, */
    /*   op.getResultTypes(), */
    /*   op.getMethod(), */
    /*   op.getCallee(), */
    /*   op.getOperands(), */
    /*   op.getDiffArgIndices().value(), */
    /*   op.getFiniteDiffParam().value() */
    /*   ); */

    /* auto res_type_size = op.result_type_end() - op.result_type_begin(); */
    /* auto res_type_range = ValueTypeRange<ResultRange>(op.result_type_begin(), op.result_type_begin()+res_type_size/2); */
    /* /1* llvm::errs() << "replaced JVP op_size: " << op_size << "\n"; *1/ */

    auto gradOp = rewriter.create<GradOp>(
      loc,
      op.getResultTypes(),
      op.getMethod(),
      op.getCallee(),
      func_operands,
      op.getDiffArgIndices().value(),
      op.getFiniteDiffParam().value()
    );

    for(auto t: tang_operands) {
      for(auto j: gradOp.getResults()) {
        llvm::errs() << "emitting a tensordot" << "\n";
      }
    }

    std::vector<Value> mock_results;
    mock_results.reserve(2 * fcallOp.getResults().size());
    mock_results.insert(mock_results.end(), fcallOp.getResults().begin(), fcallOp.getResults().end());
    mock_results.insert(mock_results.end(), fcallOp.getResults().begin(), fcallOp.getResults().end());

    llvm::errs() << "fcallOp.result.size(): " << fcallOp.getResults().size() << "\n";
    llvm::errs() << "mock_results.size(): " << mock_results.size() << "\n";

    rewriter.replaceOp(op, mock_results);

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
