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
#include "Gradient/Utils/EinsumLinalgGeneric.h"
#include "Gradient/Utils/GradientShape.h"

#include "Gradient/IR/GradientOps.h"
#include "Gradient/Transforms/Passes.h"
#include "Gradient/Transforms/Patterns.h"
#include "Quantum/IR/QuantumOps.h"

using namespace mlir;
using namespace catalyst::gradient;

namespace llvm {

  template<class T>
  raw_ostream& operator<<(raw_ostream& oss, const std::vector<T> &v) {
    oss << "[";
    bool first = true;
    for(auto i : v) {
      oss << (first ? "" : ", ") << i;
      first = false;
    }
    oss << "]";
    return oss;
  }
};


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

    auto func_op = rewriter.create<func::FuncOp>(loc,
      op.getCallee().str(),
      rewriter.getFunctionType(op.getOperandTypes(), func_result_types),
      rewriter.getStringAttr("private"));

    auto fcall_op = rewriter.create<func::CallOp>(loc, func_op, func_operands);

    auto grad_result_types = computeResultTypes(func_op,
      GradOp::compDiffArgIndices(op.getDiffArgIndices()));

    llvm::errs() << func_op.getSymName() << " grad result type: " << grad_result_types << " \n";

    assert(grad_result_types.size() == func_operands.size());

    /* llvm::errs() << "calling " << fnName << " \n"; */
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

    auto grad_op = rewriter.create<GradOp>(
      loc,
      grad_result_types,
      op.getMethod(),
      op.getCallee(),
      func_operands,
      op.getDiffArgIndices().value(),
      op.getFiniteDiffParam().value()
    );

    auto _tovec = [](auto x) -> std::vector<int64_t> {
        std::vector<int64_t> out(x.begin(), x.end());
        return out;
    };

    for(size_t ntang = 0; ntang < tang_operands.size(); ntang++) {
      for(size_t nparam = 0; nparam < func_operands.size(); nparam++) {
        auto jac = grad_op.getResults()[nparam];
        auto tang = tang_operands[ntang];
        auto param = func_operands[nparam];

        auto sjac = _tovec(jac.getType().cast<mlir::TensorType>().getShape());
        auto sparam = _tovec(param.getType().cast<mlir::TensorType>().getShape());
        auto stang = _tovec(tang.getType().cast<mlir::TensorType>().getShape());

        auto sjac_param = ({
          std::vector<int64_t> out(sjac.begin(), sjac.begin()+std::min(sjac.size(),sparam.size()));
          out;
        });

        llvm::errs() << "jac_type " << sjac << "\n";
        llvm::errs() << "param_type " << sparam << "\n";
        llvm::errs() << "tang_type " << stang << "\n";

        assert(sparam == stang && "Parameter and tanget shapes don't match");
        assert(sjac_param == sparam && "Jacobian shape doesn't start from parameter shape");

        auto jac_axis_names = ({
          std::vector<size_t> out;
          for(size_t i=0; i<sjac.size(); i++) out.push_back(i);
          out;
        });
        auto tang_axis_names = ({
          std::vector<size_t> out;
          for(size_t i=0; i<stang.size(); i++) out.push_back(i);
          out;
        });
        auto jvp_axis_names = ({
          std::vector<size_t> out;
          for(size_t i=0; i<sjac.size()-sparam.size(); i++) out.push_back(i+tang_axis_names.size());
          out;
        });

        llvm::errs() << "jac_axis " << jac_axis_names << "\n";
        llvm::errs() << "tang_axis " << tang_axis_names << "\n";
        llvm::errs() << "jvp_axis " << jvp_axis_names << "\n";

        /* tjac.getShape(); */

        auto res = einsumLinalgGeneric(rewriter, loc,
          jac_axis_names, tang_axis_names, jvp_axis_names,
          jac, tang);
      }
    }

    std::vector<Value> mock_results;
    mock_results.reserve(2 * fcall_op.getResults().size());
    mock_results.insert(mock_results.end(), fcall_op.getResults().begin(), fcall_op.getResults().end());
    mock_results.insert(mock_results.end(), fcall_op.getResults().begin(), fcall_op.getResults().end());

    llvm::errs() << "fcall_op.result.size(): " << fcall_op.getResults().size() << "\n";
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
