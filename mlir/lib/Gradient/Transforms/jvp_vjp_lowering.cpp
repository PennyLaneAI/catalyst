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

#define DEBUG_TYPE "jvpvjp"

#include <memory>
#include <vector>
#include <algorithm>

#include "llvm/Support/Errc.h"
#include "llvm/Support/Debug.h"

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
using llvm::dbgs;

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

template<class T>
std::vector<int64_t> _tovec(const T& x)
{
    return std::vector<int64_t>(x.begin(), x.end());
};

struct JVPLoweringPattern : public OpRewritePattern<JVPOp> { /*{{{*/
    using OpRewritePattern<JVPOp>::OpRewritePattern;

    LogicalResult match(JVPOp op) const override;
    void rewrite(JVPOp op, PatternRewriter &rewriter) const override;
};

LogicalResult JVPLoweringPattern::match(JVPOp op) const
{
    LLVM_DEBUG(dbgs() << "matched JVP op\n");
    return success();
}

void JVPLoweringPattern::rewrite(JVPOp op, PatternRewriter &rewriter) const
{
    MLIRContext *ctx = getContext();

    Location loc = op.getLoc();
    LLVM_DEBUG(dbgs() << "replacing VJP op\n");

    auto func_diff_operand_indices = GradOp::compDiffArgIndices(op.getDiffArgIndices());
    LLVM_DEBUG(dbgs() << "func_diff_operand_indices: " << func_diff_operand_indices << " \n");
    LLVM_DEBUG(dbgs() << "jvp_num_operands " << op.getOperands().size() << " \n");
    assert(func_diff_operand_indices.size() <= op.getOperands().size()/2);

    size_t func_operands_size = op.getOperands().size() - func_diff_operand_indices.size();
    size_t tang_operands_size = func_diff_operand_indices.size();

    auto func_operands = OperandRange(op.operand_begin(), op.operand_begin() + func_operands_size);
    auto tang_operands = OperandRange(op.operand_begin() + tang_operands_size, op.operand_end());

    for(auto idx: func_diff_operand_indices) {
      assert(idx < func_operands.size() && "all diffArgIndices reference valid arguments");
    }

    auto res_halfsize = (op.result_type_end() - op.result_type_begin()) / 2;
    auto func_operand_types = ({
      std::vector<Type> out;
      for(auto o: func_operands) out.push_back(o.getType());
      out;
    });
    auto func_result_types = ValueTypeRange<ResultRange>(op.result_type_begin(), op.result_type_begin() + res_halfsize);

    auto func_op = SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(op, op.getCalleeAttr());

    auto grad_result_types = computeResultTypes(func_op, func_diff_operand_indices);
    LLVM_DEBUG(dbgs() << "grad_result_types: " << grad_result_types << " \n");
    assert(grad_result_types.size() == func_diff_operand_indices.size()*func_result_types.size() &&
      "GradOp does't seem to return a tuple of Jacobians");

    auto fcall_op =
      rewriter.create<func::CallOp>(loc, func_op, func_operands);


    auto grad_op = rewriter.create<GradOp>(
      loc,
      grad_result_types,
      op.getMethod(),
      op.getCallee(),
      func_operands,
      op.getDiffArgIndicesAttr(),
      op.getFiniteDiffParamAttr()
    );

    std::vector<Value> einsum_results;
    for(size_t nout = 0; nout < func_result_types.size(); nout++) {
      Optional<Value> acc;
      for(size_t nparam = 0; nparam < func_diff_operand_indices.size(); nparam++) {
        auto jac = grad_op.getResults()[nparam*func_diff_operand_indices.size() + nout];
        auto tang = tang_operands[nparam];
        auto param = func_operands[func_diff_operand_indices[nparam]];

        auto sjac = _tovec(jac.getType().cast<mlir::TensorType>().getShape());
        auto sparam = _tovec(param.getType().cast<mlir::TensorType>().getShape());
        auto stang = _tovec(tang.getType().cast<mlir::TensorType>().getShape());

        auto sjac_param = ({
          std::vector<int64_t> out(sjac.begin(), sjac.begin()+std::min(sjac.size(),sparam.size()));
          out;
        });

        LLVM_DEBUG(dbgs() << "jac_type " << sjac << "\n");
        LLVM_DEBUG(dbgs() << "param_type " << sparam << "\n");
        LLVM_DEBUG(dbgs() << "tang_type " << stang << "\n");

        assert(sparam == stang && "Parameter and tanget shapes don't match");
        assert(sjac_param == sparam && "Jacobian shape doesn't contain the parameter shape as a prefix");

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

        LLVM_DEBUG(dbgs() << "jac_axis " << jac_axis_names << "\n");
        LLVM_DEBUG(dbgs() << "tang_axis " << tang_axis_names << "\n");
        LLVM_DEBUG(dbgs() << "jvp_axis " << jvp_axis_names << "\n");

        /* tjac.getShape(); */

        auto res = einsumLinalgGeneric(rewriter, loc,
          jac_axis_names, tang_axis_names, jvp_axis_names,
          jac, tang);

        LLVM_DEBUG(dbgs() << "jvp result type " << res.getType() << "\n");

        if(!acc.has_value()) {
          acc = res;
        }
        else {
          assert(acc.value().getType() == res.getType());

          auto add_op = rewriter.create<linalg::ElemwiseBinaryOp>(
            loc,
            res.getType(), ValueRange({acc.value(), res}), acc.value(),
            linalg::BinaryFnAttr::get(ctx, linalg::BinaryFn::add),
            linalg::TypeFnAttr::get(ctx,linalg::TypeFn::cast_signed)
            );
          acc = add_op.getResultTensors()[0];
        }
      }
      assert(acc.has_value());
      einsum_results.push_back(acc.value());
    }

    auto results = ({
      std::vector<Value> out;
      out.insert(out.end(), fcall_op.getResults().begin(), fcall_op.getResults().end());
      out.insert(out.end(), einsum_results.begin(), einsum_results.end());
      out;
    });

    rewriter.replaceOp(op, results);

    LLVM_DEBUG(dbgs() << "replaced JVP\n");
}
/*}}}*/

struct VJPLoweringPattern : public OpRewritePattern<VJPOp> { /*{{{*/
    using OpRewritePattern<VJPOp>::OpRewritePattern;

    LogicalResult match(VJPOp op) const override;
    void rewrite(VJPOp op, PatternRewriter &rewriter) const override;
};

LogicalResult VJPLoweringPattern::match(VJPOp op) const
{
    LLVM_DEBUG(dbgs() << "matched VJP op\n");
    return success();
}

void VJPLoweringPattern::rewrite(VJPOp op, PatternRewriter &rewriter) const
{
    MLIRContext *ctx = getContext();

    Location loc = op.getLoc();
    LLVM_DEBUG(dbgs() << "replacing VJP op\n");

    auto func_diff_operand_indices = GradOp::compDiffArgIndices(op.getDiffArgIndices());
    LLVM_DEBUG(dbgs() << "func_diff_operand_indices: " << func_diff_operand_indices << " \n");
    LLVM_DEBUG(dbgs() << "vjp_num_operands " << op.getOperands().size() << " \n");
    assert(func_diff_operand_indices.size() <= op.getOperands().size()/2);

    auto func_op = SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(op, op.getCalleeAttr());

    size_t func_operands_size = func_op.getFunctionType().getNumInputs();
    size_t cotang_operands_size = op.getOperands().size() - func_operands_size;
    assert(func_op.getFunctionType().getNumResults() == cotang_operands_size &&
        "the number of function results doesn't match the number of cotangent arguments");

    auto func_operands = OperandRange(op.operand_begin(), op.operand_begin() + func_operands_size);
    auto cotang_operands = OperandRange(op.operand_begin() + cotang_operands_size, op.operand_end());

    for(auto idx: func_diff_operand_indices) {
      assert(idx < func_operands.size() && "all diffArgIndices reference valid arguments");
    }

    auto func_operand_types = ({
      std::vector<Type> out;
      for(auto o: func_operands) out.push_back(o.getType());
      out;
    });
    auto func_result_types = func_op.getResultTypes();

    auto grad_result_types = computeResultTypes(func_op, func_diff_operand_indices);
    LLVM_DEBUG(dbgs() << "grad_result_types: " << grad_result_types << " \n");
    assert(grad_result_types.size() == func_diff_operand_indices.size()*func_result_types.size() &&
      "GradOp does't seem to return a tuple of Jacobians");

    auto fcall_op =
      rewriter.create<func::CallOp>(loc, func_op, func_operands);

    auto grad_op = rewriter.create<GradOp>(
      loc,
      grad_result_types,
      op.getMethod(),
      op.getCallee(),
      func_operands,
      op.getDiffArgIndicesAttr(),
      op.getFiniteDiffParamAttr()
    );

    std::vector<Value> einsum_results;
    for(size_t nparam = 0; nparam < func_diff_operand_indices.size(); nparam++) {
      Optional<Value> acc;
      for(size_t nout = 0; nout < func_result_types.size(); nout++) {
        auto jac = grad_op.getResults()[nparam*func_diff_operand_indices.size() + nout];
        auto param = func_operands[func_diff_operand_indices[nparam]];
        auto cotang = cotang_operands[nout];

        auto sjac = _tovec(jac.getType().cast<mlir::TensorType>().getShape());
        auto sparam = _tovec(param.getType().cast<mlir::TensorType>().getShape());
        auto scotang = _tovec(cotang.getType().cast<mlir::TensorType>().getShape());

        auto sjac_cotang = ({
          std::vector<int64_t> out(sjac.begin()+sparam.size(), sjac.end());
          out;
        });

        LLVM_DEBUG(dbgs() << "jac_type " << sjac << "\n");
        LLVM_DEBUG(dbgs() << "param_type " << sparam << "\n");
        LLVM_DEBUG(dbgs() << "cotang_type " << scotang << "\n");

        assert(sjac_cotang == scotang && "Jacobian shape doesn't contain the cotang shape as a suffix");

        auto jac_axis_names = ({
          std::vector<size_t> out;
          for(size_t i=0; i<sjac.size(); i++) out.push_back(i);
          out;
        });
        auto cotang_axis_names = ({
          std::vector<size_t> out;
          for(size_t i=sjac.size()-scotang.size(); i<sjac.size(); i++) out.push_back(i);
          out;
        });
        auto vjp_axis_names = ({
          std::vector<size_t> out;
          for(size_t i=0; i<sjac.size()-scotang.size(); i++) out.push_back(i);
          out;
        });

        LLVM_DEBUG(dbgs() << "jac_axis " << jac_axis_names << "\n");
        LLVM_DEBUG(dbgs() << "cotang_axis " << cotang_axis_names << "\n");
        LLVM_DEBUG(dbgs() << "vjp_axis " << vjp_axis_names << "\n");

        auto res = einsumLinalgGeneric(rewriter, loc,
          cotang_axis_names, jac_axis_names, vjp_axis_names,
          cotang, jac);

        LLVM_DEBUG(dbgs() << "vjp result type " << res.getType() << "\n");

        if(!acc.has_value()) {
          acc = res;
        }
        else {
          assert(acc.value().getType() == res.getType());

          auto add_op = rewriter.create<linalg::ElemwiseBinaryOp>(
            loc,
            res.getType(), ValueRange({acc.value(), res}), acc.value(),
            linalg::BinaryFnAttr::get(ctx, linalg::BinaryFn::add),
            linalg::TypeFnAttr::get(ctx,linalg::TypeFn::cast_signed)
            );
          acc = add_op.getResultTensors()[0];
        }
      }
      assert(acc.has_value());
      einsum_results.push_back(acc.value());
    }

    auto results = ({
      std::vector<Value> out;
      out.insert(out.end(), fcall_op.getResults().begin(), fcall_op.getResults().end());
      out.insert(out.end(), einsum_results.begin(), einsum_results.end());
      /* out.insert(out.end(), func_operands.begin(), func_operands.end()); */
      out;
    });

    rewriter.replaceOp(op, results);

    LLVM_DEBUG(dbgs() << "replaced VJP\n");
}
/*}}}*/


struct JVPVJPLoweringPass
    : public PassWrapper<JVPVJPLoweringPass, OperationPass<ModuleOp>> {

    JVPVJPLoweringPass() {}

    StringRef getArgument() const override { return "lower-jvpvjp"; }

    StringRef getDescription() const override
    {
      return "Lower JVP/VJP operations down to grad and linalg.generic operations.";
    }

    void getDependentDialects(DialectRegistry &registry) const override
    {
        // FIXME: What about Linalg dialect?
        registry.insert<bufferization::BufferizationDialect>();
        registry.insert<memref::MemRefDialect>();
    }

    void runOnOperation() final
    {
        ModuleOp op = getOperation();

        RewritePatternSet patterns(&getContext());
        patterns.add<JVPLoweringPattern>(patterns.getContext(), 1);
        patterns.add<VJPLoweringPattern>(patterns.getContext(), 1);

        if (failed(applyPatternsAndFoldGreedily(op, std::move(patterns)))) {
            return signalPassFailure();
        }
    }
};

} // namespace gradient

std::unique_ptr<Pass> createJVPVJPLoweringPass()
{
    return std::make_unique<gradient::JVPVJPLoweringPass>();
}

} // namespace catalyst
