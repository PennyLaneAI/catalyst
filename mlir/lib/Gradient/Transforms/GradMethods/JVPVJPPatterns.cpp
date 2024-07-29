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

#define DEBUG_TYPE "jvpvjp"

#include <algorithm>
#include <memory>
#include <vector>

#include "llvm/Support/Debug.h"
#include "llvm/Support/Errc.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/SymbolTable.h"

#include "Gradient/Utils/EinsumLinalgGeneric.h"
#include "Gradient/Utils/GradientShape.h"
#include "Quantum/IR/QuantumOps.h"

#include "JVPVJPPatterns.hpp"

using namespace mlir;
using namespace catalyst::gradient;
using llvm::dbgs;

namespace llvm {

template <class T> raw_ostream &operator<<(raw_ostream &oss, const std::vector<T> &v)
{
    oss << "[";
    bool first = true;
    for (auto i : v) {
        oss << (first ? "" : ", ") << i;
        first = false;
    }
    oss << "]";
    return oss;
}
}; // namespace llvm

namespace catalyst {
namespace gradient {

template <class T> std::vector<int64_t> _tovec(const T &x)
{
    return std::vector<int64_t>(x.begin(), x.end());
};

LogicalResult JVPLoweringPattern::matchAndRewrite(JVPOp op, PatternRewriter &rewriter) const
{
    MLIRContext *ctx = getContext();

    Location loc = op.getLoc();

    auto func_diff_operand_indices = computeDiffArgIndices(op.getDiffArgIndices());
    LLVM_DEBUG(dbgs() << "jvp_num_operands " << op.getOperands().size() << " \n");
    LLVM_DEBUG(dbgs() << "func_diff_operand_indices: " << func_diff_operand_indices << " \n");
    assert(func_diff_operand_indices.size() <= op.getOperands().size() / 2);
    size_t func_operands_size = op.getOperands().size() - func_diff_operand_indices.size();

    auto calleeOperands = op.getParams();
    auto tangOperands = OperandRange(op.operand_begin() + func_operands_size, op.operand_end());

    for (auto idx : func_diff_operand_indices) {
        assert(idx < calleeOperands.size() && "all diffArgIndices reference valid arguments");
    }

    auto resHalfsize = (op.result_type_end() - op.result_type_begin()) / 2;
    std::vector<Type> calleeOperandsSize;
    {
        for (auto o : calleeOperands)
            calleeOperandsSize.push_back(o.getType());
    }
    auto funcResultTypes =
        ValueTypeRange<ResultRange>(op.result_type_begin(), op.result_type_begin() + resHalfsize);

    auto calleeOp = SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(op, op.getCalleeAttr());

    auto grad_result_types = computeResultTypes(calleeOp, func_diff_operand_indices);
    LLVM_DEBUG(dbgs() << "grad_result_types: " << grad_result_types << " \n");
    assert(grad_result_types.size() == func_diff_operand_indices.size() * funcResultTypes.size() &&
           "GradOp does't seem to return a tuple of Jacobians");

    auto fCallOp = rewriter.create<func::CallOp>(loc, calleeOp, calleeOperands);

    auto gradOp = rewriter.create<GradOp>(loc, grad_result_types, op.getMethod(), op.getCallee(),
                                          calleeOperands, op.getDiffArgIndicesAttr(),
                                          op.getFiniteDiffParamAttr());

    std::vector<Value> einsumResults;
    for (size_t nout = 0; nout < funcResultTypes.size(); nout++) {
        std::optional<Value> acc;
        for (size_t nparam = 0; nparam < func_diff_operand_indices.size(); nparam++) {
            LLVM_DEBUG(dbgs() << "iteration: nout " << nout << " nparam " << nparam << "\n");
            auto jac = gradOp.getResults()[nparam + nout * func_diff_operand_indices.size()];
            auto tang = tangOperands[nparam];
            auto param = calleeOperands[func_diff_operand_indices[nparam]];

            auto sjac = _tovec(cast<TensorType>(jac.getType()).getShape());
            auto sparam = _tovec(cast<TensorType>(param.getType()).getShape());
            auto stang = _tovec(cast<TensorType>(tang.getType()).getShape());

            std::vector<int64_t> sjac_param(sjac.begin() + sjac.size() - sparam.size(), sjac.end());

            LLVM_DEBUG(dbgs() << "jac_type " << sjac << "\n");
            LLVM_DEBUG(dbgs() << "param_type " << sparam << "\n");
            LLVM_DEBUG(dbgs() << "tang_type " << stang << "\n");

            assert(sparam == stang && "Parameter and tanget shapes don't match");
            assert(sjac_param == sparam &&
                   "Jacobian shape doesn't contain the parameter shape as a suffix");

            std::vector<int64_t> jacAxisNames;
            {
                for (size_t i = 0; i < sjac.size(); i++) {
                    jacAxisNames.push_back(i);
                }
            }

            std::vector<int64_t> tangAxisNames;
            {
                for (size_t i = sjac.size() - sparam.size(); i < sjac.size(); i++) {
                    tangAxisNames.push_back(i);
                }
            }
            std::vector<int64_t> jvpAxisNames;
            {
                for (size_t i = 0; i < sjac.size() - sparam.size(); i++) {
                    jvpAxisNames.push_back(i);
                }
            }

            LLVM_DEBUG(dbgs() << "jac_axis " << jacAxisNames << "\n");
            LLVM_DEBUG(dbgs() << "tang_axis " << tangAxisNames << "\n");
            LLVM_DEBUG(dbgs() << "jvp_axis " << jvpAxisNames << "\n");

            /* tjac.getShape(); */

            auto res = einsumLinalgGeneric(rewriter, loc, jacAxisNames, tangAxisNames, jvpAxisNames,
                                           jac, tang);

            LLVM_DEBUG(dbgs() << "jvp result type " << res.getType() << "\n");

            if (!acc.has_value()) {
                acc = res;
            }
            else {
                assert(acc.value().getType() == res.getType());

                auto add_op = rewriter.create<linalg::ElemwiseBinaryOp>(
                    loc, res.getType(), ValueRange({acc.value(), res}), acc.value(),
                    linalg::BinaryFnAttr::get(ctx, linalg::BinaryFn::add),
                    linalg::TypeFnAttr::get(ctx, linalg::TypeFn::cast_signed));
                acc = add_op.getResultTensors()[0];
            }
        }
        assert(acc.has_value());
        einsumResults.push_back(acc.value());
    }

    std::vector<Value> results;
    results.insert(results.end(), fCallOp.getResults().begin(), fCallOp.getResults().end());
    results.insert(results.end(), einsumResults.begin(), einsumResults.end());

    rewriter.replaceOp(op, results);
    return success();
}

LogicalResult VJPLoweringPattern::matchAndRewrite(VJPOp op, PatternRewriter &rewriter) const
{
    MLIRContext *ctx = getContext();

    Location loc = op.getLoc();

    auto func_diff_operand_indices = computeDiffArgIndices(op.getDiffArgIndices());
    LLVM_DEBUG(dbgs() << "vjp_num_operands " << op.getOperands().size() << " \n");
    LLVM_DEBUG(dbgs() << "func_diff_operand_indices: " << func_diff_operand_indices << " \n");

    auto calleeOp = SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(op, op.getCalleeAttr());

    size_t func_operands_size = calleeOp.getFunctionType().getNumInputs();
    size_t cotang_operands_size = op.getOperands().size() - func_operands_size;
    assert(calleeOp.getFunctionType().getNumResults() == cotang_operands_size &&
           "the number of function results doesn't match the number of cotangent arguments");

    auto calleeOperands = op.getParams();
    {
        for (auto idx : func_diff_operand_indices) {
            assert(idx < calleeOperands.size() && "all diffArgIndices reference valid arguments");
        }
    }

    auto cotang_operands = OperandRange(op.operand_begin() + func_operands_size, op.operand_end());
    std::vector<Type> calleeOperandsSize;
    {
        for (auto o : calleeOperands)
            calleeOperandsSize.push_back(o.getType());
    }
    auto funcResultTypes = calleeOp.getResultTypes();

    auto grad_result_types = computeResultTypes(calleeOp, func_diff_operand_indices);
    LLVM_DEBUG(dbgs() << "grad_result_types: " << grad_result_types << " \n");
    assert(grad_result_types.size() == func_diff_operand_indices.size() * funcResultTypes.size() &&
           "GradOp does't seem to return a tuple of Jacobians");

    auto fCallOp = rewriter.create<func::CallOp>(loc, calleeOp, calleeOperands);

    auto gradOp = rewriter.create<GradOp>(loc, grad_result_types, op.getMethod(), op.getCallee(),
                                          calleeOperands, op.getDiffArgIndicesAttr(),
                                          op.getFiniteDiffParamAttr());

    std::vector<Value> einsumResults;
    for (size_t nparam = 0; nparam < func_diff_operand_indices.size(); nparam++) {
        std::optional<Value> acc;
        for (size_t nout = 0; nout < funcResultTypes.size(); nout++) {
            auto jac = gradOp.getResults()[nparam + nout * func_diff_operand_indices.size()];
            auto param = calleeOperands[func_diff_operand_indices[nparam]];
            auto cotang = cotang_operands[nout];

            auto sjac = _tovec(cast<TensorType>(jac.getType()).getShape());
            auto sparam = _tovec(cast<TensorType>(param.getType()).getShape());
            auto scotang = _tovec(cast<TensorType>(cotang.getType()).getShape());

            std::vector<int64_t> sjac_cotang(sjac.begin(), sjac.end() - sparam.size());

            LLVM_DEBUG(dbgs() << "jac_type " << sjac << "\n");
            LLVM_DEBUG(dbgs() << "param_type " << sparam << "\n");
            LLVM_DEBUG(dbgs() << "cotang_type " << scotang << "\n");

            assert(sjac_cotang == scotang &&
                   "Jacobian shape doesn't contain the cotang shape as a prefix");

            std::vector<int64_t> jacAxisNames;
            {
                for (size_t i = 0; i < sjac.size(); i++) {
                    jacAxisNames.push_back(i);
                }
            }
            std::vector<int64_t> cotangAxisNames;
            {
                for (size_t i = 0; i < scotang.size(); i++) {
                    cotangAxisNames.push_back(i);
                }
            }
            std::vector<int64_t> vjpAxisNames;
            {
                for (size_t i = scotang.size(); i < sjac.size(); i++) {
                    vjpAxisNames.push_back(i);
                }
            }

            LLVM_DEBUG(dbgs() << "jac_axis " << jacAxisNames << "\n");
            LLVM_DEBUG(dbgs() << "cotang_axis " << cotangAxisNames << "\n");
            LLVM_DEBUG(dbgs() << "vjp_axis " << vjpAxisNames << "\n");

            auto res = einsumLinalgGeneric(rewriter, loc, cotangAxisNames, jacAxisNames,
                                           vjpAxisNames, cotang, jac);

            LLVM_DEBUG(dbgs() << "vjp result type " << res.getType() << "\n");

            if (!acc.has_value()) {
                acc = res;
            }
            else {
                assert(acc.value().getType() == res.getType());

                auto add_op = rewriter.create<linalg::ElemwiseBinaryOp>(
                    loc, res.getType(), ValueRange({acc.value(), res}), acc.value(),
                    linalg::BinaryFnAttr::get(ctx, linalg::BinaryFn::add),
                    linalg::TypeFnAttr::get(ctx, linalg::TypeFn::cast_signed));
                acc = add_op.getResultTensors()[0];
            }
        }
        assert(acc.has_value());
        einsumResults.push_back(acc.value());
    }

    std::vector<Value> results;
    results.insert(results.end(), fCallOp.getResults().begin(), fCallOp.getResults().end());
    results.insert(results.end(), einsumResults.begin(), einsumResults.end());

    rewriter.replaceOp(op, results);
    return success();
}

} // namespace gradient
} // namespace catalyst
