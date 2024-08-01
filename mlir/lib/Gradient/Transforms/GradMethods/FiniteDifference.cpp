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

#include "FiniteDifference.hpp"
#include "HybridGradient.hpp"

#include <algorithm>
#include <sstream>
#include <vector>

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"

#include "Gradient/Utils/DifferentialQNode.h"
#include "Gradient/Utils/GradientShape.h"

namespace catalyst {
namespace gradient {

LogicalResult FiniteDiffLowering::match(GradOp op) const
{
    if (op.getMethod() == "fd") {
        return success();
    }

    return failure();
}

void FiniteDiffLowering::rewrite(GradOp op, PatternRewriter &rewriter) const
{
    Location loc = op.getLoc();
    const std::vector<size_t> &diffArgIndices = computeDiffArgIndices(op.getDiffArgIndices());
    std::stringstream uniquer;
    std::copy(diffArgIndices.begin(), diffArgIndices.end(), std::ostream_iterator<int>(uniquer));
    std::string fnName = op.getCallee().str() + ".finitediff" + uniquer.str();
    FunctionType fnType = rewriter.getFunctionType(op.getOperandTypes(), op.getResultTypes());
    StringAttr visibility = rewriter.getStringAttr("private");
    func::FuncOp callee =
        SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(op, op.getCalleeAttr());

    double hValue =
        op.getFiniteDiffParam().has_value() ? op.getFiniteDiffParamAttr().getValueAsDouble() : 1e-7;

    func::FuncOp gradFn =
        SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(op, rewriter.getStringAttr(fnName));
    if (!gradFn) {
        PatternRewriter::InsertionGuard insertGuard(rewriter);
        rewriter.setInsertionPointAfter(callee);

        gradFn = rewriter.create<func::FuncOp>(loc, fnName, fnType, visibility, nullptr, nullptr);
        rewriter.setInsertionPointToStart(gradFn.addEntryBlock());

        computeFiniteDiff(rewriter, loc, gradFn, callee, diffArgIndices, hValue);
    }

    rewriter.replaceOpWithNewOp<func::CallOp>(op, gradFn, op.getArgOperands());
}

void FiniteDiffLowering::computeFiniteDiff(PatternRewriter &rewriter, Location loc,
                                           func::FuncOp gradFn, func::FuncOp callee,
                                           const std::vector<size_t> &diffArgIndices, double hValue)
{
    ValueRange callArgs = gradFn.getArguments();
    TypeRange gradResTypes = gradFn.getResultTypes();
    std::vector<Value> gradients;
    gradients.reserve(gradFn.getNumResults());

    func::CallOp callOp = rewriter.create<func::CallOp>(loc, callee, callArgs);
    for (size_t diffResIdx = 0; diffResIdx < callee.getNumResults(); ++diffResIdx) {
        for (size_t diffArgIdxIdx = 0; diffArgIdxIdx < diffArgIndices.size(); ++diffArgIdxIdx) {
            size_t diffArgIdx = diffArgIndices[diffArgIdxIdx];
            Value diffArg = callArgs[diffArgIdx];
            Value callRes = callOp.getResult(diffResIdx);

            Type operandTy = gradFn.getArgumentTypes()[diffArgIdx];
            Type resultTy = callee.getResultTypes()[diffResIdx];

            Type gradientTy = gradResTypes[diffArgIdxIdx + diffResIdx * diffArgIndices.size()];

            const bool isOperandTensor = isa<TensorType>(operandTy);
            const bool isResultTensor = isa<TensorType>(resultTy);
            const bool isGradientTensor = isa<TensorType>(gradientTy);

            int64_t operandRank = isOperandTensor ? cast<TensorType>(operandTy).getRank() : -1;
            int64_t resultRank = isResultTensor ? cast<TensorType>(resultTy).getRank() : -1;

            const bool isOperandScalarTensor = operandRank == 0;

            ArrayRef<int64_t> operandShape =
                isOperandTensor ? cast<TensorType>(operandTy).getShape() : ArrayRef<int64_t>();
            ArrayRef<int64_t> resultShape =
                isResultTensor ? cast<TensorType>(resultTy).getShape() : ArrayRef<int64_t>();
            ArrayRef<int64_t> gradientShape =
                isGradientTensor ? cast<TensorType>(gradientTy).getShape() : ArrayRef<int64_t>();

            Type baseOperandTy =
                isOperandTensor ? cast<TensorType>(operandTy).getElementType() : operandTy;
            Type baseResultTy =
                isResultTensor ? cast<TensorType>(resultTy).getElementType() : resultTy;

            std::vector<Value> dynamicDimSizes;
            for (int64_t j = 0; j < resultRank; j++) {
                if (resultShape[j] == ShapedType::kDynamic) {
                    dynamicDimSizes.push_back(rewriter.create<tensor::DimOp>(loc, callRes, j));
                }
            }
            for (int64_t i = 0; i < operandRank; i++) {
                if (operandShape[i] == ShapedType::kDynamic) {
                    dynamicDimSizes.push_back(rewriter.create<tensor::DimOp>(loc, diffArg, i));
                }
            }

            TypedAttr shiftForResult = rewriter.getFloatAttr(baseResultTy, hValue);
            Value hForResult = rewriter.create<arith::ConstantOp>(loc, shiftForResult);
            if (isGradientTensor && cast<TensorType>(gradientTy).hasStaticShape()) {
                hForResult = rewriter.create<tensor::SplatOp>(loc, hForResult, gradientTy);
            }
            else if (isGradientTensor) {
                Value outTensor = rewriter.create<tensor::EmptyOp>(loc, gradientShape, baseResultTy,
                                                                   dynamicDimSizes);
                hForResult =
                    rewriter.create<linalg::FillOp>(loc, hForResult, outTensor).getResult(0);
            }

            TypedAttr shiftForOperand =
                isOperandScalarTensor
                    ? (TypedAttr)DenseFPElementsAttr::get(cast<ShapedType>(operandTy), hValue)
                    : (TypedAttr)rewriter.getFloatAttr(baseOperandTy, hValue);
            Value hForOperand = rewriter.create<arith::ConstantOp>(loc, shiftForOperand);

            Value gradient;
            if (!isOperandTensor || isOperandScalarTensor) {
                Value diffArgShifted = rewriter.create<arith::AddFOp>(loc, diffArg, hForOperand);

                std::vector<Value> callArgsForward(callArgs.begin(), callArgs.end());
                callArgsForward[diffArgIdx] = diffArgShifted;

                func::CallOp callOpForward =
                    rewriter.create<func::CallOp>(loc, callee, callArgsForward);
                Value callResForward = callOpForward.getResult(diffResIdx);

                gradient = rewriter.create<arith::SubFOp>(loc, callResForward, callRes);
            }
            else {
                auto bodyBuilder = [&](OpBuilder &rewriter, Location loc,
                                       ValueRange tensorIndices) -> void {
                    Value diffArgElem = rewriter.create<tensor::ExtractOp>(
                        loc, diffArg, tensorIndices.take_back(operandRank));
                    Value diffArgElemShifted =
                        rewriter.create<arith::AddFOp>(loc, diffArgElem, hForOperand);
                    Value diffArgShifted = rewriter.create<tensor::InsertOp>(
                        loc, diffArgElemShifted, diffArg, tensorIndices.take_back(operandRank));

                    std::vector<Value> callArgsForward(callArgs.begin(), callArgs.end());
                    callArgsForward[diffArgIdx] = diffArgShifted;

                    func::CallOp callOpForward =
                        rewriter.create<func::CallOp>(loc, callee, callArgsForward);
                    Value callResForward = callOpForward.getResult(diffResIdx);

                    Value result = rewriter.create<arith::SubFOp>(loc, callResForward, callRes);
                    if (isResultTensor) {
                        result = rewriter.create<tensor::ExtractOp>(
                            loc, result, tensorIndices.take_front(resultRank));
                    }

                    rewriter.create<tensor::YieldOp>(loc, result);
                };

                gradient = rewriter.create<tensor::GenerateOp>(loc, gradientTy, dynamicDimSizes,
                                                               bodyBuilder);
            }

            gradient = rewriter.create<arith::DivFOp>(loc, gradient, hForResult);
            gradients.push_back(gradient);
        }
    }

    rewriter.create<func::ReturnOp>(loc, gradients);
}

} // namespace gradient
} // namespace catalyst
