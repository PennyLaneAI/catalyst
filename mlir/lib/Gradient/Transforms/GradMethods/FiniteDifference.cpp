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
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"

#include "Gradient/Utils/DifferentialQNode.h"
#include "Gradient/Utils/GradientShape.h"

namespace catalyst {
namespace gradient {

LogicalResult FiniteDiffLowering::matchAndRewrite(GradOp op, PatternRewriter &rewriter) const
{
    if (op.getMethod() != "fd") {
        return failure();
    }

    Location loc = op.getLoc();
    const std::vector<size_t> &diffArgIndices = computeDiffArgIndices(op.getDiffArgIndices());
    std::stringstream uniquer;
    std::copy(diffArgIndices.begin(), diffArgIndices.end(), std::ostream_iterator<int>(uniquer));
    std::string fnName = op.getCallee().getLeafReference().str() + ".finitediff" + uniquer.str();
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

        gradFn = func::FuncOp::create(rewriter, loc, fnName, fnType, visibility, nullptr, nullptr);
        rewriter.setInsertionPointToStart(gradFn.addEntryBlock());

        computeFiniteDiff(rewriter, loc, gradFn, callee, diffArgIndices, hValue);
    }

    rewriter.replaceOpWithNewOp<func::CallOp>(op, gradFn, op.getArgOperands());
    return success();
}

void FiniteDiffLowering::computeFiniteDiff(PatternRewriter &rewriter, Location loc,
                                           func::FuncOp gradFn, func::FuncOp callee,
                                           const std::vector<size_t> &diffArgIndices, double hValue)
{
    ValueRange callArgs = gradFn.getArguments();
    TypeRange gradResTypes = gradFn.getResultTypes();
    std::vector<Value> gradients;
    gradients.reserve(gradFn.getNumResults());

    func::CallOp callOp = func::CallOp::create(rewriter, loc, callee, callArgs);
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
                    dynamicDimSizes.push_back(tensor::DimOp::create(rewriter, loc, callRes, j));
                }
            }
            for (int64_t i = 0; i < operandRank; i++) {
                if (operandShape[i] == ShapedType::kDynamic) {
                    dynamicDimSizes.push_back(tensor::DimOp::create(rewriter, loc, diffArg, i));
                }
            }

            TypedAttr shiftForResult = rewriter.getFloatAttr(baseResultTy, hValue);
            Value hForResult = arith::ConstantOp::create(rewriter, loc, shiftForResult);
            if (isGradientTensor && cast<TensorType>(gradientTy).hasStaticShape()) {
                hForResult = tensor::SplatOp::create(rewriter, loc, hForResult, gradientTy);
            }
            else if (isGradientTensor) {
                Value outTensor = tensor::EmptyOp::create(rewriter, loc, gradientShape,
                                                          baseResultTy, dynamicDimSizes);
                hForResult =
                    linalg::FillOp::create(rewriter, loc, hForResult, outTensor).getResult(0);
            }

            TypedAttr shiftForOperand =
                isOperandScalarTensor
                    ? (TypedAttr)DenseFPElementsAttr::get(cast<ShapedType>(operandTy), hValue)
                    : (TypedAttr)rewriter.getFloatAttr(baseOperandTy, hValue);
            Value hForOperand = arith::ConstantOp::create(rewriter, loc, shiftForOperand);

            Value gradient;
            if (!isOperandTensor || isOperandScalarTensor) {
                Value diffArgShifted = arith::AddFOp::create(rewriter, loc, diffArg, hForOperand);

                std::vector<Value> callArgsForward(callArgs.begin(), callArgs.end());
                callArgsForward[diffArgIdx] = diffArgShifted;

                func::CallOp callOpForward =
                    func::CallOp::create(rewriter, loc, callee, callArgsForward);
                Value callResForward = callOpForward.getResult(diffResIdx);

                gradient = arith::SubFOp::create(rewriter, loc, callResForward, callRes);
            }
            else {
                auto bodyBuilder = [&](OpBuilder &rewriter, Location loc,
                                       ValueRange tensorIndices) -> void {
                    // we need to do this to guarantee a copy here.
                    // otherwise, each time we enter this scope, we will have a different
                    // value for diffArgElemen
                    //
                    // %memref = bufferization.to_memref %arg0 : memref<2xf64>
                    // %copy = bufferization.clone %memref : memref<2xf64> to memref<2xf64>
                    // %tensor = bufferization.to_tensor %copy restrict : memref<2xf64>
                    auto tensorTy = diffArg.getType();
                    auto memrefTy = bufferization::getMemRefTypeWithStaticIdentityLayout(
                        cast<TensorType>(tensorTy));
                    auto toBufferOp =
                        bufferization::ToBufferOp::create(rewriter, loc, memrefTy, diffArg);

                    auto cloneOp = bufferization::CloneOp::create(rewriter, loc, toBufferOp);

                    auto toTensorOp = bufferization::ToTensorOp::create(
                        rewriter, loc,
                        memref::getTensorTypeFromMemRefType(cloneOp.getOutput().getType()), cloneOp,
                        true);

                    auto diffArgCopy = toTensorOp.getResult();

                    Value diffArgElem = tensor::ExtractOp::create(
                        rewriter, loc, diffArgCopy, tensorIndices.take_back(operandRank));
                    Value diffArgElemShifted =
                        arith::AddFOp::create(rewriter, loc, diffArgElem, hForOperand);
                    Value diffArgShifted =
                        tensor::InsertOp::create(rewriter, loc, diffArgElemShifted, diffArgCopy,
                                                 tensorIndices.take_back(operandRank));

                    std::vector<Value> callArgsForward(callArgs.begin(), callArgs.end());
                    callArgsForward[diffArgIdx] = diffArgShifted;

                    func::CallOp callOpForward =
                        func::CallOp::create(rewriter, loc, callee, callArgsForward);
                    Value callResForward = callOpForward.getResult(diffResIdx);

                    Value result = arith::SubFOp::create(rewriter, loc, callResForward, callRes);
                    if (isResultTensor) {
                        result = tensor::ExtractOp::create(rewriter, loc, result,
                                                           tensorIndices.take_front(resultRank));
                    }

                    tensor::YieldOp::create(rewriter, loc, result);
                };

                gradient = tensor::GenerateOp::create(rewriter, loc, gradientTy, dynamicDimSizes,
                                                      bodyBuilder);
            }

            gradient = arith::DivFOp::create(rewriter, loc, gradient, hForResult);
            gradients.push_back(gradient);
        }
    }

    func::ReturnOp::create(rewriter, loc, gradients);
}

} // namespace gradient
} // namespace catalyst
