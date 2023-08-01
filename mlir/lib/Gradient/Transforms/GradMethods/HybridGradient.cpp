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

#include "ParameterShift.hpp"

#include "iostream"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <sstream>
#include <vector>

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Index/IR/IndexOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"

#include "Gradient/Utils/GradientShape.h"

namespace catalyst {
namespace gradient {

/// Generate an mlir function to compute the full gradient of a quantum function.
///
/// With the parameter-shift method (and certain other methods) the gradient of a quantum function
/// is computed as two separate parts: the gradient of the classical pre-processing function for
/// gate parameters, termed "classical Jacobian", and the purely "quantum gradient" of a
/// differentiable output of a circuit. The two components can be combined to form the gradient of
/// the entire quantum function via tensor contraction along the gate parameter dimension.
///
func::FuncOp genFullGradFunction(PatternRewriter &rewriter, Location loc, GradOp gradOp,
                                 func::FuncOp paramCountFn, func::FuncOp argMapFn,
                                 func::FuncOp qGradFn, StringRef method)
{
    // Define the properties of the full gradient function.
    const std::vector<size_t> &diffArgIndices = computeDiffArgIndices(gradOp.getDiffArgIndices());
    std::stringstream uniquer;
    std::copy(diffArgIndices.begin(), diffArgIndices.end(), std::ostream_iterator<int>(uniquer));
    std::string fnName = gradOp.getCallee().str() + ".fullgrad" + uniquer.str() + method.str();
    FunctionType fnType =
        rewriter.getFunctionType(gradOp.getOperandTypes(), gradOp.getResultTypes());
    StringAttr visibility = rewriter.getStringAttr("private");

    func::FuncOp fullGradFn =
        SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(gradOp, rewriter.getStringAttr(fnName));
    if (!fullGradFn) {
        PatternRewriter::InsertionGuard insertGuard(rewriter);
        rewriter.setInsertionPointAfter(qGradFn);

        fullGradFn =
            rewriter.create<func::FuncOp>(loc, fnName, fnType, visibility, nullptr, nullptr);
        Block *entryBlock = fullGradFn.addEntryBlock();
        rewriter.setInsertionPointToStart(entryBlock);

        // Collect arguments and invoke the classical jacobian and quantum gradient functions.
        SmallVector<Value> callArgs(fullGradFn.getArguments());

        Value numParams = rewriter.create<func::CallOp>(loc, paramCountFn, callArgs).getResult(0);
        SmallVector<Value> qGradArgs(callArgs);
        qGradArgs.push_back(numParams);
        ValueRange quantumGradients =
            rewriter.create<func::CallOp>(loc, qGradFn, qGradArgs).getResults();

        DenseIntElementsAttr diffArgIndicesAttr = gradOp.getDiffArgIndices().value_or(nullptr);

        auto resultsBackpropTypes = computeBackpropTypes(argMapFn, diffArgIndices);
        // Compute hybrid gradients via Enzyme
        std::vector<Value> hybridGradients;
        int j = 0;
        // Loop over the measurements
        for (Value quantumGradient : quantumGradients) {
            Type resultType = gradOp.getResult(j).getType();
            int64_t rankResult = 0;
            ArrayRef<int64_t> shapeResult;
            if (auto resultTensorType = dyn_cast<RankedTensorType>(resultType)) {
                rankResult = resultTensorType.getRank();
                shapeResult = resultTensorType.getShape();
            }
            j++;

            std::vector<BackpropOp> intermediateGradients;
            auto rank = quantumGradient.getType().cast<RankedTensorType>().getRank();

            if (rank > 1) {
                Value result = rewriter.create<tensor::EmptyOp>(loc, resultType, ValueRange{});
                std::vector<int64_t> sizes =
                    quantumGradient.getType().cast<RankedTensorType>().getShape();

                std::vector<std::vector<int64_t>> allOffsets;
                std::vector<int64_t> cutOffset(sizes.begin() + 1, sizes.end());

                std::vector<int64_t> currentOffset(cutOffset.size(), 0);

                int64_t totalOutcomes = 1;
                for (int64_t dim : cutOffset) {
                    totalOutcomes *= dim;
                }

                for (int64_t outcome = 0; outcome < totalOutcomes; outcome++) {
                    allOffsets.push_back(currentOffset);

                    for (int64_t i = cutOffset.size() - 1; i >= 0; i--) {
                        currentOffset[i]++;
                        if (currentOffset[i] < cutOffset[i]) {
                            break;
                        }
                        currentOffset[i] = 0;
                    }
                }

                std::vector<int64_t> strides(rank, 1);
                std::vector<Value> dynStrides = {};

                std::vector<Value> dynOffsets = {};

                std::vector<Value> dynSizes;

                for (size_t index = 0; index < sizes.size(); ++index) {
                    if (index == 0) {
                        Value idx = rewriter.create<index::ConstantOp>(loc, index);
                        Value dimSize = rewriter.create<tensor::DimOp>(loc, quantumGradient, idx);
                        dynSizes.push_back(dimSize);
                    }
                    else {
                        sizes[index] = 1;
                    }
                }
                for (auto offsetRight : allOffsets) {
                    std::vector<int64_t> offsets{0};
                    offsets.insert(offsets.end(), offsetRight.begin(), offsetRight.end());
                    auto rankReducedType =
                        tensor::ExtractSliceOp::inferCanonicalRankReducedResultType(
                            1, quantumGradient.getType().cast<RankedTensorType>(), offsets, sizes,
                            strides)
                            .cast<RankedTensorType>();
                    Value extractQuantumGradient = rewriter.create<tensor::ExtractSliceOp>(
                        loc, rankReducedType, quantumGradient, dynOffsets, dynSizes, dynStrides,
                        offsets, sizes, strides);
                    BackpropOp backpropOp = rewriter.create<BackpropOp>(
                        loc, resultsBackpropTypes, argMapFn.getName(), callArgs, ValueRange{},
                        ValueRange{}, extractQuantumGradient, diffArgIndicesAttr);

                    intermediateGradients.push_back(backpropOp);
                }
                for (size_t i = 0; i < resultsBackpropTypes.size(); i++) {
                    // strides
                    std::vector<int64_t> stridesSlice(rankResult, 1);

                    for (int64_t index = 0; index < totalOutcomes; index++) {
                        auto intermediateGradient = intermediateGradients[index];
                        Value gradient = intermediateGradient.getResult(i);

                        Type gradientType = gradient.getType();
                        if (auto gradientTensorType = dyn_cast<RankedTensorType>(gradientType)) {
                            int64_t rankGradient = gradientTensorType.getRank();
                            // sizes
                            std::vector<int64_t> sizesSlice{shapeResult};
                            for (int64_t sliceIndex = rankResult - 1; sliceIndex >= rankGradient;
                                 sliceIndex--) {
                                sizesSlice[sliceIndex] = 1;
                            }

                            // offset
                            auto offsetSlice = allOffsets[index];
                            for (int64_t offsetIndex = 0; offsetIndex < rankGradient;
                                 offsetIndex++) {
                                offsetSlice.insert(offsetSlice.begin(), 0);
                            }
                            result = rewriter.create<tensor::InsertSliceOp>(
                                loc, resultType, gradient, result, ValueRange{}, ValueRange{},
                                ValueRange{}, offsetSlice, sizesSlice, stridesSlice);
                        }
                        else {
                            assert(isa<FloatType>(gradient.getType()));
                            SmallVector<Value> insertIndices;
                            for (int64_t offset : allOffsets[index]) {
                                insertIndices.push_back(
                                    rewriter.create<index::ConstantOp>(loc, offset));
                            }
                            result = rewriter.create<tensor::InsertOp>(loc, gradient, result,
                                                                       insertIndices);
                        }
                    }
                    hybridGradients.push_back(result);
                }
            }
            else {
                // The quantum gradient is a rank 1 tensor
                BackpropOp backpropOp = rewriter.create<BackpropOp>(
                    loc, resultsBackpropTypes, argMapFn.getName(), callArgs, ValueRange{},
                    ValueRange{}, quantumGradient, diffArgIndicesAttr);
                for (OpResult result : backpropOp.getResults()) {
                    Value hybridGradient = result;
                    Type gradResultType = gradOp.getResult(result.getResultNumber()).getType();
                    if (gradResultType != result.getType()) {
                        // The backprop op produces a row of the Jacobian, which always has the same
                        // type as the differentiated argument. If the rank of the quantum gradient
                        // is 1, this implies the callee returns a rank-0 value (either a
                        // scalar or a tensor<scalar>). The Jacobian of a scalar -> scalar should be
                        // a scalar, but as a special case, the Jacobian of a scalar ->
                        // tensor<scalar> should be tensor<scalar>.
                        if (isa<RankedTensorType>(gradResultType) &&
                            isa<FloatType>(result.getType())) {
                            Value jacobian =
                                rewriter.create<tensor::EmptyOp>(loc, gradResultType, ValueRange{});
                            hybridGradient = rewriter.create<tensor::InsertOp>(
                                loc, result, jacobian, ValueRange{});
                        }

                        // We also support where the argument is a tensor<scalar> but the desired
                        // hybrid gradient is a scalar. This is less about mathematical precision
                        // and more about ergonomics.
                        if (isa<FloatType>(gradResultType) &&
                            isa<RankedTensorType>(result.getType())) {
                            hybridGradient =
                                rewriter.create<tensor::ExtractOp>(loc, result, ValueRange{});
                        }
                    }

                    hybridGradients.push_back(hybridGradient);
                }
            }
        }
        rewriter.create<func::ReturnOp>(loc, hybridGradients);
    }

    return fullGradFn;
}

} // namespace gradient
} // namespace catalyst
