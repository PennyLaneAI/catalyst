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

#include "Gradient/Utils/CompDiffArgIndices.h"
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
    const std::vector<size_t> &diffArgIndices = compDiffArgIndices(gradOp.getDiffArgIndices());
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
            Value result = rewriter.create<tensor::EmptyOp>(loc, resultType, ValueRange{});
            auto rankResult = resultType.cast<RankedTensorType>().getRank();
            auto shapeResult = resultType.cast<RankedTensorType>().getShape();
            j++;

            std::vector<BackpropOp> intermediateGradients;
            auto rank = quantumGradient.getType().cast<RankedTensorType>().getRank();

            if (rank > 1) {
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
                        loc, resultsBackpropTypes, argMapFn.getName(), callArgs,
                        extractQuantumGradient, ValueRange{}, diffArgIndicesAttr);

                    intermediateGradients.push_back(backpropOp);
                }
                for (size_t i = 0; i < resultsBackpropTypes.size(); i++) {

                    // strides
                    std::vector<int64_t> stridesSlice(rankResult, 1);

                    for (int64_t index = 0; index < totalOutcomes; index++) {
                        auto intermediateGradient = intermediateGradients[index];
                        Value gradient = intermediateGradient.getResult(i);

                        Type gradientType = gradient.getType();
                        auto rankGradient = gradientType.cast<RankedTensorType>().getRank();

                        // sizes
                        std::vector<int64_t> sizesSlice{shapeResult};
                        for (int64_t sliceIndex = rankResult - 1; sliceIndex >= rankGradient;
                             sliceIndex--) {
                            sizesSlice[sliceIndex] = 1;
                        }

                        // offset
                        auto offsetSlice = allOffsets[index];
                        for (int64_t offsetIndex = 0; offsetIndex < rankGradient; offsetIndex++) {
                            offsetSlice.insert(offsetSlice.begin(), 0);
                        }
                        result = rewriter.create<tensor::InsertSliceOp>(
                            loc, resultType, gradient, result, ValueRange{}, ValueRange{},
                            ValueRange{}, offsetSlice, sizesSlice, stridesSlice);
                    }
                    hybridGradients.push_back(result);
                }
            }
            else {
                BackpropOp backpropOp = rewriter.create<BackpropOp>(
                    loc, resultsBackpropTypes, argMapFn.getName(), callArgs, quantumGradient,
                    ValueRange{}, diffArgIndicesAttr);
                // Loop over params
                for (size_t i = 0; i < backpropOp.getNumResults(); i++) {
                    Value result = backpropOp.getResult(i);
                    hybridGradients.push_back(result);
                }
            }
        }
        rewriter.create<func::ReturnOp>(loc, hybridGradients);
    }

    return fullGradFn;
}

} // namespace gradient
} // namespace catalyst
