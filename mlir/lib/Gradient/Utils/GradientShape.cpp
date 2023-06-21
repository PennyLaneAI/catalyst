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
#include "Gradient/Utils/GradientShape.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"

using namespace mlir;

namespace catalyst {
namespace gradient {

/// Compute the result types of the gradient of a function.
///
/// The argument signature of the GradOp is the same as that of the differentiated function,
/// whereas the result signature is the set of shape unions for each combination of differentiable
/// argument function result.
///
std::vector<Type> computeResultTypes(func::FuncOp callee,
                                     const std::vector<uint64_t> &diffArgIndices)
{
    std::vector<Type> gradResultTypes;
    FunctionType fnType = callee.getFunctionType();

    // The grad output should contain one set of results (equal in size to
    // the number of function results) for each differentiable argument.
    size_t numDiffArgs = diffArgIndices.size();
    size_t numFnResults = fnType.getNumResults();
    size_t numGradResults = numDiffArgs * numFnResults;
    gradResultTypes.reserve(numGradResults);

    // The numeric type of a grad result should match the numeric type of the corresponding
    // function result. The shape is given by grouping the differentiated argument shape with
    // the corresponding function result shape.
    for (size_t i = 0; i < numDiffArgs; i++) {
        assert(diffArgIndices[i] < callee.getNumArguments() && "invalid diff argument index");

        Type diffArgType = fnType.getInput(diffArgIndices[i]);

        std::vector<int64_t> diffArgShape;
        if (auto tensorType = diffArgType.dyn_cast<TensorType>()) {
            diffArgShape.reserve(tensorType.getRank());
            diffArgShape.insert(diffArgShape.end(), tensorType.getShape().begin(),
                                tensorType.getShape().end());
        }

        for (size_t j = 0; j < numFnResults; j++) {
            Type fnResType = fnType.getResult(j);

            std::vector<int64_t> gradResShape = diffArgShape;
            auto tensorType = fnResType.dyn_cast<TensorType>();
            if (tensorType) {
                gradResShape.reserve(diffArgShape.size() + tensorType.getRank());
                gradResShape.insert(gradResShape.end(), tensorType.getShape().begin(),
                                    tensorType.getShape().end());
                fnResType = tensorType.getElementType();
            }

            Type gradResType = !gradResShape.empty() || tensorType
                                   ? RankedTensorType::get(gradResShape, fnResType)
                                   : fnResType;

            gradResultTypes.push_back(gradResType);
        }
    }

    return gradResultTypes;
}

std::vector<Type> computeQGradTypes(func::FuncOp callee)
{
    std::vector<Type> qGradResTypes;
    qGradResTypes.reserve(callee.getNumResults());

    for (Type resultType : callee.getResultTypes()) {
        std::vector<int64_t> gradShape = {ShapedType::kDynamic};

        if (auto tensorType = resultType.dyn_cast<RankedTensorType>()) {
            assert(tensorType.hasStaticShape() && "only static tensors supported for autodiff");
            ArrayRef<int64_t> tensorShape = tensorType.getShape();
            gradShape.insert(gradShape.end(), tensorShape.begin(), tensorShape.end());
            resultType = tensorType.getElementType();
        }

        qGradResTypes.push_back(RankedTensorType::get(gradShape, resultType));
    }

    return qGradResTypes;
}

std::vector<Type> computeBackpropTypes(func::FuncOp callee,
                                       const std::vector<uint64_t> &diffArgIndices)
{
    std::vector<Type> backpropResTypes;
    FunctionType fnType = callee.getFunctionType();

    size_t numDiffArgs = diffArgIndices.size();
    backpropResTypes.reserve(numDiffArgs);

    for (size_t i = 0; i < numDiffArgs; i++) {
        assert(diffArgIndices[i] < callee.getNumArguments() && "invalid diff argument index");

        Type diffArgType = fnType.getInput(diffArgIndices[i]);

        if (auto tensorType = diffArgType.dyn_cast<RankedTensorType>()) {
            ArrayRef<int64_t> tensorShape = tensorType.getShape();
            diffArgType = tensorType.getElementType();
            backpropResTypes.push_back(RankedTensorType::get(tensorShape, diffArgType));
        }
    }
    // Assume Args are always tensor
    // else {
    //     ArrayRef<int64_t> tensorShape;
    //     backpropResTypes.push_back(RankedTensorType::get(tensorShape, argType));
    // }

    return backpropResTypes;
}

} // namespace gradient
} // namespace catalyst
