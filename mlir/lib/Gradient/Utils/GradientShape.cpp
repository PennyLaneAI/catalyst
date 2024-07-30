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

#include "mlir/Dialect/Tensor/IR/Tensor.h"

#include "Gradient/Utils/GradientShape.h"

using namespace mlir;

namespace catalyst {
namespace gradient {

/// Compute the result types of the gradient of a function.
///
/// The argument signature of the GradOp is the same as that of the differentiated function,
/// whereas the result signature is the set of shape unions for each combination of differentiable
/// argument function result.
///
std::vector<Type> computeResultTypes(func::FuncOp callee, const std::vector<size_t> &diffArgIndices)
{
    std::vector<Type> gradResultTypes;
    FunctionType fnType = callee.getFunctionType();

    // The grad output should contain one set of results (equal in size to
    // the number of function results) for each differentiable argument.
    size_t numDiffArgs = diffArgIndices.size();
    size_t numFnResults = fnType.getNumResults();
    size_t numGradResults = numFnResults * numDiffArgs;
    gradResultTypes.reserve(numGradResults);
    for (size_t j = 0; j < numFnResults; j++) {
        Type fnResType = fnType.getResult(j);

        std::vector<int64_t> resShape;
        auto tensorType = dyn_cast<TensorType>(fnResType);
        if (tensorType) {
            resShape.reserve(tensorType.getRank());
            resShape.insert(resShape.end(), tensorType.getShape().begin(),
                            tensorType.getShape().end());
        }

        for (size_t i = 0; i < numDiffArgs; i++) {
            assert(diffArgIndices[i] < callee.getNumArguments() && "invalid diff argument index");

            std::vector<int64_t> gradResShape = resShape;
            Type diffArgType = fnType.getInput(diffArgIndices[i]);
            if (auto tensorType = dyn_cast<TensorType>(diffArgType)) {
                gradResShape.reserve(resShape.size() + tensorType.getRank());
                gradResShape.insert(gradResShape.end(), tensorType.getShape().begin(),
                                    tensorType.getShape().end());
                fnResType = tensorType.getElementType();
            }
            else {
                fnResType = diffArgType;
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

        if (auto tensorType = dyn_cast<RankedTensorType>(resultType)) {
            assert(tensorType.hasStaticShape() && "only static tensors supported for autodiff");
            ArrayRef<int64_t> tensorShape = tensorType.getShape();
            gradShape.insert(gradShape.end(), tensorShape.begin(), tensorShape.end());
            resultType = tensorType.getElementType();
        }

        qGradResTypes.push_back(RankedTensorType::get(gradShape, resultType));
    }

    return qGradResTypes;
}

/// Produce a vector of the expected types of the backpropagation results.
///
/// The non differentiable params are filtered out.
///
std::vector<Type> computeBackpropTypes(func::FuncOp callee,
                                       const std::vector<size_t> &diffArgIndices)
{
    std::vector<Type> backpropResTypes;
    FunctionType fnType = callee.getFunctionType();

    size_t numDiffArgs = diffArgIndices.size();
    backpropResTypes.reserve(numDiffArgs);

    for (size_t i = 0; i < numDiffArgs; i++) {
        assert(diffArgIndices[i] < callee.getNumArguments() && "invalid diff argument index");
        Type diffArgType = fnType.getInput(diffArgIndices[i]);
        assert(isDifferentiable(diffArgType) &&
               "diff argument must be a float, complex or tensor of either");
        backpropResTypes.push_back(diffArgType);
    }

    return backpropResTypes;
}

bool isDifferentiable(Type type)
{
    // Only real-numbers are supported for differentiation
    if (isa<FloatType>(type)) {
        return true;
    }
    if (auto shapedType = dyn_cast<ShapedType>(type)) {
        return isDifferentiable(shapedType.getElementType());
    }
    return false;
}

/// Produce a normalized array of argument indices considered differentiable.
///
/// This is typically based on an attribute attached to gradient operations, but in the
/// absence thereof it is assumed that the first argument is differentiable.
///
std::vector<size_t> computeDiffArgIndices(std::optional<DenseIntElementsAttr> indices)
{
    // By default only the first argument is differentiated, otherwise gather indices.
    std::vector<size_t> diffArgIndices{0};
    if (indices.has_value()) {
        auto range = indices.value().getValues<size_t>();
        diffArgIndices = std::vector<size_t>(range.begin(), range.end());
    }
    return diffArgIndices;
}

/// Produce a filtered list of arguments which are differentiable.
///
std::vector<Value> computeDiffArgs(ValueRange args, std::optional<DenseIntElementsAttr> indices)
{
    const std::vector<size_t> &diffArgIndices = computeDiffArgIndices(indices);

    std::vector<Value> diffArgs;
    diffArgs.reserve(diffArgIndices.size());
    for (size_t idx : diffArgIndices) {
        diffArgs.push_back(args[idx]);
    }

    return diffArgs;
}

} // namespace gradient
} // namespace catalyst
