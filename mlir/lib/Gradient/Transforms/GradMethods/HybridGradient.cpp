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

#include <algorithm>
#include <sstream>
#include <vector>

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"

#include "Gradient/Utils/CompDiffArgIndices.h"
#include "Gradient/Utils/GradientShape.h"

namespace catalyst {
namespace gradient {

static Value combineGradients(PatternRewriter &rewriter, Location loc, Value classicalJacobian,
                              Value quantumGradient, Value numParams, bool isResultScalar)
{
    TensorType cjacType = classicalJacobian.getType().cast<TensorType>();
    TensorType qgradType = quantumGradient.getType().cast<TensorType>();

    assert(cjacType.getRank() >= 1 && ShapedType::isDynamic(cjacType.getShape().back()) &&
           "classical jacobian is not a >=1D tensor of dynamic length in the last dimension");

    assert(qgradType.getRank() >= 1 && ShapedType::isDynamic(qgradType.getShape().front()) &&
           "quantum gradient is not a >=1D tensor of dynamic length in the first dimension");

    if (cjacType.getElementType() != qgradType.getElementType()) {
        bool larger = cjacType.getElementType().getIntOrFloatBitWidth() >
                      qgradType.getElementType().getIntOrFloatBitWidth();

        cjacType = RankedTensorType::get(cjacType.getShape(), qgradType.getElementType());
        classicalJacobian =
            larger ? rewriter.create<arith::TruncFOp>(loc, cjacType, classicalJacobian).getResult()
                   : rewriter.create<arith::ExtFOp>(loc, cjacType, classicalJacobian).getResult();
    }

    std::vector<int64_t> staticShape;
    staticShape.reserve(cjacType.getRank() + qgradType.getRank() - 2);
    staticShape.insert(staticShape.end(), cjacType.getShape().begin(),
                       cjacType.getShape().end() - 1);
    staticShape.insert(staticShape.end(), qgradType.getShape().begin() + 1,
                       qgradType.getShape().end());

    std::vector<Value> dynamicDimSizes;
    for (int i = 0; i < cjacType.getRank() - 1; i++) {
        if (ShapedType::isDynamic(staticShape[i])) {
            dynamicDimSizes.push_back(rewriter.create<tensor::DimOp>(loc, classicalJacobian, i));
        }
    }
    for (int i = 0; i < qgradType.getRank() - 1; i++) {
        if (ShapedType::isDynamic(staticShape[cjacType.getRank() - 1 + i])) {
            dynamicDimSizes.push_back(rewriter.create<tensor::DimOp>(loc, quantumGradient, i));
        }
    }

    Type baseType = qgradType.getElementType();
    TensorType resultTensorType = RankedTensorType::get(staticShape, baseType);
    bool isResultScalarTensor = resultTensorType.getRank() == 0;
    if (isResultScalarTensor) {
        resultTensorType = RankedTensorType::get({1}, baseType);
    }

    Value cZero = rewriter.create<arith::ConstantOp>(loc, rewriter.getF64FloatAttr(0.0));
    // Compute the contraction of the classical Jacobian and quantum gradient using
    // tensor.generate. Each element of the result tensor is constructed and yielded from within
    // the ops region by applying the dot product to a an appropriate slice of each tensor.
    auto bodyBuilder = [&](OpBuilder &rewriter, Location loc, ValueRange tensorIndices) {
        size_t cjacRank = cjacType.getRank();
        Type cjacSliceType = RankedTensorType::get({ShapedType::kDynamic}, baseType);

        std::vector<int64_t> staticOffsets(cjacRank, ShapedType::kDynamic);
        staticOffsets[cjacRank - 1] = 0;
        std::vector<int64_t> staticSizes(cjacRank, 1);
        staticSizes[cjacRank - 1] = ShapedType::kDynamic;
        std::vector<int64_t> staticStrides(cjacRank, 1);

        ValueRange dynamicOffsets = tensorIndices.take_front(cjacRank - 1);
        ValueRange dynamicSizes = numParams;
        ValueRange dynamicStrides = {};

        Value cjacSlice = rewriter.create<tensor::ExtractSliceOp>(
            loc, cjacSliceType, classicalJacobian, dynamicOffsets, dynamicSizes, dynamicStrides,
            staticOffsets, staticSizes, staticStrides);

        size_t qgradRank = qgradType.getRank();
        Type qgradSliceType = RankedTensorType::get({ShapedType::kDynamic}, baseType);

        staticOffsets = std::vector<int64_t>(qgradRank, ShapedType::kDynamic);
        staticOffsets[0] = 0;
        staticSizes = std::vector<int64_t>(qgradRank, 1);
        staticSizes[0] = ShapedType::kDynamic;
        staticStrides = std::vector<int64_t>(qgradRank, 1);

        dynamicOffsets = tensorIndices.take_back(qgradRank - 1);
        dynamicSizes = numParams;
        dynamicStrides = {};

        Value qgradSlice = rewriter.create<tensor::ExtractSliceOp>(
            loc, qgradSliceType, quantumGradient, dynamicOffsets, dynamicSizes, dynamicStrides,
            staticOffsets, staticSizes, staticStrides);

        Value resultTensor = rewriter.create<tensor::EmptyOp>(loc, ArrayRef<int64_t>{}, baseType);
        Value zeroTensor = rewriter.create<linalg::FillOp>(loc, cZero, resultTensor).getResult(0);
        Value result =
            rewriter.create<linalg::DotOp>(loc, ValueRange{cjacSlice, qgradSlice}, zeroTensor)
                .getResult(0);
        Value resultElement = rewriter.create<tensor::ExtractOp>(loc, result);

        rewriter.create<tensor::YieldOp>(loc, resultElement);
    };

    Value hybridGradient =
        rewriter.create<tensor::GenerateOp>(loc, resultTensorType, dynamicDimSizes, bodyBuilder);

    if (isResultScalarTensor) {
        TensorType reshapedInputType = RankedTensorType::get({}, baseType);
        // The result tensor type of a reshape can be zero-ranked if the operand tensor type is
        // statically shaped with all dimensions being unit extent. In such case the reassociation
        // map is empty.
        SmallVector<ReassociationIndices> reassociationIndices = {};
        hybridGradient = rewriter.create<tensor::CollapseShapeOp>(
            loc, reshapedInputType, hybridGradient, reassociationIndices);
    }

    if (isResultScalar) {
        hybridGradient = rewriter.create<tensor::ExtractOp>(loc, hybridGradient);
    }

    return hybridGradient;
}

/// Generate an mlir function to compute the full gradient of a quantum function.
///
/// With the parameter-shift method (and certain other methods) the gradient of a quantum function
/// is computed as two sperate parts: the gradient of the classical pre-processing function for
/// gate parameters, termed "classical Jacobian", and the purely "quantum gradient" of a
/// differentiable output of a circuit. The two components can be combined to form the gradient of
/// the entire quantum function via tensor contraction along the gate parameter dimension.
///
func::FuncOp genFullGradFunction(PatternRewriter &rewriter, Location loc, GradOp gradOp,
                                 func::FuncOp argMapFn, func::FuncOp qGradFn, StringRef method)
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

        fullGradFn = rewriter.create<func::FuncOp>(loc, fnName, fnType, visibility);
        Block *entryBlock = fullGradFn.addEntryBlock();
        rewriter.setInsertionPointToStart(entryBlock);

        // Collect arguments and invoke the classical jacobian and quantum gradient functions.
        std::vector<Value> callArgs(fullGradFn.getArguments().begin(),
                                    fullGradFn.getArguments().end());

        std::vector<Type> resTypes = computeResultTypes(argMapFn, diffArgIndices);
        DenseIntElementsAttr diffArgIndicesAttr = gradOp.getDiffArgIndices().value_or(nullptr);
        GradOp jacOp = rewriter.create<GradOp>(loc, resTypes, "fd", argMapFn.getName(), callArgs,
                                               diffArgIndicesAttr, nullptr);
        ValueRange classicalJacobians = jacOp.getResults();

        Value numParams =
            rewriter.create<tensor::DimOp>(loc, classicalJacobians.front(), /*index=*/0);
        callArgs.push_back(numParams);
        ValueRange quantumGradients =
            rewriter.create<func::CallOp>(loc, qGradFn, callArgs).getResults();

        // Compute the hybrid gradients via tensor contraction.
        std::vector<Value> hybridGradients;
        hybridGradients.reserve(quantumGradients.size() * classicalJacobians.size());
        size_t idx = 0;
        for (Value classicalJacobian : classicalJacobians) {
            for (Value quantumGradient : quantumGradients) {
                bool isResultScalar = !gradOp.getResult(idx++).getType().isa<TensorType>();
                hybridGradients.push_back(combineGradients(
                    rewriter, loc, classicalJacobian, quantumGradient, numParams, isResultScalar));
            }
        }

        rewriter.create<func::ReturnOp>(loc, hybridGradients);
    }

    return fullGradFn;
}

} // namespace gradient
} // namespace catalyst
