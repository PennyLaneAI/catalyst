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

#include <algorithm>
#include <sstream>
#include <vector>

#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Index/IR/IndexOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"

#include "ClassicalJacobian.hpp"
#include "HybridGradient.hpp"

#include "Catalyst/Utils/CallGraph.h"
#include "Gradient/Utils/DifferentialQNode.h"
#include "Gradient/Utils/GradientShape.h"
#include "Quantum/IR/QuantumInterfaces.h"
#include "Quantum/IR/QuantumOps.h"
#include "Quantum/Utils/RemoveQuantumMeasurements.h"

using namespace mlir;

namespace catalyst {
namespace gradient {

/// Given a statically-shaped tensor type, execute `processWithIndices` for every entry of the
/// tensor. For example, a tensor<3x2xf64> will cause `processWithIndices` to be called with
/// indices:
/// [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1)].
void iterateOverEntries(RankedTensorType resultType, OpBuilder &builder, Location loc,
                        function_ref<void(ValueRange indices)> processWithIndices)
{
    assert(resultType.hasStaticShape());

    ArrayRef<int64_t> shape = resultType.getShape();
    // Compute the strides of the tensor
    unsigned product = 1;
    SmallVector<unsigned> strides;
    for (int64_t dim = resultType.getRank() - 1; dim >= 0; dim--) {
        strides.push_back(product);
        product *= shape[dim];
    }
    std::reverse(strides.begin(), strides.end());

    // Unflatten the tensor indices, using the strides to map from the flat to rank-aware
    // indices
    for (unsigned flatIdx = 0; flatIdx < resultType.getNumElements(); flatIdx++) {
        SmallVector<Value> indices;
        for (int64_t dim = 0; dim < resultType.getRank(); dim++) {
            indices.push_back(
                builder.create<index::ConstantOp>(loc, flatIdx / strides[dim] % shape[dim]));
        }

        processWithIndices(indices);
    }
}

void initializeCotangents(TypeRange primalResultTypes, unsigned activeResult, ValueRange indices,
                          OpBuilder &builder, Location loc, SmallVectorImpl<Value> &cotangents)
{
    Type activeResultType = primalResultTypes[activeResult];
    FloatType elementType =
        cast<FloatType>(isa<RankedTensorType>(activeResultType)
                            ? cast<RankedTensorType>(activeResultType).getElementType()
                            : activeResultType);

    Value zero = builder.create<arith::ConstantFloatOp>(loc, APFloat(0.0), elementType);
    Value one = builder.create<arith::ConstantFloatOp>(loc, APFloat(1.0), elementType);

    Value zeroTensor;
    if (auto activeResultTensor = dyn_cast<RankedTensorType>(activeResultType)) {
        zeroTensor = builder.create<tensor::EmptyOp>(loc, activeResultTensor,
                                                     /*dynamicSizes=*/ValueRange{});
    }
    else {
        zeroTensor = builder.create<tensor::EmptyOp>(loc, ArrayRef<int64_t>(), activeResultType);
    }
    zeroTensor = builder.create<linalg::FillOp>(loc, zero, zeroTensor).getResult(0);
    Value cotangent = builder.create<tensor::InsertOp>(loc, one, zeroTensor, indices);

    // Initialize cotangents for all of the primal outputs
    for (const auto &[resultIdx, primalResultType] : llvm::enumerate(primalResultTypes)) {
        if (resultIdx == activeResult) {
            cotangents.push_back(cotangent);
        }
        else {
            // We're not differentiating this output, so we push back a completely
            // zero cotangent. This memory still needs to be writable, hence using
            // an explicit empty + fill vs a constant tensor.
            Value zeroTensor;
            if (isa<RankedTensorType>(primalResultType)) {
                zeroTensor = builder.create<tensor::EmptyOp>(loc, primalResultType, ValueRange{});
            }
            else {
                zeroTensor =
                    builder.create<tensor::EmptyOp>(loc, ArrayRef<int64_t>(), primalResultType);
            }
            cotangents.push_back(
                builder.create<linalg::FillOp>(loc, zero, zeroTensor).getResult(0));
        }
    }
}

FailureOr<func::FuncOp> HybridGradientLowering::cloneCallee(PatternRewriter &rewriter,
                                                            GradOp gradOp, func::FuncOp callee,
                                                            SmallVectorImpl<Value> &backpropArgs)
{
    Location loc = callee.getLoc();
    std::string clonedCalleeName = (callee.getName() + ".cloned").str();
    func::FuncOp clonedCallee = SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(
        callee, rewriter.getStringAttr(clonedCalleeName));
    backpropArgs.append(gradOp.getArgOperands().begin(), gradOp.getArgOperands().end());
    if (!clonedCallee) {
        rewriter.setInsertionPointAfter(callee);
        // Replace calls with the QNode with the split QNode in the callee.
        clonedCallee = cast<func::FuncOp>(rewriter.clone(*callee));
        clonedCallee.setName(clonedCalleeName);
        SmallPtrSet<Operation *, 4> qnodes;
        SymbolTableCollection symbolTable;
        traverseCallGraph(callee, &symbolTable, [&](func::FuncOp funcOp) {
            if (funcOp == callee && isQNode(callee)) {
                qnodes.insert(callee);
            }
            else if (isQNode(funcOp)) {
                qnodes.insert(funcOp);
            }
        });

        for (Operation *qnodeOp : qnodes) {
            auto qnode = cast<func::FuncOp>(qnodeOp);
            if (getQNodeDiffMethod(qnode) == "finite-diff") {
                return callee.emitError(
                    "A QNode with diff_method='finite-diff' cannot be specified in a "
                    "callee of method='defer'");
            }

            // In order to allocate memory for various tensors relating to the number of gate
            // parameters at runtime we run a function that merely counts up for each gate parameter
            // encountered.
            func::FuncOp paramCountFn = genParamCountFunction(rewriter, loc, qnode);
            func::FuncOp qnodeWithParams = genQNodeWithParams(rewriter, loc, qnode);
            func::FuncOp qnodeSplit = genSplitPreprocessed(rewriter, loc, qnode, qnodeWithParams);

            // This attribute tells downstream patterns that this QNode requires the registration of
            // a custom quantum gradient.
            setRequiresCustomGradient(qnode, FlatSymbolRefAttr::get(qnodeWithParams));

            // Enzyme will fail if this function gets inlined.
            qnodeWithParams->setAttr("passthrough",
                                     rewriter.getArrayAttr(rewriter.getStringAttr("noinline")));

            // Replace calls to the original QNode with calls to the split QNode
            if (isQNode(clonedCallee) && qnode == callee) {
                // As the split preprocessed QNode requires the number of gate params as an extra
                // argument, we need to insert a call to the parameter count function at the
                // location of the grad op.
                PatternRewriter::InsertionGuard insertionGuard(rewriter);
                rewriter.setInsertionPoint(gradOp);

                Value paramCount =
                    rewriter.create<func::CallOp>(loc, paramCountFn, gradOp.getArgOperands())
                        .getResult(0);
                backpropArgs.push_back(paramCount);
                // If the callee is a QNode, we want to backprop through the split preprocessed
                // version.
                rewriter.eraseOp(clonedCallee);
                clonedCallee = qnodeSplit;
            }
            // We modify the callgraph in-place as we traverse it, so we can't use a symbol table
            // here or else the lookup will be stale.
            traverseCallGraph(clonedCallee, /*symbolTable=*/nullptr, [&](func::FuncOp funcOp) {
                funcOp.walk([&](func::CallOp callOp) {
                    if (callOp.getCallee() == qnode.getName()) {
                        PatternRewriter::InsertionGuard insertionGuard(rewriter);
                        rewriter.setInsertionPointToStart(&funcOp.getFunctionBody().front());
                        Value paramCount =
                            rewriter
                                .create<func::CallOp>(loc, paramCountFn, callOp.getArgOperands())
                                .getResult(0);
                        callOp.setCallee(qnodeSplit.getName());
                        callOp.getOperandsMutable().append(paramCount);
                    }
                });
            });
        }
    }
    return clonedCallee;
}

LogicalResult HybridGradientLowering::matchAndRewrite(GradOp op, PatternRewriter &rewriter) const
{
    if (op.getMethod() != "defer") {
        return failure();
    }

    Location loc = op.getLoc();
    func::FuncOp callee =
        SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(op, op.getCalleeAttr());

    SmallVector<Value> backpropArgs;
    FailureOr<func::FuncOp> clonedCallee = cloneCallee(rewriter, op, callee, backpropArgs);

    rewriter.setInsertionPoint(op);
    std::vector<size_t> diffArgIndices = computeDiffArgIndices(op.getDiffArgIndices());
    SmallVector<Value> backpropResults{op.getNumResults()};
    // Iterate over the primal results
    for (const auto &[cotangentIdx, primalResult] :
         llvm::enumerate(clonedCallee->getResultTypes())) {
        // There is one Jacobian per distinct differential argument.
        SmallVector<Value> jacobians;
        for (unsigned argIdx = 0; argIdx < diffArgIndices.size(); argIdx++) {
            Type jacobianType =
                op.getResultTypes()[argIdx * clonedCallee->getNumResults() + cotangentIdx];
            if (isa<RankedTensorType>(jacobianType)) {
                jacobians.push_back(rewriter.create<tensor::EmptyOp>(
                    loc, jacobianType, /*dynamicSizes=*/ValueRange{}));
            }
            else {
                jacobians.push_back(rewriter.create<arith::ConstantFloatOp>(
                    loc, APFloat(0.0), cast<FloatType>(jacobianType)));
            }
        }

        if (auto primalTensorResultType = dyn_cast<RankedTensorType>(primalResult)) {
            // Loop over every entry of this result, creating a one-hot cotangent vector and running
            // a backward pass via the BackpropOp.
            iterateOverEntries(
                primalTensorResultType, rewriter, loc,
                [&, cotangentIdx = cotangentIdx](ValueRange indices) {
                    // Initialize cotangents for all of the primal outputs
                    SmallVector<Value> cotangents;
                    initializeCotangents(clonedCallee->getResultTypes(), cotangentIdx, indices,
                                         rewriter, loc, cotangents);

                    auto backpropOp = rewriter.create<gradient::BackpropOp>(
                        loc, computeBackpropTypes(*clonedCallee, diffArgIndices),
                        clonedCallee->getName(), backpropArgs,
                        /*arg_shadows=*/ValueRange{}, /*primal results=*/ValueRange{}, cotangents,
                        op.getDiffArgIndicesAttr());

                    // Backprop gives a gradient of a single output entry w.r.t. all active inputs.
                    // Catalyst gives transposed Jacobians, such that the Jacobians have shape
                    // [...shape_inputs, ...shape_outputs].
                    for (const auto &[backpropIdx, jacobianSlice] :
                         llvm::enumerate(backpropOp.getResults())) {
                        if (auto sliceType = dyn_cast<RankedTensorType>(jacobianSlice.getType())) {
                            size_t sliceRank = sliceType.getRank();
                            auto jacobianType =
                                cast<RankedTensorType>(jacobians[backpropIdx].getType());
                            size_t jacobianRank = jacobianType.getRank();
                            if (sliceRank < jacobianRank) {
                                // Offsets are [0] * rank of backprop result + [...indices]
                                SmallVector<OpFoldResult> offsets;
                                offsets.append(sliceRank, rewriter.getIndexAttr(0));
                                offsets.append(indices.begin(), indices.end());

                                // Sizes are [...sliceShape] + [1] * (jacobianRank - sliceRank)
                                SmallVector<OpFoldResult> sizes;
                                for (int64_t dim : sliceType.getShape()) {
                                    sizes.push_back(rewriter.getIndexAttr(dim));
                                }
                                sizes.append(jacobianRank - sliceRank, rewriter.getIndexAttr(1));

                                // Strides are [1] * jacobianRank
                                SmallVector<OpFoldResult> strides{jacobianRank,
                                                                  rewriter.getIndexAttr(1)};

                                jacobians[backpropIdx] = rewriter.create<tensor::InsertSliceOp>(
                                    loc, jacobianSlice, jacobians[backpropIdx], offsets, sizes,
                                    strides);
                            }
                            else {
                                jacobians[backpropIdx] = jacobianSlice;
                            }
                        }
                        else {
                            assert(isa<FloatType>(jacobianSlice.getType()));
                            jacobians[backpropIdx] = rewriter.create<tensor::InsertOp>(
                                loc, jacobianSlice, jacobians[backpropIdx], indices);
                        }
                        backpropResults[backpropIdx * clonedCallee->getNumResults() +
                                        cotangentIdx] = jacobians[backpropIdx];
                    }
                });
        }
        else {
            SmallVector<Value> cotangents;
            initializeCotangents(clonedCallee->getResultTypes(), cotangentIdx, ValueRange(),
                                 rewriter, loc, cotangents);

            auto backpropOp = rewriter.create<gradient::BackpropOp>(
                loc, computeBackpropTypes(*clonedCallee, diffArgIndices), clonedCallee->getName(),
                backpropArgs,
                /*arg_shadows=*/ValueRange{}, /*primal results=*/ValueRange{}, cotangents,
                op.getDiffArgIndicesAttr());
            for (const auto &[backpropIdx, jacobianSlice] :
                 llvm::enumerate(backpropOp.getResults())) {
                size_t resultIdx = backpropIdx * clonedCallee->getNumResults() + cotangentIdx;
                backpropResults[resultIdx] = jacobianSlice;
                if (jacobianSlice.getType() != op.getResultTypes()[resultIdx]) {
                    // For ergonomics, if the backprop result is a point tensor and the
                    // user requests a scalar, give it to them.
                    if (isa<RankedTensorType>(jacobianSlice.getType()) &&
                        isa<FloatType>(op.getResultTypes()[resultIdx])) {
                        backpropResults[resultIdx] =
                            rewriter.create<tensor::ExtractOp>(loc, jacobianSlice, ValueRange{});
                    }
                }
            }
        }
    }

    rewriter.replaceOp(op, backpropResults);
    return success();
}

func::FuncOp HybridGradientLowering::genQNodeWithParams(PatternRewriter &rewriter, Location loc,
                                                        func::FuncOp qnode)
{
    std::string fnName = (qnode.getName() + ".withparams").str();
    SmallVector<Type> fnArgTypes(qnode.getArgumentTypes());
    auto paramsTensorType = RankedTensorType::get({ShapedType::kDynamic}, rewriter.getF64Type());
    fnArgTypes.push_back(paramsTensorType);
    FunctionType fnType = rewriter.getFunctionType(fnArgTypes, qnode.getResultTypes());

    func::FuncOp modifiedCallee =
        SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(qnode, rewriter.getStringAttr(fnName));
    if (modifiedCallee) {
        return modifiedCallee;
    }

    modifiedCallee = rewriter.create<func::FuncOp>(loc, fnName, fnType);
    modifiedCallee.setPrivate();
    rewriter.cloneRegionBefore(qnode.getBody(), modifiedCallee.getBody(), modifiedCallee.end());
    Block &entryBlock = modifiedCallee.getFunctionBody().front();
    BlockArgument paramsTensor = entryBlock.addArgument(paramsTensorType, loc);

    PatternRewriter::InsertionGuard insertionGuard(rewriter);
    rewriter.setInsertionPointToStart(&modifiedCallee.getFunctionBody().front());

    MemRefType paramsProcessedType = MemRefType::get({}, rewriter.getIndexType());
    Value paramCounter = rewriter.create<memref::AllocaOp>(loc, paramsProcessedType);
    Value cZero = rewriter.create<index::ConstantOp>(loc, 0);
    rewriter.create<memref::StoreOp>(loc, cZero, paramCounter);
    Value cOne = rewriter.create<index::ConstantOp>(loc, 1);

    auto loadThenIncrementCounter = [&](OpBuilder &builder, Value counter,
                                        Value paramTensor) -> Value {
        Value index = builder.create<memref::LoadOp>(loc, counter);
        Value nextIndex = builder.create<index::AddOp>(loc, index, cOne);
        builder.create<memref::StoreOp>(loc, nextIndex, counter);
        return builder.create<tensor::ExtractOp>(loc, paramTensor, index);
    };

    modifiedCallee.walk([&](quantum::DifferentiableGate gateOp) {
            OpBuilder::InsertionGuard insertGuard(rewriter);
            rewriter.setInsertionPoint(gateOp);

            ValueRange diffParams = gateOp.getDiffParams();
            SmallVector<Value> newParams{diffParams.size()};
            for (const auto [paramIdx, recomputedParam] : llvm::enumerate(diffParams)) {
                newParams[paramIdx] =
                    loadThenIncrementCounter(rewriter, paramCounter, paramsTensor);
            }
            MutableOperandRange range{gateOp, static_cast<unsigned>(gateOp.getDiffOperandIdx()),
                                      static_cast<unsigned>(diffParams.size())};
            range.assign(newParams);
    });

    // This function is the point where we can remove the classical preprocessing as a later
    // optimization.
    return modifiedCallee;
}

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
        callArgs.push_back(numParams);
        ValueRange quantumGradients =
            rewriter.create<func::CallOp>(loc, qGradFn, callArgs).getResults();

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
