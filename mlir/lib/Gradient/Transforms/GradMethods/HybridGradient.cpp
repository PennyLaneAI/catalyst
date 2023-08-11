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
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/Transforms/Bufferize.h"
#include "mlir/Dialect/Index/IR/IndexOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"

#include "ClassicalJacobian.hpp"
#include "HybridGradient.hpp"

#include "Catalyst/Utils/CallGraph.h"
#include "Gradient/Utils/GradientShape.h"
#include "Quantum/IR/QuantumInterfaces.h"
#include "Quantum/IR/QuantumOps.h"
#include "Quantum/Utils/RemoveQuantumMeasurements.h"

using namespace mlir;

namespace catalyst {
namespace gradient {
using llvm::errs;
func::FuncOp genQuantumGradient(PatternRewriter &rewriter, Location loc, func::FuncOp qgradFn,
                                TypeConverter &typeConverter)
{
    // A version of the qgrad function that is compatible with the custom gradient interface in
    // Enzyme.

    // inputs: arguments and their shadows
    std::string fnName = (qgradFn.getName() + ".qvjp").str();
    func::FuncOp quantumGradient =
        SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(qgradFn, rewriter.getStringAttr(fnName));
    if (quantumGradient) {
        return quantumGradient;
    }

    SmallVector<Type> fnArgTypes;
    MLIRContext *ctx = rewriter.getContext();
    LLVMTypeConverter llvmTypeConverter(ctx);
    auto indexType = llvmTypeConverter.getIndexType();
    auto ptrType = LLVM::LLVMPointerType::get(ctx);
    for (Type argType : qgradFn.getArgumentTypes()) {
        // All tensor types need handling unfortunately because we can't mark int tensors as
        // constant
        if (auto shapedType = dyn_cast<ShapedType>(argType)) {
            // We'll need to unpack the memrefs here, likely requiring an analogue of the
            // MemRefDescriptor class.
            int64_t rank = shapedType.getRank();
            // Allocated and shadow allocated, aligned and shadow aligned
            fnArgTypes.append({ptrType, ptrType, ptrType, ptrType});

            // Offset, sizes, and strides
            fnArgTypes.push_back(indexType);
            for (int64_t dim = 0; dim < rank; dim++) {
                fnArgTypes.append({indexType, indexType});
            }
        }
        else {
            fnArgTypes.push_back(argType);
        }
    }
    FunctionType fnType = rewriter.getFunctionType(fnArgTypes, {});

    quantumGradient = rewriter.create<func::FuncOp>(loc, fnName, fnType);
    quantumGradient.setPrivate();

    rewriter.cloneRegionBefore(qgradFn.getBody(), quantumGradient.getBody(), quantumGradient.end());
    Block &entryBlock = quantumGradient.getFunctionBody().front();
    PatternRewriter::InsertionGuard insertionGuard(rewriter);
    rewriter.setInsertionPointToStart(&entryBlock);
    // Need to replace tensor types in the block
    SmallVector<BlockArgument> oldArgs{entryBlock.getArguments()};
    size_t argIdx = 0;
    for (BlockArgument arg : oldArgs) {
        if (auto shapedType = dyn_cast<ShapedType>(arg.getType())) {
            int64_t rank = shapedType.getRank();
            SmallVector<Value> unpackedValues;
            SmallVector<Value> unpackedShadow;
            Value allocatedPtr = entryBlock.insertArgument(argIdx++, ptrType, loc);
            Value shadowAllocated = entryBlock.insertArgument(argIdx++, ptrType, loc);
            Value alignedPtr = entryBlock.insertArgument(argIdx++, ptrType, loc);
            Value shadowAligned = entryBlock.insertArgument(argIdx++, ptrType, loc);

            unpackedValues.append({allocatedPtr, alignedPtr});
            unpackedShadow.append({shadowAllocated, shadowAligned});

            // The offsets, sizes, and strides are shared between the primal and shadows
            SmallVector<Value> sizeValues;
            sizeValues.push_back(entryBlock.insertArgument(argIdx++, indexType, loc));
            for (int64_t dim = 0; dim < rank; dim++) {
                sizeValues.push_back(entryBlock.insertArgument(argIdx++, indexType, loc));
            }
            for (int64_t dim = 0; dim < rank; dim++) {
                sizeValues.push_back(entryBlock.insertArgument(argIdx++, indexType, loc));
            }

            unpackedValues.insert(unpackedValues.end(), sizeValues.begin(), sizeValues.end());
            unpackedShadow.insert(unpackedShadow.end(), sizeValues.begin(), sizeValues.end());

            Value reconstructedTensor =
                rewriter
                    .create<UnrealizedConversionCastOp>(
                        arg.getLoc(), typeConverter.convertType(arg.getType()), unpackedValues)
                    .getResult(0);
            reconstructedTensor =
                rewriter.create<bufferization::ToTensorOp>(arg.getLoc(), reconstructedTensor);

            if (arg == oldArgs.back()) {
                // The last argument should be replaced with its shadow
                Value reconstructedShadow = rewriter
                                                .create<UnrealizedConversionCastOp>(
                                                    arg.getLoc(), arg.getType(), unpackedShadow)
                                                .getResult(0);
                arg.replaceAllUsesWith(reconstructedShadow);
            }
            else {
                arg.replaceAllUsesWith(reconstructedTensor);
            }
            entryBlock.eraseArgument(argIdx);
        }
        else {
            argIdx++;
        }
    }

    quantumGradient.walk([&](func::ReturnOp returnOp) { returnOp.getOperandsMutable().clear(); });
    return quantumGradient;
}

func::FuncOp genAugmentedForwardPass(PatternRewriter &rewriter, Location loc, func::FuncOp callee)
{
    // The main difference of the augmented forward pass is that it returns any values that need to
    // be cached from the quantum side (which is usually nothing). If we don't use destination
    // passing style and return tensors, we need to return zeroed out tensors here.
    std::string fnName = (callee.getName() + ".augfwd").str();
    SmallVector<Type> fnArgTypes(callee.getArgumentTypes());
    MLIRContext *ctx = rewriter.getContext();
    SmallVector<Type> resultTypes{LLVM::LLVMStructType::getLiteral(ctx, {})};
    // LLVM Assumes a single return value
    assert(callee.getResultTypes().size() == 1 && "Assumed callee has a single return");
    resultTypes.push_back(callee.getResultTypes()[0]);
    // We also need to return the shadow here
    resultTypes.push_back(callee.getResultTypes()[0]);

    auto paramsBufferType = MemRefType::get({ShapedType::kDynamic}, rewriter.getF64Type());
    fnArgTypes.push_back(paramsBufferType);
    FunctionType fnType = rewriter.getFunctionType(fnArgTypes, resultTypes);

    func::FuncOp augmentedForwardPass =
        SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(callee, rewriter.getStringAttr(fnName));
    if (augmentedForwardPass) {
        return augmentedForwardPass;
    }

    augmentedForwardPass = rewriter.create<func::FuncOp>(loc, fnName, fnType);
    augmentedForwardPass.setPrivate();
    rewriter.cloneRegionBefore(callee.getBody(), augmentedForwardPass.getBody(),
                               augmentedForwardPass.end());
    Block &entryBlock = augmentedForwardPass.getFunctionBody().front();
    entryBlock.addArgument(paramsBufferType, loc);

    augmentedForwardPass.walk([&](func::ReturnOp returnOp) {
        Location loc = returnOp.getLoc();
        PatternRewriter::InsertionGuard insertionGuard(rewriter);
        rewriter.setInsertionPoint(returnOp);
        Value emptyStruct =
            rewriter.create<LLVM::UndefOp>(loc, LLVM::LLVMStructType::getLiteral(ctx, {}));
        Value primalReturn = returnOp.getOperand(0);
        auto resultType = cast<RankedTensorType>(primalReturn.getType());
        Value shadow = rewriter.create<tensor::EmptyOp>(loc, resultType.getShape(),
                                                        resultType.getElementType());
        Value zero =
            rewriter.create<arith::ConstantFloatOp>(loc, APFloat(0.0), rewriter.getF64Type());
        shadow = rewriter.create<linalg::FillOp>(loc, zero, shadow).getResult(0);
        returnOp.getOperandsMutable().assign({emptyStruct, primalReturn, shadow});
    });
    // This is the point where we can remove the classical preprocessing as a later optimization.
    return augmentedForwardPass;
}

LogicalResult HybridGradientLowering::matchAndRewrite(GradOp op, PatternRewriter &rewriter) const
{
    Location loc = op.getLoc();
    func::FuncOp callee =
        SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(op, op.getCalleeAttr());
    rewriter.setInsertionPointAfter(callee);
    // Replace calls with the QNode with the split QNode in the callee.
    auto clonedCallee = cast<func::FuncOp>(rewriter.clone(*callee));
    std::string clonedCalleeName = (callee.getName() + ".cloned").str();
    clonedCallee.setName(clonedCalleeName);
    SmallPtrSet<Operation *, 4> qnodes;
    SymbolTableCollection symbolTable;
    auto isQNode = [](func::FuncOp funcOp) { return funcOp->hasAttr("qnode"); };
    if (isQNode(clonedCallee)) {
        qnodes.insert(callee);
    }
    else {
        traverseCallGraph(clonedCallee, symbolTable, [&qnodes, &isQNode](func::FuncOp funcOp) {
            if (isQNode(funcOp)) {
                qnodes.insert(funcOp);
            }
        });
    }

    for (Operation *qnodeOp : qnodes) {
        auto qnode = cast<func::FuncOp>(qnodeOp);

        // In order to allocate memory for various tensors relating to the number of gate parameters
        // at runtime we run a function that merely counts up for each gate parameter encountered.
        func::FuncOp paramCountFn = genParamCountFunction(rewriter, loc, qnode);
        func::FuncOp qnodeWithParams = genQNodeWithParams(rewriter, loc, qnode);
        func::FuncOp qnodeSplit = genSplitPreprocessed(rewriter, loc, qnode, qnodeWithParams);

        // This attribute tells downstream patterns that this QNode requires the registration of a
        // custom quantum gradient.
        qnode->setAttr("withparams", FlatSymbolRefAttr::get(qnodeWithParams));
        // Enzyme will fail if this function gets inlined.
        qnodeWithParams->setAttr("passthrough",
                                 rewriter.getArrayAttr(rewriter.getStringAttr("noinline")));

        // Replace calls to the original QNode with calls to the split QNode
        if (isQNode(clonedCallee)) {
            PatternRewriter::InsertionGuard insertionGuard(rewriter);
            rewriter.eraseBlock(&clonedCallee.getFunctionBody().front());
            Block *entryBlock = clonedCallee.addEntryBlock();

            rewriter.setInsertionPointToStart(entryBlock);
            Value paramCount =
                rewriter.create<func::CallOp>(loc, paramCountFn, clonedCallee.getArguments())
                    .getResult(0);
            SmallVector<Value> splitArgs{clonedCallee.getArguments()};
            splitArgs.push_back(paramCount);

            auto splitCall = rewriter.create<func::CallOp>(loc, qnodeSplit, splitArgs);
            rewriter.create<func::ReturnOp>(loc, splitCall.getResults());
        }
        else {
            traverseCallGraph(clonedCallee, symbolTable, [&](func::FuncOp funcOp) {
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

    rewriter.setInsertionPoint(op);
    std::vector<size_t> diffArgIndices = computeDiffArgIndices(op.getDiffArgIndices());
    SmallVector<Value> backpropResults{op.getNumResults()};
    // Iterate over the primal results
    for (const auto &[cotangentIdx, primalResult] :
         llvm::enumerate(clonedCallee.getResultTypes())) {
        // There is one Jacobian per distinct differential argument.
        SmallVector<Value> jacobians;
        for (unsigned argIdx = 0; argIdx < diffArgIndices.size(); argIdx++) {
            Type jacobianType =
                op.getResultTypes()[argIdx * clonedCallee.getNumResults() + cotangentIdx];
            jacobians.push_back(
                rewriter.create<tensor::EmptyOp>(loc, jacobianType, /*dynamicSizes=*/ValueRange{}));
        }

        auto primalTensorResultType = cast<RankedTensorType>(primalResult);
        assert(primalTensorResultType.hasStaticShape());

        ArrayRef<int64_t> shape = primalTensorResultType.getShape();
        // Compute the strides in reverse
        unsigned product = 1;
        SmallVector<unsigned> strides;
        for (int64_t dim = primalTensorResultType.getRank() - 1; dim >= 0; dim--) {
            strides.push_back(product);
            product *= shape[dim];
        }
        std::reverse(strides.begin(), strides.end());

        Value zero = rewriter.create<arith::ConstantOp>(
            loc, FloatAttr::get(primalTensorResultType.getElementType(), 0.0));
        Value one = rewriter.create<arith::ConstantOp>(
            loc, FloatAttr::get(primalTensorResultType.getElementType(), 1.0));
        Value zeroTensor = rewriter.create<tensor::EmptyOp>(loc, primalTensorResultType,
                                                            /*dynamicSizes=*/ValueRange{});
        zeroTensor = rewriter.create<linalg::FillOp>(loc, zero, zeroTensor).getResult(0);

        for (unsigned flatIdx = 0; flatIdx < primalTensorResultType.getNumElements(); flatIdx++) {
            // Unflatten the tensor indices
            SmallVector<Value> indices;
            for (int64_t dim = 0; dim < primalTensorResultType.getRank(); dim++) {
                indices.push_back(
                    rewriter.create<index::ConstantOp>(loc, flatIdx / strides[dim] % shape[dim]));
            }

            SmallVector<Value> cotangents;
            Value cotangent = rewriter.create<tensor::InsertOp>(loc, one, zeroTensor, indices);
            for (const auto &[resultIdx, primalResultType] :
                 llvm::enumerate(clonedCallee.getResultTypes())) {
                if (resultIdx == cotangentIdx) {
                    cotangents.push_back(cotangent);
                }
                else {
                    // Push back a zeroed-out cotangent
                    Value zeroTensor =
                        rewriter.create<tensor::EmptyOp>(loc, primalResultType, ValueRange{});
                    cotangents.push_back(
                        rewriter.create<linalg::FillOp>(loc, zero, zeroTensor).getResult(0));
                }
            }

            auto backpropOp = rewriter.create<gradient::BackpropOp>(
                loc, computeBackpropTypes(clonedCallee, diffArgIndices), clonedCallee.getName(),
                op.getArgOperands(),
                /*arg_shadows=*/ValueRange{}, /*primal results=*/ValueRange{}, cotangents,
                op.getDiffArgIndicesAttr());

            // Backprop gives a gradient of a single output entry w.r.t. all active inputs.
            for (const auto &[backpropIdx, jacobianSlice] :
                 llvm::enumerate(backpropOp.getResults())) {
                auto sliceType = cast<RankedTensorType>(jacobianSlice.getType());
                size_t sliceRank = sliceType.getRank();
                auto jacobianType = cast<RankedTensorType>(jacobians[backpropIdx].getType());
                size_t jacobianRank = jacobianType.getRank();
                if (sliceRank < jacobianRank) {
                    // Offsets are [...indices] + [0] * rank of backprop result
                    SmallVector<OpFoldResult> offsets;
                    offsets.append(indices.begin(), indices.end());
                    offsets.append(sliceRank, rewriter.getIndexAttr(0));

                    // Sizes are [1] * (jacobianRank - sliceRank) + [...sliceShape]
                    SmallVector<OpFoldResult> sizes;
                    sizes.append(jacobianRank - sliceRank, rewriter.getIndexAttr(1));
                    for (int64_t dim : sliceType.getShape()) {
                        sizes.push_back(rewriter.getIndexAttr(dim));
                    }

                    // Strides are [1] * jacobianRank
                    SmallVector<OpFoldResult> strides{jacobianRank, rewriter.getIndexAttr(1)};

                    jacobians[backpropIdx] = rewriter.create<tensor::InsertSliceOp>(
                        loc, jacobianSlice, jacobians[backpropIdx], offsets, sizes, strides);
                }
                else {
                    jacobians[backpropIdx] = jacobianSlice;
                }
                backpropResults[backpropIdx * clonedCallee.getNumResults() + cotangentIdx] =
                    jacobians[backpropIdx];
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

    modifiedCallee.walk([&](Operation *op) {
        if (auto gateOp = dyn_cast<quantum::DifferentiableGate>(op)) {
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
        }
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
    auto callee =
        SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(gradOp, gradOp.getCalleeAttr());
    // For each QNode, generate a wrapper containing classical preprocessing that then calls a
    // function that accepts the parameters. This conceptually is splitting the QNode into classical
    // preprocessing and quantum parts that end in a measurement.
    SmallVector<func::FuncOp> qnodes;
    if (callee->hasAttr("qnode")) {
        qnodes.push_back(callee);
    }
    else {
        callee.walk([&qnodes](func::CallOp callOp) {
            auto callee =
                SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(callOp, callOp.getCalleeAttr());
            if (callee->hasAttr("qnode")) {
                qnodes.push_back(callee);
            }
        });
    }

    // for (func::FuncOp qnode : qnodes) {
    //     func::FuncOp withParams = genQNodeWithParams(rewriter, qnode.getLoc(), qnode);
    //     func::FuncOp splitPreprocessing =
    //         genSplitPreprocessed(rewriter, qnode.getLoc(), qnode, withParams);
    // }

    // The modified callee
    // func::FuncOp modifiedCallee = genModifiedCallee(rewriter, loc, callee);
    // func::FuncOp primal = genEnzymeWrapper(rewriter, loc, callee, modifiedCallee);
    // func::FuncOp augmented = genAugmentedForwardPass(rewriter, loc, callee);
    // func::FuncOp gradient = genQuantumGradient(rewriter, loc, qGradFn, typeConverter);
    // modifiedCallee->setAttr("gradient.augment", augmented.getNameAttr());
    // modifiedCallee->setAttr("gradient.vjp", gradient.getNameAttr());

    func::FuncOp fullGradFn =
        SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(gradOp, rewriter.getStringAttr(fnName));
    if (!fullGradFn) {
        PatternRewriter::InsertionGuard insertGuard(rewriter);
        rewriter.setInsertionPointAfter(qGradFn);

        fullGradFn =
            rewriter.create<func::FuncOp>(loc, fnName, fnType, visibility, nullptr, nullptr);
        Block *entryBlock = fullGradFn.addEntryBlock();
        rewriter.setInsertionPointToStart(entryBlock);

        // // Collect arguments and invoke the classical jacobian and quantum gradient functions.
        // SmallVector<Value> callArgs(fullGradFn.getArguments());

        // Value numParams = rewriter.create<func::CallOp>(loc, paramCountFn,
        // callArgs).getResult(0); SmallVector<Value> qGradArgs(callArgs);
        // qGradArgs.push_back(numParams);
        // ValueRange quantumGradients =
        //     rewriter.create<func::CallOp>(loc, qGradFn, qGradArgs).getResults();

        // DenseIntElementsAttr diffArgIndicesAttr = gradOp.getDiffArgIndices().value_or(nullptr);

        // auto resultsBackpropTypes = computeBackpropTypes(argMapFn, diffArgIndices);
        // // Compute hybrid gradients via Enzyme
        // std::vector<Value> hybridGradients;
        // int j = 0;
        // // Loop over the measurements
        // for (Value quantumGradient : quantumGradients) {
        //     Type resultType = gradOp.getResult(j).getType();
        //     Value result = rewriter.create<tensor::EmptyOp>(loc, resultType, ValueRange{});
        //     auto rankResult = resultType.cast<RankedTensorType>().getRank();
        //     auto shapeResult = resultType.cast<RankedTensorType>().getShape();
        //     j++;

        //     std::vector<BackpropOp> intermediateGradients;
        //     auto rank = quantumGradient.getType().cast<RankedTensorType>().getRank();

        //     if (rank > 1) {
        //         std::vector<int64_t> sizes =
        //             quantumGradient.getType().cast<RankedTensorType>().getShape();

        //         std::vector<std::vector<int64_t>> allOffsets;
        //         std::vector<int64_t> cutOffset(sizes.begin() + 1, sizes.end());

        //         std::vector<int64_t> currentOffset(cutOffset.size(), 0);

        //         int64_t totalOutcomes = 1;
        //         for (int64_t dim : cutOffset) {
        //             totalOutcomes *= dim;
        //         }

        //         for (int64_t outcome = 0; outcome < totalOutcomes; outcome++) {
        //             allOffsets.push_back(currentOffset);

        //             for (int64_t i = cutOffset.size() - 1; i >= 0; i--) {
        //                 currentOffset[i]++;
        //                 if (currentOffset[i] < cutOffset[i]) {
        //                     break;
        //                 }
        //                 currentOffset[i] = 0;
        //             }
        //         }

        //         std::vector<int64_t> strides(rank, 1);
        //         std::vector<Value> dynStrides = {};

        //         std::vector<Value> dynOffsets = {};

        //         std::vector<Value> dynSizes;

        //         for (size_t index = 0; index < sizes.size(); ++index) {
        //             if (index == 0) {
        //                 Value idx = rewriter.create<index::ConstantOp>(loc, index);
        //                 Value dimSize = rewriter.create<tensor::DimOp>(loc, quantumGradient,
        //                 idx); dynSizes.push_back(dimSize);
        //             }
        //             else {
        //                 sizes[index] = 1;
        //             }
        //         }
        //         for (auto offsetRight : allOffsets) {
        //             std::vector<int64_t> offsets{0};
        //             offsets.insert(offsets.end(), offsetRight.begin(), offsetRight.end());
        //             auto rankReducedType =
        //                 tensor::ExtractSliceOp::inferCanonicalRankReducedResultType(
        //                     1, quantumGradient.getType().cast<RankedTensorType>(), offsets,
        //                     sizes, strides) .cast<RankedTensorType>();
        //             Value extractQuantumGradient = rewriter.create<tensor::ExtractSliceOp>(
        //                 loc, rankReducedType, quantumGradient, dynOffsets, dynSizes, dynStrides,
        //                 offsets, sizes, strides);
        //             BackpropOp backpropOp = rewriter.create<BackpropOp>(
        //                 loc, resultsBackpropTypes, argMapFn.getName(), callArgs,
        //                 extractQuantumGradient, ValueRange{}, diffArgIndicesAttr);

        //             intermediateGradients.push_back(backpropOp);
        //         }
        //         for (size_t i = 0; i < resultsBackpropTypes.size(); i++) {
        //             // strides
        //             std::vector<int64_t> stridesSlice(rankResult, 1);

        //             for (int64_t index = 0; index < totalOutcomes; index++) {
        //                 auto intermediateGradient = intermediateGradients[index];
        //                 Value gradient = intermediateGradient.getResult(i);

        //                 Type gradientType = gradient.getType();
        //                 auto rankGradient = gradientType.cast<RankedTensorType>().getRank();

        //                 // sizes
        //                 std::vector<int64_t> sizesSlice{shapeResult};
        //                 for (int64_t sliceIndex = rankResult - 1; sliceIndex >= rankGradient;
        //                      sliceIndex--) {
        //                     sizesSlice[sliceIndex] = 1;
        //                 }

        //                 // offset
        //                 auto offsetSlice = allOffsets[index];
        //                 for (int64_t offsetIndex = 0; offsetIndex < rankGradient; offsetIndex++)
        //                 {
        //                     offsetSlice.insert(offsetSlice.begin(), 0);
        //                 }
        //                 result = rewriter.create<tensor::InsertSliceOp>(
        //                     loc, resultType, gradient, result, ValueRange{}, ValueRange{},
        //                     ValueRange{}, offsetSlice, sizesSlice, stridesSlice);
        //             }
        //             hybridGradients.push_back(result);
        //         }
        //     }
        //     else {
        //         BackpropOp backpropOp = rewriter.create<BackpropOp>(
        //             loc, resultsBackpropTypes, argMapFn.getName(), callArgs, quantumGradient,
        //             ValueRange{}, diffArgIndicesAttr);
        //         // Loop over params
        //         for (size_t i = 0; i < backpropOp.getNumResults(); i++) {
        //             Value result = backpropOp.getResult(i);
        //             hybridGradients.push_back(result);
        //         }
        //     }
        // }
        rewriter.create<func::ReturnOp>(loc, entryBlock->getArguments().take_front(2));
    }

    return fullGradFn;
}

} // namespace gradient
} // namespace catalyst
