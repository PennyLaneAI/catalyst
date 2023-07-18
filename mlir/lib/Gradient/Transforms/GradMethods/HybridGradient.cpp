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

#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/Transforms/Bufferize.h"
#include "mlir/Dialect/Index/IR/IndexOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"

#include "Gradient/Utils/GradientShape.h"
#include "Quantum/IR/QuantumInterfaces.h"
#include "Quantum/IR/QuantumOps.h"
#include "Quantum/Utils/RemoveQuantumMeasurements.h"

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
                    .create<UnrealizedConversionCastOp>(arg.getLoc(), arg.getType(), unpackedValues)
                    .getResult(0);

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

func::FuncOp genEnzymeWrapper(PatternRewriter &rewriter, Location loc, func::FuncOp callee,
                              func::FuncOp modifiedCallee)
{
    // Copied from the argmap function because it's very similar.
    // Define the properties of the classical preprocessing function.
    std::string fnName = callee.getSymName().str() + ".enzymewrapper";
    SmallVector<Type> fnArgTypes(callee.getArgumentTypes());
    auto paramsBufferType = MemRefType::get({ShapedType::kDynamic}, rewriter.getF64Type());
    // fnArgTypes.push_back(paramsBufferType);
    fnArgTypes.push_back(rewriter.getIndexType()); // parameter count

    // fnArgTypes.push_back()
    // Need this to be in destination passing style
    bufferization::BufferizeTypeConverter typeConverter;
    for (Type resultType : callee.getResultTypes()) {
        fnArgTypes.push_back(typeConverter.convertType(resultType));
    }

    FunctionType fnType = rewriter.getFunctionType(fnArgTypes, {});
    StringAttr visibility = rewriter.getStringAttr("private");

    func::FuncOp argMapFn =
        SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(callee, rewriter.getStringAttr(fnName));
    if (!argMapFn) {
        // First copy the original function as is, then we can replace all quantum ops by collecting
        // their gate parameters in a memory buffer instead. The size of this vector is passed as an
        // input to the new function.
        argMapFn = rewriter.create<func::FuncOp>(loc, fnName, fnType, visibility, nullptr, nullptr);
        rewriter.cloneRegionBefore(callee.getBody(), argMapFn.getBody(), argMapFn.end());
        Block &argMapBlock = argMapFn.getFunctionBody().front();
        SmallVector<Value> modifiedCalleeArgs{argMapBlock.getArguments()};

        Value paramCount = argMapBlock.addArgument(rewriter.getIndexType(), loc);
        SmallVector<Value> dpsOutputs;
        for (Type resultType : callee.getResultTypes()) {
            dpsOutputs.push_back(
                argMapBlock.addArgument(typeConverter.convertType(resultType), loc));
        }

        PatternRewriter::InsertionGuard insertGuard(rewriter);
        rewriter.setInsertionPointToStart(&argMapFn.getBody().front());
        Value paramsBuffer = rewriter.create<memref::AllocOp>(loc, paramsBufferType, paramCount);

        modifiedCalleeArgs.push_back(paramsBuffer);
        MemRefType paramsProcessedType = MemRefType::get({}, rewriter.getIndexType());
        Value paramsProcessed = rewriter.create<memref::AllocaOp>(loc, paramsProcessedType);
        Value cZero = rewriter.create<index::ConstantOp>(loc, 0);
        rewriter.create<memref::StoreOp>(loc, cZero, paramsProcessed);
        Value cOne = rewriter.create<index::ConstantOp>(loc, 1);

        argMapFn.walk([&](Operation *op) {
            // Insert gate parameters into the params buffer.
            if (auto gate = dyn_cast<quantum::DifferentiableGate>(op)) {
                PatternRewriter::InsertionGuard insertGuard(rewriter);
                rewriter.setInsertionPoint(gate);

                ValueRange diffParams = gate.getDiffParams();
                if (!diffParams.empty()) {
                    Value paramIdx = rewriter.create<memref::LoadOp>(loc, paramsProcessed);
                    for (auto param : diffParams) {
                        rewriter.create<memref::StoreOp>(loc, param, paramsBuffer, paramIdx);
                        paramIdx = rewriter.create<index::AddOp>(loc, paramIdx, cOne);
                    }
                    rewriter.create<memref::StoreOp>(loc, paramIdx, paramsProcessed);
                }

                rewriter.replaceOp(op, gate.getQubitOperands());
            }
            // Return ops should be preceded with calls to the modified quantum callee
            else if (auto returnOp = dyn_cast<func::ReturnOp>(op)) {
                PatternRewriter::InsertionGuard insertionGuard(rewriter);
                rewriter.setInsertionPoint(returnOp);
                auto modifiedCall =
                    rewriter.create<func::CallOp>(loc, modifiedCallee, modifiedCalleeArgs);

                // Copy over results
                for (const auto &[dpsOut, funcResult] :
                     llvm::zip(dpsOutputs, modifiedCall.getResults())) {
                    Value castedResult = rewriter.create<bufferization::ToMemrefOp>(
                        loc, typeConverter.convertType(funcResult.getType()), funcResult);
                    // Let's hope this doesn't break type analysis
                    rewriter.create<memref::CopyOp>(loc, castedResult, dpsOut);
                }
                returnOp.getOperandsMutable().clear();
            }
            // Erase redundant device specifications.
            else if (isa<quantum::DeviceOp>(op)) {
                rewriter.eraseOp(op);
            }
        });

        quantum::removeQuantumMeasurements(argMapFn);
    }

    return argMapFn;
}

func::FuncOp genModifiedCallee(PatternRewriter &rewriter, Location loc, func::FuncOp callee)
{
    // The callee requires two modifications, but the most important one is that it accepts the gate
    // parameters as an argument. This is so Enzyme will see that the gate params flow into the
    // custom quantum function.
    std::string fnName = (callee.getName() + ".withparams").str();
    SmallVector<Type> fnArgTypes(callee.getArgumentTypes());
    auto paramsBufferType = MemRefType::get({ShapedType::kDynamic}, rewriter.getF64Type());
    fnArgTypes.push_back(paramsBufferType);
    FunctionType fnType = rewriter.getFunctionType(fnArgTypes, callee.getResultTypes());

    func::FuncOp modifiedCallee =
        SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(callee, rewriter.getStringAttr(fnName));
    if (modifiedCallee) {
        return modifiedCallee;
    }

    modifiedCallee = rewriter.create<func::FuncOp>(loc, fnName, fnType);
    modifiedCallee.setPrivate();
    rewriter.cloneRegionBefore(callee.getBody(), modifiedCallee.getBody(), modifiedCallee.end());
    Block &entryBlock = modifiedCallee.getFunctionBody().front();
    entryBlock.addArgument(paramsBufferType, loc);

    // This is the point where we can remove the classical preprocessing as a later optimization.
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
    func::FuncOp modifiedCallee = genModifiedCallee(rewriter, loc, callee);
    // TODO: Pass this is in wherever it's required
    bufferization::BufferizeTypeConverter typeConverter;
    genEnzymeWrapper(rewriter, loc, callee, modifiedCallee);
    genAugmentedForwardPass(rewriter, loc, callee);
    genQuantumGradient(rewriter, loc, qGradFn, typeConverter);
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
