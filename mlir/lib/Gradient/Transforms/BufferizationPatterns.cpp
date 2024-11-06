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

#include "iostream"
#include "llvm/Support/raw_ostream.h"

#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Index/IR/IndexOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"

#include "Gradient/IR/GradientOps.h"
#include "Gradient/Transforms/Passes.h"
#include "Gradient/Utils/GradientShape.h"

using namespace mlir;
using namespace catalyst::gradient;

namespace {

Value generateAllocation(OpBuilder &builder, Location loc, Value reference)
{
    auto memrefType = cast<MemRefType>(reference.getType());
    // Get dynamic dimension sizes from the provided reference value if necessary.
    SmallVector<Value> dynamicDims;
    if (!memrefType.hasStaticShape()) {
        for (int64_t dim = 0; dim < memrefType.getRank(); dim++) {
            if (memrefType.isDynamicDim(dim)) {
                Value dimIndex = builder.create<index::ConstantOp>(loc, dim);
                dynamicDims.push_back(builder.create<memref::DimOp>(loc, reference, dimIndex));
            }
        }
    }

    return builder.create<memref::AllocOp>(loc, memrefType, dynamicDims);
}

/// Helper function to generate a set of memref allocations.
///
/// The allocation size and shape is deduced from a list of existing memref values.
///
void generateAllocations(PatternRewriter &rewriter, Location loc,
                         SmallVectorImpl<Value> &allocations, ValueRange referenceValues)
{
    for (Value memref : referenceValues) {
        allocations.push_back(
            generateAllocation(rewriter, loc, cast<TypedValue<MemRefType>>(memref)));
    }
}

class BufferizeAdjointOp : public OpConversionPattern<AdjointOp> {
  public:
    using OpConversionPattern::OpConversionPattern;

    LogicalResult matchAndRewrite(AdjointOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override
    {
        SmallVector<Type> resTypes;
        if (failed(getTypeConverter()->convertTypes(op.getResultTypes(), resTypes)))
            return failure();

        Location loc = op.getLoc();
        Value gradSize = op.getGradSize();
        SmallVector<Value> memrefValues;
        for (Type resType : resTypes) {
            MemRefType memrefType = cast<MemRefType>(resType);
            Value memrefValue = rewriter.create<memref::AllocOp>(loc, memrefType, gradSize);
            memrefValues.push_back(memrefValue);
        }

        rewriter.create<AdjointOp>(loc, TypeRange{}, op.getCalleeAttr(), adaptor.getGradSize(),
                                   adaptor.getArgs(), memrefValues);
        rewriter.replaceOp(op, memrefValues);
        return success();
    }
};

class BufferizeBackpropOp : public OpConversionPattern<BackpropOp> {
  public:
    using OpConversionPattern::OpConversionPattern;

    LogicalResult matchAndRewrite(BackpropOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override
    {
        Location loc = op.getLoc();
        SmallVector<Value> gradients;
        SmallVector<Value> argShadows;
        // Conceptually a map from scalar result indices (w.r.t. other scalars) to the position in
        // the overall list of returned gradients.
        // For instance, a backprop op that returns (tensor, f64, tensor, f64, f64) will have
        // scalarIndices = {1, 3, 4}.
        SmallVector<unsigned> scalarIndices;
        SmallVector<Type> scalarReturnTypes;
        std::vector<Value> diffArgs =
            computeDiffArgs(adaptor.getArgs(), op.getDiffArgIndicesAttr());
        for (const auto &[idx, diffArg] : llvm::enumerate(diffArgs)) {
            // Allocate buffers to place the differentiation results (gradients) into. Enzyme refers
            // to these as shadow arguments. There is one result for each differentiable MemRef
            // argument, with a matching shape and type.
            if (isa<MemRefType>(diffArg.getType())) {
                Value shadow = generateAllocation(rewriter, loc, diffArg);
                gradients.push_back(shadow);
                argShadows.push_back(shadow);
            }
            else if (isa<FloatType>(diffArg.getType())) {
                scalarReturnTypes.push_back(diffArg.getType());
                scalarIndices.push_back(idx);
                // Put a null placeholder value that will be filled in with the result of the
                // bufferized BackpropOp.
                gradients.push_back(Value());
            }
        }

        // Enzyme requires buffers for the primal outputs as well, even though we don't need their
        // values. We'll mark them dupNoNeed later on to allow Enzyme to optimize away their
        // computation.
        SmallVector<Value> calleeResults, resShadows;
        ValueRange cotangents = adaptor.getCotangents();
        generateAllocations(rewriter, loc, calleeResults, cotangents);
        // Enzyme mutates the result shadows but the cotangent tensors must be immutable, so we
        // create copies to pass into Enzyme. Concretely, this issue pops up with multiple
        // BackpropOps that have the same cotangent tensor due to a CSE effect from one-shot
        // bufferization.
        generateAllocations(rewriter, loc, resShadows, cotangents);
        for (const auto &[cotangent, resShadow] : llvm::zip(cotangents, resShadows)) {
            rewriter.create<memref::CopyOp>(loc, cotangent, resShadow);
        }

        DenseIntElementsAttr diffArgIndicesAttr = adaptor.getDiffArgIndices().value_or(nullptr);
        auto bufferizedBackpropOp = rewriter.create<BackpropOp>(
            loc, TypeRange{}, scalarReturnTypes, op.getCalleeAttr(), adaptor.getArgs(), argShadows,
            calleeResults, resShadows, diffArgIndicesAttr, op.getKeepValueResultsAttr());

        // Fill in the null placeholders.
        for (const auto &[idx, scalarResult] :
             llvm::enumerate(bufferizedBackpropOp.getGradients())) {
            gradients[scalarIndices[idx]] = scalarResult;
        }

        // BackpropOp can return two results for value_and_grad: values and gradients
        // or only one for grad: gradients
        SmallVector<Value> results;
        {
            // If we are lowering a value_and_grad operation, then take values from the
            // calleeResults
            if (!op.getVals().empty()) {
                results.insert(results.end(), calleeResults.begin(), calleeResults.end());
            }
            results.insert(results.end(), gradients.begin(), gradients.end());
        }

        rewriter.replaceOp(op, results);
        return success();
    }
};

struct BufferizeForwardOp : public OpConversionPattern<ForwardOp> {
    using OpConversionPattern::OpConversionPattern;

    LogicalResult match(ForwardOp op) const override
    {
        // Only match with functions that are empty (i.e., just declarations, not definitions)
        if (!op.empty()) {
            return failure();
        }

        // Only match here if we have all memref arguments and return values.
        if (llvm::any_of(op.getArgumentTypes(),
                         [](Type argType) { return !isa<MemRefType>(argType); })) {
            return failure();
        }
        if (llvm::any_of(op.getResultTypes(),
                         [](Type argType) { return !isa<MemRefType>(argType); })) {
            return failure();
        }

        return success();
    }

    void rewrite(ForwardOp op, OpAdaptor adaptor,
                 ConversionPatternRewriter &rewriter) const override
    {
        auto argc = op.getArgc();
        auto resc = op.getResc();
        SmallVector<Value> inputs;
        SmallVector<Value> differentials;
        SmallVector<Value> outputs;
        SmallVector<Value> cotangents;

        Block *block;
        rewriter.modifyOpInPlace(op, [&] { block = op.addEntryBlock(); });

        PatternRewriter::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointToStart(block);
        auto params = op.getArguments();

        for (size_t i = 0; i < argc * 2; i++) {
            bool isDup = (i % 2) != 0;
            Value val = params[i];
            isDup ? differentials.push_back(val) : inputs.push_back(val);
        }

        auto upperLimit = (argc * 2) + (resc * 2);
        for (size_t i = argc * 2; i < upperLimit; i++) {
            bool isDup = (i % 2) != 0;
            Value val = params[i];
            isDup ? cotangents.push_back(val) : outputs.push_back(val);
        }

        auto implAttr = adaptor.getImplementationAttr();
        auto impl = adaptor.getImplementation();
        auto implOp = SymbolTable::lookupNearestSymbolFrom<FunctionOpInterface>(op, implAttr);
        auto implResTy = implOp.getResultTypes();
        Location loc = op.getLoc();

        SmallVector<Value> tensorInputs;
        for (auto input : inputs) {
            Value tensorIn = rewriter.create<bufferization::ToTensorOp>(loc, input);
            tensorInputs.push_back(tensorIn);
        }

        auto callOp = rewriter.create<func::CallOp>(loc, impl, implResTy, tensorInputs);
        SmallVector<Value> tensorOutputs(callOp.getResults());

        for (auto [memrefOutput, tensorOutput] : llvm::zip(outputs, tensorOutputs)) {
            Value castVal = rewriter.create<bufferization::ToMemrefOp>(loc, memrefOutput.getType(),
                                                                       tensorOutput);
            rewriter.create<memref::CopyOp>(loc, castVal, memrefOutput);
        }

        auto tapeCount = op.getTape();
        SmallVector<Value> tapeOutputs;
        tapeOutputs.insert(tapeOutputs.begin(), tensorOutputs.end() - tapeCount,
                           tensorOutputs.end());

        SmallVector<Value> tapeMemrefOutputs;
        for (auto [tapeTensorOutput, memrefTapeOutput] :
             llvm::zip(tapeOutputs, op.getResultTypes())) {
            Value castVal =
                rewriter.create<bufferization::ToMemrefOp>(loc, memrefTapeOutput, tapeTensorOutput);
            tapeMemrefOutputs.push_back(castVal);
        }

        auto F = rewriter.getIntegerAttr(rewriter.getI1Type(), 0);
        rewriter.create<catalyst::gradient::ReturnOp>(loc, tapeMemrefOutputs, F);
    }
};

struct BufferizeReverseOp : public OpConversionPattern<ReverseOp> {
    using OpConversionPattern::OpConversionPattern;

    LogicalResult match(ReverseOp op) const override
    {
        // Only match with functions that are empty (i.e., just declarations, not definitions)
        if (!op.empty()) {
            return failure();
        }

        // Only match here if we have all memref arguments and return values.
        if (llvm::any_of(op.getArgumentTypes(),
                         [](Type argType) { return !isa<MemRefType>(argType); })) {
            return failure();
        }
        if (llvm::any_of(op.getResultTypes(),
                         [](Type argType) { return !isa<MemRefType>(argType); })) {
            return failure();
        }

        return success();
    }

    void rewrite(ReverseOp op, OpAdaptor adaptor,
                 ConversionPatternRewriter &rewriter) const override
    {
        auto argc = op.getArgc();
        auto resc = op.getResc();
        SmallVector<Value> inputs;
        SmallVector<Value> differentials;
        SmallVector<Value> outputs;
        SmallVector<Value> cotangents;
        SmallVector<Value> tapeElements;

        Block *block;
        rewriter.modifyOpInPlace(op, [&] { block = op.addEntryBlock(); });

        PatternRewriter::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointToStart(block);
        auto params = op.getArguments();

        for (size_t i = 0; i < argc * 2; i++) {
            bool isDup = (i % 2) != 0;
            Value val = params[i];
            isDup ? differentials.push_back(val) : inputs.push_back(val);
        }

        auto upperLimit = (argc * 2) + (resc * 2);
        for (size_t i = argc * 2; i < upperLimit; i++) {
            bool isDup = (i % 2) != 0;
            Value val = params[i];
            isDup ? cotangents.push_back(val) : outputs.push_back(val);
        }

        auto tapeCount = op.getTape();
        auto uppestLimit = upperLimit + tapeCount;
        for (size_t i = upperLimit; i < uppestLimit; i++) {
            tapeElements.push_back(params[i]);
        }

        auto implAttr = adaptor.getImplementationAttr();
        auto impl = adaptor.getImplementation();
        auto implOp = SymbolTable::lookupNearestSymbolFrom<FunctionOpInterface>(op, implAttr);
        auto implResTy = implOp.getResultTypes();
        Location loc = op.getLoc();

        SmallVector<Value> tensorInputs;
        for (auto tapeElement : tapeElements) {
            Value tensorIn = rewriter.create<bufferization::ToTensorOp>(loc, tapeElement);
            tensorInputs.push_back(tensorIn);
        }

        for (auto cotangent : cotangents) {
            Value tensorIn = rewriter.create<bufferization::ToTensorOp>(loc, cotangent);
            tensorInputs.push_back(tensorIn);
        }

        auto callOp = rewriter.create<func::CallOp>(loc, impl, implResTy, tensorInputs);
        SmallVector<Value> tensorOutputs(callOp.getResults());

        for (auto [differential, tensorOutput] : llvm::zip(differentials, tensorOutputs)) {
            Value castVal = rewriter.create<bufferization::ToMemrefOp>(loc, differential.getType(),
                                                                       tensorOutput);
            rewriter.create<memref::CopyOp>(loc, castVal, differential);
        }

        auto T = rewriter.getIntegerAttr(rewriter.getI1Type(), 1);
        rewriter.create<catalyst::gradient::ReturnOp>(loc, ValueRange{}, T);
    }
};

class BufferizeReturnOp : public OpConversionPattern<ReturnOp> {
  public:
    using OpConversionPattern::OpConversionPattern;

    LogicalResult matchAndRewrite(ReturnOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override
    {
        if (!llvm::any_of(op->getOperands().getType(),
                          [](Type argType) { return isa<TensorType>(argType); })) {
            return failure();
        }

        auto outTypes = op->getParentOfType<FunctionOpInterface>().getResultTypes();

        if (llvm::any_of(outTypes, [](Type argType) { return isa<TensorType>(argType); })) {
            return failure();
        }

        Location loc = op->getLoc();
        auto returnOperands = op->getOpOperands();
        SmallVector<Value> returnValues;
        for (auto [outType, returnOperand] : llvm::zip(outTypes, returnOperands)) {
            Value returnVal = returnOperand.get();
            auto tensorType = dyn_cast<TensorType>(returnVal.getType());

            if (!tensorType) {
                returnValues.push_back(returnVal);
                continue;
            }
            Value toMemrefOp = rewriter.create<bufferization::ToMemrefOp>(loc, outType, returnVal);
            returnValues.push_back(toMemrefOp);
        }

        rewriter.modifyOpInPlace(op, [&] { op->setOperands(returnValues); });

        return success();
    }
};

} // namespace

namespace catalyst {
namespace gradient {

void populateBufferizationPatterns(TypeConverter &typeConverter, RewritePatternSet &patterns)
{
    patterns.add<BufferizeAdjointOp>(typeConverter, patterns.getContext());
    patterns.add<BufferizeBackpropOp>(typeConverter, patterns.getContext());
    patterns.add<BufferizeReturnOp>(typeConverter, patterns.getContext());
    patterns.add<BufferizeForwardOp>(typeConverter, patterns.getContext());
    patterns.add<BufferizeReverseOp>(typeConverter, patterns.getContext());
}

} // namespace gradient
} // namespace catalyst
