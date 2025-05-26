// Copyright 2024-2025 Xanadu Quantum Technologies Inc.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <algorithm> // std::find
#include <vector>

#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/IR/UnstructuredControlFlow.h"
#include "mlir/Dialect/Bufferization/Transforms/Bufferize.h"
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
#include "Gradient/Transforms/BufferizableOpInterfaceImpl.h"
#include "Gradient/Transforms/Passes.h"
#include "Gradient/Utils/GradientShape.h"

using namespace mlir;
using namespace catalyst::gradient;

/**
 * Implementation of the BufferizableOpInterface for use with one-shot bufferization.
 * For more information on the interface, refer to the documentation below:
 *  https://mlir.llvm.org/docs/Bufferization/#extending-one-shot-bufferize
 *  https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir/Dialect/Bufferization/IR/BufferizableOpInterface.td
 */

// TODO: Investigate how to get rid of identity-layout-map
//
// By default, one-shot-bufferization chooses dynamic memory layout.
// See https://mlir.llvm.org/docs/Bufferization/#memory-layouts
// In practice, this means the `getBuffer()` methods will return memrefs with dynamic
// memory layout, e.g. `memref<2x3xf64, strided<[?, ?], offset: ?>>`
// However, this causes some issues, namely:
// - Type mismatches during pattern rewriting
// - Ops taking in dynamic layouts need to supply operands for them
//
// For now, we force identity layout on all generated memrefs.
// This is be done by
// - For the `getBuffer()` methods: setting `unknown-type-conversion=identity-layout-map` on the
// `one-shot-bufferize` pass
// - For the memrefs we generate ourselves: creating a new MemRefType object by
// `MemRefType::get(shape, elementType)`, without supplying detailed layouts.
//
//
// The goal of the TODO is to eliminate the need to specify
// `unknown-type-conversion=identity-layout-map` on the pass.
// An easy strategy is just to insert memref.cast ops everywhere such mismatches happen.
// See https://mlir.llvm.org/docs/Dialects/MemRef/#memrefcast-memrefcastop:
//   * The source and destination types are compatible if:
//   *   - ...
//   *   - The individual sizes (resp. offset and strides in the case of strided memrefs) may
//   *     convert constant dimensions to dynamic dimensions and vice-versa.

namespace {

// A helper to generate a memref.alloc() with an identical type as the
// (possibly dynamically-shaped) reference Value.
Value generateAllocation(OpBuilder &builder, Location loc, Value reference)
{
    auto origMemrefType = cast<MemRefType>(reference.getType());
    auto memrefType = MemRefType::get(origMemrefType.getShape(), origMemrefType.getElementType());

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

// Helper function to generate a list of memref allocations.
void generateAllocations(RewriterBase &rewriter, Location loc, SmallVectorImpl<Value> &allocations,
                         ValueRange referenceValues)
{
    for (Value memref : referenceValues) {
        allocations.push_back(
            generateAllocation(rewriter, loc, cast<TypedValue<MemRefType>>(memref)));
    }
}

// A helper to collect a list of tensor types into corresponding memref types.
//
// This function essentially is the BufferizeTypeConverter. It just converts types without
// doing any real heavy-duty.
// However, the converter was removed upstream.
// See https://github.com/llvm/llvm-project/pull/114155/files
//
// Note that as stated in the overall TODO, we force identity layout at the moment.
void TensorType2MemrefType(const TypeRange &inTypes, SmallVector<Type> &convertedResults)
{
    for (Type inType : inTypes) {
        if (isa<TensorType>(inType)) {
            convertedResults.push_back(
                bufferization::getMemRefTypeWithStaticIdentityLayout(cast<TensorType>(inType)));
        }
        else {
            convertedResults.push_back(inType);
        }
    }
}

static BaseMemRefType
getBufferizedFunctionArgType(FunctionOpInterface funcOp, int64_t index,
                             const bufferization::BufferizationOptions &options)
{
    auto tensorType = dyn_cast<TensorType>(funcOp.getArgumentTypes()[index]);
    assert(tensorType && "expected TensorType");

    BaseMemRefType memrefType = options.functionArgTypeConverterFn(
        tensorType, *options.defaultMemorySpaceFn(tensorType), nullptr, options);

    return memrefType;
}

static ReturnOp getAssumedUniqueReturnOp(FunctionOpInterface funcOp)
{
    ReturnOp returnOp;
    for (Block &b : funcOp.getFunctionBody()) {
        if (auto candidateOp = dyn_cast<ReturnOp>(b.getTerminator())) {
            if (returnOp) {
                return nullptr;
            }
            returnOp = candidateOp;
        }
    }
    return returnOp;
}

// Bufferization of gradient.adjoint.
// Argument tensor is converted to memrefs by bufferization.to_memref.
// Result tensor of gradient.adjoint is bufferized with a corresponding memref.alloc.
// Users of the result tensor are updated to use the new memref.
struct AdjointOpInterface
    : public bufferization::BufferizableOpInterface::ExternalModel<AdjointOpInterface, AdjointOp> {
    bool bufferizesToMemoryRead(Operation *op, OpOperand &opOperand,
                                const bufferization::AnalysisState &state) const
    {
        return true;
    }

    bool bufferizesToMemoryWrite(Operation *op, OpOperand &opOperand,
                                 const bufferization::AnalysisState &state) const
    {
        return false;
    }

    bufferization::AliasingValueList
    getAliasingValues(Operation *op, OpOperand &opOperand,
                      const bufferization::AnalysisState &state) const
    {
        return {};
    }

    LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                            const bufferization::BufferizationOptions &options) const
    {
        auto adjointOp = cast<AdjointOp>(op);
        Location loc = adjointOp.getLoc();
        Value gradSize = adjointOp.getGradSize();

        SmallVector<Type> resTypes;
        TensorType2MemrefType(adjointOp.getResultTypes(), resTypes);
        assert(adjointOp->getNumResults() == resTypes.size() &&
               "Number of memrefs do not match number of tensor results!");

        SmallVector<Value> memrefValues;
        SmallVector<Type> nonTensorResultTypes;
        std::vector<size_t> nonTensorResultIndices;
        for (const auto &[i, resType] : llvm::enumerate(resTypes)) {
            if (isa<MemRefType>(resType)) {
                MemRefType memrefType = cast<MemRefType>(resType);
                Value memrefValue = rewriter.create<memref::AllocOp>(loc, memrefType, gradSize);
                memrefValues.push_back(memrefValue);
            }
            else {
                nonTensorResultTypes.push_back(adjointOp->getResultTypes()[i]);
                nonTensorResultIndices.push_back(i);
            }
        }

        SmallVector<Value> bufferArgs;
        ValueRange operands = adjointOp.getArgs();
        for (Value operand : operands) {
            if (isa<TensorType>(operand.getType())) {
                FailureOr<Value> opBuffer = getBuffer(rewriter, operand, options);
                if (failed(opBuffer)) {
                    return failure();
                }
                bufferArgs.push_back(*opBuffer);
            }
            else {
                bufferArgs.push_back(operand);
            }
        }

        auto newAdjointOp =
            rewriter.create<AdjointOp>(loc, nonTensorResultTypes, adjointOp.getCalleeAttr(),
                                       adjointOp.getGradSize(), bufferArgs, memrefValues);
        SmallVector<Value> bufferdNewValues;
        size_t nonTensorResultCounter = 0;
        size_t tensorResultCounter = 0;
        for (size_t i = 0; i < adjointOp->getNumResults(); i++) {
            if (std::find(nonTensorResultIndices.begin(), nonTensorResultIndices.end(), i) !=
                nonTensorResultIndices.end()) {
                // a non tensor result, just use the Value
                bufferdNewValues.push_back(newAdjointOp->getResult(nonTensorResultCounter));
                nonTensorResultCounter++;
            }
            else {
                // a tensor result, use the buffer
                bufferdNewValues.push_back(memrefValues[tensorResultCounter]);
                tensorResultCounter++;
            }
        }
        bufferization::replaceOpWithBufferizedValues(rewriter, op, bufferdNewValues);
        return success();
    }
};

// Bufferization of gradient.backprop.
// Argument tensor is converted to memrefs by bufferization.to_memref.
// Result tensor of gradient.backprop is bufferized with a corresponding memref.alloc.
// Users of the result tensor are updated to use the new memref.
// Cotangent operands are copied.
struct BackpropOpInterface
    : public bufferization::BufferizableOpInterface::ExternalModel<BackpropOpInterface,
                                                                   BackpropOp> {
    bool bufferizesToMemoryRead(Operation *op, OpOperand &opOperand,
                                const bufferization::AnalysisState &state) const
    {
        return true;
    }

    bool bufferizesToMemoryWrite(Operation *op, OpOperand &opOperand,
                                 const bufferization::AnalysisState &state) const
    {
        // Enzyme mutates the result shadows. This means the cotangents will be written into.
        // The other visible operand before bufferization is $args, the arguments to the
        // differentiated callee. It will not be written into.
        ValueRange cotangents = cast<BackpropOp>(op).getCotangents();
        bool operandIsCotangent =
            std::find(cotangents.begin(), cotangents.end(), opOperand.get()) != cotangents.end();
        return operandIsCotangent;
    }

    bufferization::AliasingValueList
    getAliasingValues(Operation *op, OpOperand &opOperand,
                      const bufferization::AnalysisState &state) const
    {
        return {};
    }

    LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                            const bufferization::BufferizationOptions &options) const
    {
        auto backpropOp = cast<BackpropOp>(op);
        Location loc = backpropOp.getLoc();

        // Conceptually a map from scalar result indices (w.r.t. other scalars) to the position in
        // the overall list of returned gradients.
        // For instance, a backprop op that returns (tensor, f64, tensor, f64, f64) will have
        // scalarIndices = {1, 3, 4}.
        SmallVector<unsigned> scalarIndices;
        SmallVector<Type> scalarReturnTypes;

        // 1. Convert callee's tensor arguments into memrefs
        SmallVector<Value> bufferArgs;
        ValueRange operands = backpropOp.getArgs();
        for (Value operand : operands) {
            if (isa<TensorType>(operand.getType())) {
                FailureOr<Value> opBuffer = getBuffer(rewriter, operand, options);
                if (failed(opBuffer)) {
                    return failure();
                }
                bufferArgs.push_back(*opBuffer);
            }
            else {
                bufferArgs.push_back(operand);
            }
        }

        // 2. Allocate buffers to place the differentiation results (gradients) into.
        // Enzyme refers to these as shadow arguments. There is one result for each
        // differentiable MemRef argument, with a matching shape and type.
        SmallVector<Value> gradients, argShadows;
        std::vector<Value> diffArgs =
            computeDiffArgs(bufferArgs, backpropOp.getDiffArgIndicesAttr());
        for (const auto &[idx, diffArg] : llvm::enumerate(diffArgs)) {
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

        // 3. Convert cotangent operands into memrefs.
        // Enzyme requires buffers for the primal outputs as well, even though we don't need their
        // values. We'll mark them dupNoNeed later on to allow Enzyme to optimize away their
        // computation.
        // Note that cotangents cannot be scalars.
        ValueRange cotangents = backpropOp.getCotangents();
        SmallVector<Value> bufferCotangents;
        for (Value operand : cotangents) {
            FailureOr<Value> opBuffer = getBuffer(rewriter, operand, options);
            if (failed(opBuffer)) {
                return failure();
            }
            bufferCotangents.push_back(*opBuffer);
        }

        SmallVector<Value> calleeResults;
        generateAllocations(rewriter, loc, calleeResults, bufferCotangents);

        // 4. Create bufferized backprop op
        DenseIntElementsAttr diffArgIndicesAttr = backpropOp.getDiffArgIndices().value_or(nullptr);
        auto bufferizedBackpropOp = rewriter.create<BackpropOp>(
            loc, TypeRange{}, scalarReturnTypes, backpropOp.getCalleeAttr(), bufferArgs, argShadows,
            calleeResults, bufferCotangents, diffArgIndicesAttr,
            backpropOp.getKeepValueResultsAttr());
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
            if (!backpropOp.getVals().empty()) {
                results.insert(results.end(), calleeResults.begin(), calleeResults.end());
            }
            results.insert(results.end(), gradients.begin(), gradients.end());
        }

        bufferization::replaceOpWithBufferizedValues(rewriter, op, results);
        return success();
    }
};

// Bufferization of gradient.forward.
struct ForwardOpInterface
    : public bufferization::OpWithUnstructuredControlFlowBufferizableOpInterfaceExternalModel<
          ForwardOpInterface, ForwardOp> {
    bool hasTensorSemantics(Operation *op) const
    {
        auto isaTensor = llvm::IsaPred<TensorType>;

        // A function has tensor semantics if it has tensor arguments/results.
        auto forwardOp = cast<ForwardOp>(op);
        bool hasTensorArg = any_of(forwardOp.getArgumentTypes(), isaTensor);
        bool hasTensorResult = any_of(forwardOp.getResultTypes(), isaTensor);
        bool hasTensorFuncInType = any_of(forwardOp.getFunctionType().getInputs(), isaTensor);
        bool hasTensorFuncOutType = any_of(forwardOp.getFunctionType().getResults(), isaTensor);
        if (hasTensorArg || hasTensorResult || hasTensorFuncInType || hasTensorFuncOutType) {
            return true;
        }

        return false;
    }

    bufferization::AliasingOpOperandList
    getAliasingOpOperands(Operation *op, Value value,
                          const bufferization::AnalysisState &state) const
    {
        return {};
    }

    FailureOr<BaseMemRefType> getBufferType(Operation *op, Value value,
                                            const bufferization::BufferizationOptions &options,
                                            SmallVector<Value> &invocationStack) const
    {
        // The getBufferType() method is called on either BlockArguments or OpResults.
        // https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir/Dialect/Bufferization/IR/BufferizableOpInterface.td#L506
        // Since forward and reverse ops are funcop-like, they do not have result Values,
        // so this method will only be called on BlockArguments.
        //
        // Among the block arguments, function arguments are special.
        // One-shot bufferize has two options to control type conversions from tensor to memref:
        // 1. The `unknown-type-conversion` controls the conversion of a generic Value.
        //   * Without implementing this `getBufferType()` method, all BlockArgument conversions
        //   * go through this default path.
        //   * https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir/Dialect/Bufferization/IR/BufferizableOpInterface.td#L533
        //   * https://github.com/llvm/llvm-project/blob/main/mlir/lib/Dialect/Bufferization/IR/BufferizableOpInterface.cpp#L952
        // 2. The `function-boundary-type-conversion` controls the conversion of function arguments.
        //
        // Since they go through separate bufferization API, so we need to separete them here too.
        auto forwardOp = cast<ForwardOp>(op);
        auto bbArg = cast<BlockArgument>(value);

        if (bbArg.getOwner() == &forwardOp.getBody().front()) {
            return getBufferizedFunctionArgType(forwardOp, bbArg.getArgNumber(), options);
        }

        return bufferization::detail::defaultGetBufferType(value, options, invocationStack);
    }

    LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                            const bufferization::BufferizationOptions &options) const
    {
        auto forwardOp = cast<ForwardOp>(op);
        FunctionType funcType = forwardOp.getFunctionType();

        // Construct the bufferized function type.
        SmallVector<Type> argTypes;
        for (const auto &it : llvm::enumerate(funcType.getInputs())) {
            Type argType = it.value();
            if (dyn_cast<TensorType>(argType)) {
                argTypes.push_back(getBufferizedFunctionArgType(forwardOp, it.index(), options));
                continue;
            }
            argTypes.push_back(argType);
        }

        ReturnOp returnOp = getAssumedUniqueReturnOp(forwardOp);
        Location loc = returnOp.getLoc();

        // 1. Bufferize every block.
        for (Block &block : forwardOp.getBody()) {
            if (failed(bufferization::bufferizeBlockSignature(&block, rewriter, options))) {
                return failure();
            }
        }

        // 2. For each result, keep track of which inplace argument it reuses.
        SmallVector<Value> returnValues;
        for (OpOperand &returnOperand : returnOp->getOpOperands()) {
            Value returnVal = returnOperand.get();
            auto tensorType = dyn_cast<TensorType>(returnVal.getType());
            rewriter.setInsertionPoint(returnOp);

            // If not a tensor type just forward it.
            if (!tensorType) {
                returnValues.push_back(returnVal);
                continue;
            }

            // Note: If `inferFunctionResultLayout = true`, cast are later folded
            // away.
            BaseMemRefType resultType = options.unknownTypeConverterFn(
                returnVal, *options.defaultMemorySpaceFn(tensorType), options);
            Value toMemrefOp =
                rewriter.create<bufferization::ToMemrefOp>(loc, resultType, returnVal);
            returnValues.push_back(toMemrefOp);
        }

        // 3. Rewrite the terminator.
        forwardOp.walk([&](ReturnOp returnOp) {
            PatternRewriter::InsertionGuard guard(rewriter);
            rewriter.setInsertionPoint(returnOp);
            rewriter.replaceOpWithNewOp<ReturnOp>(returnOp, returnValues, returnOp.getEmpty());
        });

        // 4. Rewrite the FuncOp type to buffer form. Also preserve unused return types.
        SmallVector<Type> returnTypes;
        TensorType2MemrefType(forwardOp.getResultTypes(), returnTypes);
        forwardOp.setType(FunctionType::get(op->getContext(), argTypes, returnTypes));

        return success();
    }
};

// Bufferization of gradient.reverse.
struct ReverseOpInterface
    : public bufferization::OpWithUnstructuredControlFlowBufferizableOpInterfaceExternalModel<
          ReverseOpInterface, ReverseOp> {
    bool hasTensorSemantics(Operation *op) const
    {
        auto isaTensor = llvm::IsaPred<TensorType>;

        // A function has tensor semantics if it has tensor arguments/results.
        auto reverseOp = cast<ReverseOp>(op);
        bool hasTensorArg = any_of(reverseOp.getArgumentTypes(), isaTensor);
        bool hasTensorResult = any_of(reverseOp.getResultTypes(), isaTensor);
        bool hasTensorFuncInType = any_of(reverseOp.getFunctionType().getInputs(), isaTensor);
        bool hasTensorFuncOutType = any_of(reverseOp.getFunctionType().getResults(), isaTensor);
        if (hasTensorArg || hasTensorResult || hasTensorFuncInType || hasTensorFuncOutType)
            return true;

        return false;
    }

    bufferization::AliasingOpOperandList
    getAliasingOpOperands(Operation *op, Value value,
                          const bufferization::AnalysisState &state) const
    {
        return {};
    }

    FailureOr<BaseMemRefType> getBufferType(Operation *op, Value value,
                                            const bufferization::BufferizationOptions &options,
                                            SmallVector<Value> &invocationStack) const
    {
        // See comment on the getBufferType() method on forward op.
        auto reverseOp = cast<ReverseOp>(op);
        auto bbArg = cast<BlockArgument>(value);

        // Function arguments are special.
        if (bbArg.getOwner() == &reverseOp.getBody().front()) {
            return getBufferizedFunctionArgType(reverseOp, bbArg.getArgNumber(), options);
        }

        return bufferization::detail::defaultGetBufferType(value, options, invocationStack);
    }

    LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                            const bufferization::BufferizationOptions &options) const
    {
        auto reverseOp = cast<ReverseOp>(op);
        FunctionType funcType = reverseOp.getFunctionType();

        // Construct the bufferized function type.
        SmallVector<Type> argTypes;
        for (const auto &it : llvm::enumerate(funcType.getInputs())) {
            Type argType = it.value();
            if (dyn_cast<TensorType>(argType)) {
                argTypes.push_back(getBufferizedFunctionArgType(reverseOp, it.index(), options));
                continue;
            }
            argTypes.push_back(argType);
        }

        ReturnOp returnOp = getAssumedUniqueReturnOp(reverseOp);
        Location loc = returnOp.getLoc();

        // 1. Bufferize every block.
        for (Block &block : reverseOp.getBody())
            if (failed(bufferization::bufferizeBlockSignature(&block, rewriter, options)))
                return failure();

        // 2. For each result, keep track of which inplace argument it reuses.
        SmallVector<Value> returnValues;
        for (OpOperand &returnOperand : returnOp->getOpOperands()) {
            Value returnVal = returnOperand.get();
            auto tensorType = dyn_cast<TensorType>(returnVal.getType());
            rewriter.setInsertionPoint(returnOp);

            // If not a tensor type just forward it.
            if (!tensorType) {
                returnValues.push_back(returnVal);
                continue;
            }

            // Note: If `inferFunctionResultLayout = true`, cast are later folded
            // away.
            BaseMemRefType resultType = options.unknownTypeConverterFn(
                returnVal, *options.defaultMemorySpaceFn(tensorType), options);
            Value toMemrefOp =
                rewriter.create<bufferization::ToMemrefOp>(loc, resultType, returnVal);
            returnValues.push_back(toMemrefOp);
        }

        // 3. Rewrite the terminator.
        reverseOp.walk([&](ReturnOp returnOp) {
            PatternRewriter::InsertionGuard guard(rewriter);
            rewriter.setInsertionPoint(returnOp);
            rewriter.replaceOpWithNewOp<ReturnOp>(returnOp, returnValues, returnOp.getEmpty());
        });

        // 4. Rewrite the FuncOp type to buffer form. Also preserve unused return types.
        SmallVector<Type> returnTypes;
        TensorType2MemrefType(reverseOp.getResultTypes(), returnTypes);
        reverseOp.setType(FunctionType::get(op->getContext(), argTypes, returnTypes));

        return success();
    }
};

} // namespace

void catalyst::gradient::registerBufferizableOpInterfaceExternalModels(DialectRegistry &registry)
{
    registry.addExtension(+[](MLIRContext *ctx, GradientDialect *dialect) {
        AdjointOp::attachInterface<AdjointOpInterface>(*ctx);
        BackpropOp::attachInterface<BackpropOpInterface>(*ctx);
        ForwardOp::attachInterface<ForwardOpInterface>(*ctx);
        ReverseOp::attachInterface<ReverseOpInterface>(*ctx);
    });
}
