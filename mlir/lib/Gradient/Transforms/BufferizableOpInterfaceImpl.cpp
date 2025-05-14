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

#include "iostream"
#include "llvm/Support/raw_ostream.h"

#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
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
#include "Gradient/Transforms/BufferizableOpInterfaceImpl.h"
#include "Gradient/Transforms/Passes.h"
#include "Gradient/Utils/GradientShape.h"

using namespace mlir;
using namespace mlir::bufferization;
using namespace catalyst::gradient;

/**
 * Implementation of the BufferizableOpInterface for use with one-shot bufferization.
 * For more information on the interface, refer to the documentation below:
 *  https://mlir.llvm.org/docs/Bufferization/#extending-one-shot-bufferize
 *  https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir/Dialect/Bufferization/IR/BufferizableOpInterface.td#L14
 */

namespace {

Value generateAllocation(OpBuilder &builder, Location loc, Value reference)
{
    auto origMemrefType = cast<MemRefType>(reference.getType());
    // TODO: Investigate how to get rid of identity-layout-map
    //
    //     Hi all. For one-shot-bufferization, is there any automatic way to pass all memref symbols
    //     to AllocOp? we have an example below that triggers  error: 'memref.alloc' op symbol
    //     operand count does not equal memref symbol count: expected 1, got 0 .  We think we have
    //     to pass the offset symbol to AllocOp.
    //
    //         %0 = "bufferization.to_memref"(%arg0) : (tensor<f64>) -> memref<f64, strided<[],
    //         offset: ?>> %1 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> : () ->
    //         memref<f64, strided<[], offset: ?>>
    //
    //     We know we can set function-signature-type-conversion=identity-layout-map to get rid of
    //     it. But according to the document, identity-layout-map could be less efficient, we still
    //     want to stick with the default setting.
    //
    // https://discord.com/channels/636084430946959380/642426447167881246/1281620504859512914
    //
    //     Something looks odd here.
    //     The result of a `memref.alloc` should be a memref without identity layout.
    //     I know that the op supports operands for dims/symbols in the memref type,
    //     but I never understood why.
    //     Imo, a `memref.alloc() : memref<f64>` should have been generated.
    //     The result value can then be casted to `memref<f64, strided<[], offset: ?>>`.
    //
    // https://discord.com/channels/636084430946959380/642426447167881246/1281710682160627785
    //
    // What I find interesting is that the comment says that
    //
    //     "we know we can set function-signature-type-conversion=identity-layout-map to get rid of
    //     it"
    //
    // and that is what we are using, however we still have this rebuilding a memref without the
    // layout. If that were true, then we could uncomment the following line and it should work.
    // auto memrefType = origMemrefType;
    // I can confirm that having
    // function-signature-type-conversion=identity-layout-map makes the line above succed while the
    // line below fail:
    //
    //     Get dynamic dimension sizes from the provided reference value if necessary.
    auto memrefType = MemRefType::get(origMemrefType.getShape(), origMemrefType.getElementType());
    //
    // Looking at this a little bit deeper, I can say that the variable reference
    // appears to come from a function parameter.
    // and since it is not the identity layout, then we see the following generic MLIR when not
    // using identity layout
    //
    // "func.func"() <{function_type = (memref<f64, strided<[], offset: ?>>) -> memref<f64,
    // strided<[], offset: ?>>
    //
    // and we see this when using the identity layout:
    //
    // func.func public @jit_fn(%arg0: memref<f64>) -> memref<f64>
    //
    // When not using identity layout but also not removing the layout in the alloca, there are
    // errors in some cases but not in others. I believe we have to do some casts in other places as
    // well, whenever we use allocas and the types come from the arguments.
    //
    // My recommendation: at some point it would be good to remove the identity-layout-map from the
    // frontend but until we have some more resources, let's keep it along with the origMemrefType.

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
    // Uncomment below to follow Matthias suggestion of placing a CastOp after AllocOp
    // some more tests will pass.
    // return builder.create<memref::CastOp>(loc, origMemrefType, alloc_uncasted);
}

/// Helper function to generate a set of memref allocations.
///
/// The allocation size and shape is deduced from a list of existing memref values.
///
void generateAllocations(RewriterBase &rewriter, Location loc, SmallVectorImpl<Value> &allocations,
                         ValueRange referenceValues)
{
    for (Value memref : referenceValues) {
        allocations.push_back(
            generateAllocation(rewriter, loc, cast<TypedValue<MemRefType>>(memref)));
    }
}

void TensorType2MemrefType(const SmallVector<Type> &inTypes, SmallVector<Type> &convertedResults)
{
    // A helper to collect the result tensor values into corresponding memref types.
    // We force identity layout on the memref.
    //
    // This function essentially is the BufferizeTypeConverter
    // However, the converter was removed upstream.
    // See https://github.com/llvm/llvm-project/pull/114155/files
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
        TensorType2MemrefType(SmallVector<Type>(adjointOp.getResultTypes()), resTypes);
        assert(adjointOp->getNumResults() == resTypes.size() &&
               "Number of memrefs do not match number of tensor results!");

        SmallVector<Value> memrefValues;
        for (Type resType : resTypes) {
            MemRefType memrefType = cast<MemRefType>(resType);
            Value memrefValue = rewriter.create<memref::AllocOp>(loc, memrefType, gradSize);
            memrefValues.push_back(memrefValue);
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

        rewriter.create<AdjointOp>(loc, TypeRange{}, adjointOp.getCalleeAttr(),
                                   adjointOp.getGradSize(), bufferArgs, memrefValues);
        bufferization::replaceOpWithBufferizedValues(rewriter, op, memrefValues);
        return success();
    }
};

// Bufferization of gradient.backprop.
// Argument tensor is converted to memrefs by bufferization.to_memref.
// Result tensor of gradient.backprop is bufferized with a corresponding memref.alloc.
// Users of the result tensor are updated to use the new memref.
// cotangents?
//
// Note that backprop is the only one that supports value_and_grad, so result tensors might
// include both the value tensor and the grad tensor.
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
        // I think we don't write to the cotangents. And also not to the arguments
        // so we can set bufferizesToMemoryWrite as false.
        // The safe assumption is that it should be true.
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
        auto backpropOp = cast<BackpropOp>(op);

        Location loc = backpropOp.getLoc();
        SmallVector<Value> gradients;
        SmallVector<Value> argShadows;
        // Conceptually a map from scalar result indices (w.r.t. other scalars) to the position in
        // the overall list of returned gradients.
        // For instance, a backprop op that returns (tensor, f64, tensor, f64, f64) will have
        // scalarIndices = {1, 3, 4}.
        SmallVector<unsigned> scalarIndices;
        SmallVector<Type> scalarReturnTypes;

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

        std::vector<Value> diffArgs =
            computeDiffArgs(bufferArgs, backpropOp.getDiffArgIndicesAttr());

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
        ValueRange cotangents = backpropOp.getCotangents();
        SmallVector<Value> bufferCotangentsList;
        for (Value operand : cotangents) {
            FailureOr<Value> opBuffer = getBuffer(rewriter, operand, options);
            if (failed(opBuffer)) {
                return failure();
            }
            bufferCotangentsList.push_back(*opBuffer);
        }
        mlir::ValueRange bufferCotangents(bufferCotangentsList);

        generateAllocations(rewriter, loc, calleeResults, bufferCotangents);
        // Enzyme mutates the result shadows but the cotangent tensors must be immutable, so we
        // create copies to pass into Enzyme. Concretely, this issue pops up with multiple
        // BackpropOps that have the same cotangent tensor due to a CSE effect from one-shot
        // bufferization.
        generateAllocations(rewriter, loc, resShadows, bufferCotangents);
        for (const auto &[cotangent, resShadow] : llvm::zip(bufferCotangents, resShadows)) {
            rewriter.create<memref::CopyOp>(loc, cotangent, resShadow);
        }

        DenseIntElementsAttr diffArgIndicesAttr = backpropOp.getDiffArgIndices().value_or(nullptr);
        auto bufferizedBackpropOp = rewriter.create<BackpropOp>(
            loc, TypeRange{}, scalarReturnTypes, backpropOp.getCalleeAttr(), bufferArgs, argShadows,
            calleeResults, resShadows, diffArgIndicesAttr, backpropOp.getKeepValueResultsAttr());
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

} // namespace

void catalyst::gradient::registerBufferizableOpInterfaceExternalModels(DialectRegistry &registry)
{
    registry.addExtension(+[](MLIRContext *ctx, GradientDialect *dialect) {
        AdjointOp::attachInterface<AdjointOpInterface>(*ctx);
        BackpropOp::attachInterface<BackpropOpInterface>(*ctx);
        // ForwardOp::attachInterface<ForwardOpInterface>(*ctx);
        // ReverseOp::attachInterface<ReverseOpInterface>(*ctx);
    });
}
