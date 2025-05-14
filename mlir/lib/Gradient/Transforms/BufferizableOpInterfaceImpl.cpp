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
void TensorType2MemrefType(const SmallVector<Type> &inTypes, SmallVector<Type> &convertedResults)
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
        // Enzyme mutates the result shadows, but the cotangent tensors must be immutable for SSA.
        // Thus we create copies to be the result shadows passed into Enzyme.
        // For example, the same cotangent tensor SSA value can be used by multiple backprop ops,
        // due to things like CSE. Therefore the memref corresponding to the original tensor must
        // be left untouched.
        //
        // All other operands will not be written into.

        ValueRange cotangents = cast<BackpropOp>(op).getCotangents();
        return std::find(cotangents.begin(), cotangents.end(), opOperand.get()) != cotangents.end();
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
