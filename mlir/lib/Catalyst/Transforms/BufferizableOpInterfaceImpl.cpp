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

#include "Catalyst/Transforms/BufferizableOpInterfaceImpl.h"

#include <cstdint>

#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Transforms/DialectConversion.h"

#include "Catalyst/IR/CatalystOps.h"

using namespace mlir;
using namespace catalyst;

/**
 * Implementation of the BufferizableOpInterface for use with one-shot bufferization.
 * For more information on the interface, refer to the documentation below:
 *  https://mlir.llvm.org/docs/Bufferization/#extending-one-shot-bufferize
 *  https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir/Dialect/Bufferization/IR/BufferizableOpInterface.td#L14
 */

namespace {

/// Bufferization of catalyst.print. Get memref of printOp.val.
struct PrintOpInterface
    : public bufferization::BufferizableOpInterface::ExternalModel<PrintOpInterface, PrintOp> {
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
                            const bufferization::BufferizationOptions &options,
                            bufferization::BufferizationState &state) const
    {
        auto printOp = cast<PrintOp>(op);
        if (printOp.getVal()) {
            FailureOr<Value> source = getBuffer(rewriter, printOp.getVal(), options, state);
            if (failed(source)) {
                return failure();
            }
            bufferization::replaceOpWithNewBufferizedOp<PrintOp>(
                rewriter, op, *source, printOp.getConstValAttr(), printOp.getPrintDescriptorAttr());
        }
        return success();
    }
};

/// Bufferization of catalyst.custom_call. Mainly get buffers for arguments.
struct CustomCallOpInterface
    : public bufferization::BufferizableOpInterface::ExternalModel<CustomCallOpInterface,
                                                                   CustomCallOp> {
    bool bufferizesToAllocation(Operation *op, Value value) const { return true; }

    bool bufferizesToMemoryRead(Operation *op, OpOperand &opOperand,
                                const bufferization::AnalysisState &state) const
    {
        // Custom Call Op always reads the operand memory no matter what.
        return true;
    }

    bool bufferizesToMemoryWrite(Operation *op, OpOperand &opOperand,
                                 const bufferization::AnalysisState &state) const
    {
        // We only use custom call for the jax lapack kernels.
        // This is actually hard-guarded: in the lowering pattern for custom call
        // we check that the name of the callee is a jax symbol for a lapack kernel.
        //
        // The lapack kernels themselves might overwrite some of the input arrays.
        // However, in jax's shim wrapper layer, a memcpy is already performed.
        // See
        // https://github.com/PennyLaneAI/catalyst/blob/main/frontend/catalyst/utils/jax_cpu_lapack_kernels/lapack_kernels.cpp
        //
        // The arguments to the underlying lapack kernel are denoted by the jax wrapper
        // function as `data`. The `data` args already contain the output array that
        // the lapack kernel is supposed to write into. The other input arrays are all marked const.
        // Jax then purifies the function by adding a new argument `out` to hold the
        // output array.
        //
        // In other words, the jax wrappers we call here with custom call op
        // are already pure, and we won't have side effects on the input tensors.

        return false;
    }

    bufferization::AliasingValueList
    getAliasingValues(Operation *op, OpOperand &opOperand,
                      const bufferization::AnalysisState &state) const
    {
        return {};
    }

    LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                            const bufferization::BufferizationOptions &options,
                            bufferization::BufferizationState &state) const
    {
        auto customCallOp = cast<CustomCallOp>(op);

        // Add bufferized arguments
        SmallVector<Value> bufferArgs;
        ValueRange operands = customCallOp.getOperands();
        for (Value operand : operands) {
            FailureOr<Value> opBuffer = getBuffer(rewriter, operand, options, state);
            if (failed(opBuffer)) {
                return failure();
            }

            // If any of the input buffers have non-identity layout, we need to make a copy
            // This is because the lapack kernel runtime demands contiguous arrays,
            // aka identity, non-strided, non-offset memory layout
            //
            // TODO: reuse this copy for the result buffer whenever the semantics of the kernel
            // allow this reuse.
            // The kernel runtime performs a copy from the input array to the output array
            // if they are separate memrefs. We can avoid the extra copy there by just passing in
            // the newly allocated memref here into the kernel as the result buffer too.
            // However, we need to check that the kernels' semantics allow this.
            MemRefType bufferedOperandMemrefType = cast<MemRefType>(opBuffer->getType());
            if (!bufferedOperandMemrefType.getLayout().isIdentity()) {
                MemRefType copiedOperandMemrefType =
                    MemRefType::get(bufferedOperandMemrefType.getShape(),
                                    bufferedOperandMemrefType.getElementType());
                auto allocOp =
                    memref::AllocOp::create(rewriter, op->getLoc(), copiedOperandMemrefType);
                auto copyOp =
                    memref::CopyOp::create(rewriter, op->getLoc(), *opBuffer, allocOp.getResult());
                bufferArgs.push_back(copyOp.getTarget());
            }
            else {
                bufferArgs.push_back(*opBuffer);
            }
        }

        // Add bufferized return values to the arguments
        ValueRange results = customCallOp.getResults();
        for (Value result : results) {
            Type resultType = result.getType();
            RankedTensorType tensorType = dyn_cast<RankedTensorType>(resultType);
            if (!tensorType) {
                return failure();
            }
            auto options = bufferization::BufferizationOptions();
            FailureOr<Value> tensorAlloc = bufferization::allocateTensorForShapedValue(
                rewriter, op->getLoc(), result, options, state, false);
            MemRefType memrefType =
                MemRefType::get(tensorType.getShape(), tensorType.getElementType());
            auto newBuffer =
                bufferization::ToBufferOp::create(rewriter, op->getLoc(), memrefType, *tensorAlloc);
            bufferArgs.push_back(newBuffer);
        }

        // Add the initial number of arguments
        int32_t numArguments = static_cast<int32_t>(customCallOp.getNumOperands());
        IntegerAttr numArgumentsAttr = rewriter.getI32IntegerAttr(numArguments);

        // Create an updated custom call operation
        CustomCallOp::create(rewriter, op->getLoc(), TypeRange{}, bufferArgs,
                             customCallOp.getCallTargetName(), numArgumentsAttr,
                             customCallOp.getBackendConfigAttr());
        size_t startIndex = bufferArgs.size() - customCallOp.getNumResults();
        SmallVector<Value> bufferResults(bufferArgs.begin() + startIndex, bufferArgs.end());
        bufferization::replaceOpWithBufferizedValues(rewriter, op, bufferResults);

        return success();
    }
};

struct CallbackOpInterface
    : public bufferization::BufferizableOpInterface::ExternalModel<CallbackOpInterface,
                                                                   CallbackOp> {
    bool hasTensorSemantics(Operation *op) const
    {
        auto isaTensor = llvm::IsaPred<TensorType>;

        // A function has tensor semantics if it has tensor arguments/results.
        auto callbackOp = cast<CallbackOp>(op);
        bool hasTensorArg = any_of(callbackOp.getArgumentTypes(), isaTensor);
        bool hasTensorResult = any_of(callbackOp.getResultTypes(), isaTensor);
        if (hasTensorArg || hasTensorResult) {
            return true;
        }

        return false;
    }

    LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                            const bufferization::BufferizationOptions &options,
                            bufferization::BufferizationState &state) const
    {
        auto callbackOp = cast<CallbackOp>(op);

        auto argTys = callbackOp.getArgumentTypes();
        auto retTys = callbackOp.getResultTypes();
        SmallVector<Type> emptyRets;
        SmallVector<Type> args(argTys.begin(), argTys.end());
        args.insert(args.end(), retTys.begin(), retTys.end());
        SmallVector<Type> bufferArgs;
        for (Type ty : args) {
            auto tensorType = dyn_cast<RankedTensorType>(ty);
            if (!tensorType) {
                bufferArgs.push_back(ty);
            }
            else {
                bufferArgs.push_back(
                    MemRefType::get(tensorType.getShape(), tensorType.getElementType()));
            }
        }
        auto callbackTy = rewriter.getFunctionType(bufferArgs, emptyRets);
        rewriter.modifyOpInPlace(op, [&] { callbackOp.setFunctionType(callbackTy); });

        return success();
    }
};

void convertTypes(SmallVector<Type> inTypes, SmallVector<Type> &convertedResults)
{
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

struct CallbackCallOpInterface
    : public bufferization::BufferizableOpInterface::ExternalModel<CallbackCallOpInterface,
                                                                   CallbackCallOp> {
    bool bufferizesToAllocation(Operation *op, Value value) const { return true; }

    bool bufferizesToMemoryRead(Operation *op, OpOperand &opOperand,
                                const bufferization::AnalysisState &state) const
    {
        return true;
    }

    bool bufferizesToMemoryWrite(Operation *op, OpOperand &opOperand,
                                 const bufferization::AnalysisState &state) const
    {
        // We can safely say false because CallbackCallOp's memrefs
        // will be put in a JAX array and JAX arrays are immutable.
        //
        //    Unlike NumPy arrays, JAX arrays are always immutable.
        //
        // https://jax.readthedocs.io/en/latest/notebooks/thinking_in_jax.html
        return false;
    }

    bufferization::AliasingValueList
    getAliasingValues(Operation *op, OpOperand &opOperand,
                      const bufferization::AnalysisState &state) const
    {
        return {};
    }

    LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                            const bufferization::BufferizationOptions &options,
                            bufferization::BufferizationState &state) const
    {
        auto callOp = cast<CallbackCallOp>(op);

        SmallVector<Type> convertedResults;
        convertTypes(SmallVector<Type>(callOp.getResultTypes()), convertedResults);
        if (callOp->getNumResults() != convertedResults.size()) {
            return failure();
        }

        SmallVector<Value> newInputs;
        auto operands = callOp.getOperands();
        for (Value operand : operands) {
            FailureOr<Value> opBuffer = getBuffer(rewriter, operand, options, state);
            if (failed(opBuffer)) {
                return failure();
            }
            newInputs.push_back(*opBuffer);
        }

        auto results = callOp.getResults();
        auto loc = callOp->getLoc();
        SmallVector<Value> outmemrefs;
        for (auto result : results) {
            FailureOr<Value> tensorAlloc = bufferization::allocateTensorForShapedValue(
                rewriter, loc, result, options, state, false);
            if (failed(tensorAlloc)) {
                return failure();
            }

            auto tensor = *tensorAlloc;
            RankedTensorType tensorTy = cast<RankedTensorType>(tensor.getType());
            auto shape = tensorTy.getShape();
            auto elementTy = tensorTy.getElementType();
            auto memrefType = MemRefType::get(shape, elementTy);
            auto toBufferOp = bufferization::ToBufferOp::create(rewriter, loc, memrefType, tensor);
            auto memref = toBufferOp.getResult();
            outmemrefs.push_back(memref);
            newInputs.push_back(memref);
        }

        SmallVector<Type> emptyRets;
        CallbackCallOp::create(rewriter, loc, emptyRets, callOp.getCallee(), newInputs,
                               /*arg_attrs=*/nullptr, /*res_attrs=*/nullptr);
        bufferization::replaceOpWithBufferizedValues(rewriter, op, outmemrefs);
        return success();
    }
};

/// Bufferization of catalyst.symbolic_array. This op is a placeholder with no
/// buffer semantics and is expected to be consumed before bufferization. If it
/// reaches this stage, emit an informative diagnostic instead of the generic
/// "op was not bufferized" error.
struct SymbolicArrayOpInterface
    : public bufferization::BufferizableOpInterface::ExternalModel<SymbolicArrayOpInterface,
                                                                   SymbolicArrayOp> {
    bool bufferizesToMemoryRead(Operation *op, OpOperand &opOperand,
                                const bufferization::AnalysisState &state) const
    {
        return false;
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
                            const bufferization::BufferizationOptions &options,
                            bufferization::BufferizationState &state) const
    {
        return op->emitError("catalyst::symbolic_array is a placeholder op for resource estimation "
                             "and cannot currently be bufferized or executed.");
    }
};

/// Bufferization of catalyst.launch_kernel. The callee reads its operands and returns each result
/// in its own freshly allocated buffer. Bufferization therefore converts the operands to buffers
/// and the tensor results to memref results (return-by-value): no operand is written in place and
/// no result aliases an operand.
struct LaunchKernelOpInterface
    : public bufferization::BufferizableOpInterface::ExternalModel<LaunchKernelOpInterface,
                                                                   LaunchKernelOp> {
    // Each result is returned in a buffer the op allocates.
    bool bufferizesToAllocation(Operation *op, Value value) const { return true; }

    // The callee reads its operands.
    bool bufferizesToMemoryRead(Operation *op, OpOperand &opOperand,
                                const bufferization::AnalysisState &state) const
    {
        return true;
    }

    // Operands are not written in place: every result is returned in a separate allocation.
    bool bufferizesToMemoryWrite(Operation *op, OpOperand &opOperand,
                                 const bufferization::AnalysisState &state) const
    {
        return false;
    }

    // Results are fresh allocations, so they alias none of the operands.
    bufferization::AliasingValueList
    getAliasingValues(Operation *op, OpOperand &opOperand,
                      const bufferization::AnalysisState &state) const
    {
        return {};
    }

    LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                            const bufferization::BufferizationOptions &options,
                            bufferization::BufferizationState &state) const
    {
        auto launchOp = cast<LaunchKernelOp>(op);

        SmallVector<Value> bufferOperands;
        for (Value operand : launchOp.getInputs()) {
            Value buffer;
            if (isa<TensorType>(operand.getType())) {
                FailureOr<Value> opBuffer = getBuffer(rewriter, operand, options, state);
                if (failed(opBuffer)) {
                    return failure();
                }
                buffer = *opBuffer;
            }
            else {
                buffer = operand;
            }

            // The callee receives each operand as a contiguous block, addressed through the
            // memref's aligned pointer with its strides/offset ignored. Copy any
            // non-identity-layout operand into a fresh contiguous buffer first so the callee sees
            // the intended elements.
            if (auto memrefTy = dyn_cast<MemRefType>(buffer.getType())) {
                if (!memrefTy.getLayout().isIdentity()) {
                    MemRefType contiguousTy =
                        MemRefType::get(memrefTy.getShape(), memrefTy.getElementType());
                    auto alloc = memref::AllocOp::create(rewriter, op->getLoc(), contiguousTy);
                    memref::CopyOp::create(rewriter, op->getLoc(), buffer, alloc.getResult());
                    buffer = alloc.getResult();
                }
            }
            bufferOperands.push_back(buffer);
        }

        SmallVector<Type> memrefResultTypes;
        convertTypes(SmallVector<Type>(launchOp.getResultTypes()), memrefResultTypes);

        auto newLaunchOp = LaunchKernelOp::create(
            rewriter, op->getLoc(), memrefResultTypes, launchOp.getCallee(), bufferOperands,
            launchOp.getArgAttrsAttr(), launchOp.getResAttrsAttr());
        // Carry over any discardable attributes, since create() only sets the declared ones.
        newLaunchOp->setDiscardableAttrs(launchOp->getDiscardableAttrDictionary());
        bufferization::replaceOpWithBufferizedValues(rewriter, op, newLaunchOp.getResults());
        return success();
    }
};

} // namespace

void catalyst::registerBufferizableOpInterfaceExternalModels(DialectRegistry &registry)
{
    registry.addExtension(+[](MLIRContext *ctx, CatalystDialect *dialect) {
        CustomCallOp::attachInterface<CustomCallOpInterface>(*ctx);
        PrintOp::attachInterface<PrintOpInterface>(*ctx);
        CallbackOp::attachInterface<CallbackOpInterface>(*ctx);
        CallbackCallOp::attachInterface<CallbackCallOpInterface>(*ctx);
        SymbolicArrayOp::attachInterface<SymbolicArrayOpInterface>(*ctx);
        LaunchKernelOp::attachInterface<LaunchKernelOpInterface>(*ctx);
    });
}
