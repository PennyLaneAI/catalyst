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

#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Transforms/DialectConversion.h"

#include "Catalyst/IR/CatalystOps.h"
#include "Catalyst/Transforms/BufferizableOpInterfaceImpl.h"

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
                            const bufferization::BufferizationOptions &options) const
    {
        auto printOp = cast<PrintOp>(op);
        if (printOp.getVal()) {
            FailureOr<Value> source = getBuffer(rewriter, printOp.getVal(), options);
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
                            const bufferization::BufferizationOptions &options) const
    {
        auto customCallOp = cast<CustomCallOp>(op);

        // Add bufferized arguments
        SmallVector<Value> bufferArgs;
        ValueRange operands = customCallOp.getOperands();
        for (Value operand : operands) {
            FailureOr<Value> opBuffer = getBuffer(rewriter, operand, options);
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
                    rewriter.create<memref::AllocOp>(op->getLoc(), copiedOperandMemrefType);
                auto copyOp =
                    rewriter.create<memref::CopyOp>(op->getLoc(), *opBuffer, allocOp.getResult());
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
                rewriter, op->getLoc(), result, options, false);
            MemRefType memrefType =
                MemRefType::get(tensorType.getShape(), tensorType.getElementType());
            auto newBuffer =
                rewriter.create<bufferization::ToMemrefOp>(op->getLoc(), memrefType, *tensorAlloc);
            bufferArgs.push_back(newBuffer);
        }

        // Add the initial number of arguments
        int32_t numArguments = static_cast<int32_t>(customCallOp.getNumOperands());
        DenseI32ArrayAttr numArgumentsDenseAttr = rewriter.getDenseI32ArrayAttr({numArguments});

        // Create an updated custom call operation
        rewriter.create<CustomCallOp>(op->getLoc(), TypeRange{}, bufferArgs,
                                      customCallOp.getCallTargetName(), numArgumentsDenseAttr);
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
                            const bufferization::BufferizationOptions &options) const
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
                            const bufferization::BufferizationOptions &options) const
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
            FailureOr<Value> opBuffer = getBuffer(rewriter, operand, options);
            if (failed(opBuffer)) {
                return failure();
            }
            newInputs.push_back(*opBuffer);
        }

        auto results = callOp.getResults();
        auto loc = callOp->getLoc();
        SmallVector<Value> outmemrefs;
        for (auto result : results) {
            FailureOr<Value> tensorAlloc =
                bufferization::allocateTensorForShapedValue(rewriter, loc, result, options, false);
            if (failed(tensorAlloc)) {
                return failure();
            }

            auto tensor = *tensorAlloc;
            RankedTensorType tensorTy = cast<RankedTensorType>(tensor.getType());
            auto shape = tensorTy.getShape();
            auto elementTy = tensorTy.getElementType();
            auto memrefType = MemRefType::get(shape, elementTy);
            auto toMemrefOp = rewriter.create<bufferization::ToMemrefOp>(loc, memrefType, tensor);
            auto memref = toMemrefOp.getResult();
            outmemrefs.push_back(memref);
            newInputs.push_back(memref);
        }

        SmallVector<Type> emptyRets;
        rewriter.create<CallbackCallOp>(loc, emptyRets, callOp.getCallee(), newInputs,
                                        /*arg_attrs=*/nullptr, /*res_attrs=*/nullptr);
        bufferization::replaceOpWithBufferizedValues(rewriter, op, outmemrefs);
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
    });
}
