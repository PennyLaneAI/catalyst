#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/IR/UnstructuredControlFlow.h"
#include "mlir/Dialect/Bufferization/Transforms/Bufferize.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"

#include "Catalyst/IR/CatalystOps.h"
#include "Catalyst/Transforms/BufferizableOpInterfaceImpl.h"

using namespace mlir;
using namespace catalyst;

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
            if (failed(source))
                return failure();
            bufferization::replaceOpWithNewBufferizedOp<PrintOp>(
                rewriter, op, *source, printOp.getConstValAttr(), printOp.getPrintDescriptorAttr());
        }
        return success();
    }
};

/// Bufferization of catalyst.print. Mainly get buffers for arguments.
struct CustomCallOpInterface
    : public bufferization::BufferizableOpInterface::ExternalModel<CustomCallOpInterface,
                                                                   CustomCallOp> {
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
        auto customCallOp = cast<CustomCallOp>(op);

        // Add bufferized arguments
        SmallVector<Value> bufferArgs;
        ValueRange operands = customCallOp.getOperands();
        for (Value operand : operands) {
            FailureOr<Value> opBuffer = getBuffer(rewriter, operand, options);
            if (failed(opBuffer))
                return failure();
            MemRefType memrefType = dyn_cast<MemRefType>(opBuffer->getType());
            if (!memrefType)
                return failure();
            if (!memrefType.getLayout().isIdentity()) {
                auto nonStrideType =
                    MemRefType::get(memrefType.getShape(), memrefType.getElementType());
                auto newMemRef =
                    rewriter.create<memref::CastOp>(op->getLoc(), nonStrideType, *opBuffer);
                bufferArgs.push_back(newMemRef);
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
    : public bufferization::OpWithUnstructuredControlFlowBufferizableOpInterfaceExternalModel<
          CallbackOpInterface, CallbackOp> {
    static bool supportsUnstructuredControlFlow() { return true; }

    bool hasTensorSemantics(Operation *op) const
    {
        auto isaTensor = llvm::IsaPred<TensorType>;

        // A function has tensor semantics if it has tensor arguments/results.
        auto callbackOp = cast<CallbackOp>(op);
        bool hasTensorArg = any_of(callbackOp.getArgumentTypes(), isaTensor);
        bool hasTensorResult = any_of(callbackOp.getResultTypes(), isaTensor);
        if (hasTensorArg || hasTensorResult)
            return true;

        return false;
    }

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
        auto callbackOp = cast<CallbackOp>(op);

        auto argTys = callbackOp.getArgumentTypes();
        auto retTys = callbackOp.getResultTypes();
        SmallVector<Type> emptyRets;
        SmallVector<Type> args(argTys.begin(), argTys.end());
        args.insert(args.end(), retTys.begin(), retTys.end());
        SmallVector<Type> bufferArgs;
        for (Type ty : args) {
            auto tensorType = dyn_cast<RankedTensorType>(ty);
            if (!tensorType)
                bufferArgs.push_back(ty);
            else
                bufferArgs.push_back(
                    MemRefType::get(tensorType.getShape(), tensorType.getElementType()));
        }
        auto callbackTy = rewriter.getFunctionType(bufferArgs, emptyRets);
        rewriter.modifyOpInPlace(op, [&] { callbackOp.setFunctionType(callbackTy); });

        return success();
    }
};

struct CallbackCallOpInterface
    : public bufferization::BufferizableOpInterface::ExternalModel<CallbackCallOpInterface,
                                                                   CallbackCallOp> {
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
        auto callOp = cast<CallbackCallOp>(op);

        bufferization::BufferizeTypeConverter typeConverter;

        SmallVector<Type> convertedResults;
        if (failed(typeConverter.convertTypes(callOp.getResultTypes(), convertedResults)))
            return failure();

        if (callOp->getNumResults() != convertedResults.size())
            return failure();

        SmallVector<Value> newInputs;
        auto operands = callOp.getOperands();
        for (Value operand : operands) {
            FailureOr<Value> opBuffer = getBuffer(rewriter, operand, options);
            if (failed(opBuffer))
                return failure();
            newInputs.push_back(*opBuffer);
        }

        auto results = callOp.getResults();
        auto loc = callOp->getLoc();
        SmallVector<Value> outmemrefs;
        for (auto result : results) {
            FailureOr<Value> tensorAlloc =
                bufferization::allocateTensorForShapedValue(rewriter, loc, result, options, false);
            if (failed(tensorAlloc))
                return failure();

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
        rewriter.create<CallbackCallOp>(loc, emptyRets, callOp.getCallee(), newInputs);
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