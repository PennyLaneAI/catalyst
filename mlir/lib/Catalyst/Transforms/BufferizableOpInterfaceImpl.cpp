#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Transforms/DialectConversion.h"

#include "Catalyst/IR/CatalystOps.h"
#include "Catalyst/Transforms/BufferizableOpInterfaceImpl.h"

using namespace mlir;
using namespace catalyst;

namespace {

/// Bufferization of catalyst.print. Get memref of printOp.val.
struct PrintOpInterface
    : public bufferization::BufferizableOpInterface::ExternalModel<PrintOpInterface,
                                                    PrintOp> {
    bool bufferizesToMemoryRead(Operation *op, OpOperand &opOperand,
                                const bufferization::AnalysisState &state) const {
        return true;
    }

    bool bufferizesToMemoryWrite(Operation *op, OpOperand &opOperand,
                                 const bufferization::AnalysisState &state) const {
        return false;
    }

    bufferization::AliasingValueList getAliasingValues(Operation *op,
                                        OpOperand &opOperand,
                                        const bufferization::AnalysisState &state) const {
        return {};
    }

    LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                            const bufferization::BufferizationOptions &options) const {
        auto printOp = cast<PrintOp>(op);
        if (printOp.getVal()) {
            FailureOr<Value> source = getBuffer(rewriter, printOp.getVal(), options);
            if (failed(source))
                return failure();
            bufferization::replaceOpWithNewBufferizedOp<PrintOp>(rewriter, op, *source,
                            printOp.getConstValAttr(), printOp.getPrintDescriptorAttr());
        }
        return success();
    }
};

struct CustomCallOpInterface
    : public bufferization::BufferizableOpInterface::ExternalModel<CustomCallOpInterface,
                                                    CustomCallOp> {
    bool bufferizesToMemoryRead(Operation *op, OpOperand &opOperand,
                                const bufferization::AnalysisState &state) const {
        return true;
    }

    bool bufferizesToMemoryWrite(Operation *op, OpOperand &opOperand,
                                 const bufferization::AnalysisState &state) const {
        return true;
    }

    bufferization::AliasingValueList getAliasingValues(Operation *op,
                                        OpOperand &opOperand,
                                        const bufferization::AnalysisState &state) const {
        return {};
    }

    LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                            const bufferization::BufferizationOptions &options) const {
        auto customCallOp = cast<CustomCallOp>(op);

        // Add bufferized arguments
        SmallVector<Value> bufferArgs;
        ValueRange operands = customCallOp.getOperands();
        for (Value operand : operands) {
            FailureOr<Value> opBuffer = getBuffer(rewriter, operand, options);
            if (failed(opBuffer))
                return failure();
            bufferArgs.push_back(*opBuffer);
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

} // namespace

void catalyst::registerBufferizableOpInterfaceExternalModels(
    DialectRegistry &registry) {
    registry.addExtension(+[](MLIRContext *ctx, CatalystDialect *dialect) {
        CustomCallOp::attachInterface<CustomCallOpInterface>(*ctx);
        PrintOp::attachInterface<PrintOpInterface>(*ctx);
    });
}