#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Transforms/DialectConversion.h"

#include "Catalyst/IR/CatalystOps.h"
#include "Catalyst/Transforms/BufferizableOpInterfaceImpl.h"

using namespace mlir;
using namespace catalyst;

namespace {

/// Bufferization of catalyst.quantum.hermitian. Convert Matrix into memref.
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

} // namespace

void catalyst::registerBufferizableOpInterfaceExternalModels(
    DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *ctx, CatalystDialect *dialect) {
    PrintOp::attachInterface<PrintOpInterface>(*ctx);
  });
}