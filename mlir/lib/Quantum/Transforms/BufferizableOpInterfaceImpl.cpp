#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Transforms/DialectConversion.h"

#include "Quantum/IR/QuantumOps.h"
#include "Quantum/Transforms/BufferizableOpInterfaceImpl.h"

using namespace mlir;
using namespace catalyst::quantum;

namespace {

/// Bufferization of tensor.extract. Replace with memref.load.
struct StateOpInterface
    : public mlir::bufferization::BufferizableOpInterface::ExternalModel<StateOpInterface,
                                                    catalyst::quantum::StateOp> {
  bool bufferizesToMemoryRead(mlir::Operation *op, mlir::OpOperand &opOperand,
                              const mlir::bufferization::AnalysisState &state) const {
    return true;
  }

  bool bufferizesToMemoryWrite(mlir::Operation *op, mlir::OpOperand &opOperand,
                               const mlir::bufferization::AnalysisState &state) const {
    return true;
  }

  mlir::bufferization::AliasingValueList getAliasingValues(mlir::Operation *op,
                                      mlir::OpOperand &opOperand,
                                      const mlir::bufferization::AnalysisState &state) const {
    return {};
  }

  LogicalResult bufferize(mlir::Operation *op, RewriterBase &rewriter,
                          const mlir::bufferization::BufferizationOptions &options) const {
    auto stateOp = cast<StateOp>(op);
    Location loc = op->getLoc();
    auto tensorType = cast<RankedTensorType>(stateOp.getState().getType());
    MemRefType resultType = MemRefType::get(tensorType.getShape(), tensorType.getElementType());

    Value allocVal = rewriter.create<memref::AllocOp>(loc, resultType);
    rewriter.create<StateOp>(loc, TypeRange{}, ValueRange{stateOp.getObs(), allocVal});
    mlir::bufferization::replaceOpWithBufferizedValues(rewriter, op, allocVal);

    return success();
  }
};

}

void catalyst::quantum::registerBufferizableOpInterfaceExternalModels(
    DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *ctx, catalyst::quantum::QuantumDialect *dialect) {
    StateOp::attachInterface<StateOpInterface>(*ctx);
  });
}