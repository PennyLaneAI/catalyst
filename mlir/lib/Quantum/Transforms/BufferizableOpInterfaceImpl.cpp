#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Transforms/DialectConversion.h"

#include "Quantum/IR/QuantumOps.h"
#include "Quantum/Transforms/BufferizableOpInterfaceImpl.h"

using namespace mlir;
using namespace catalyst::quantum;

namespace {

/// Bufferization of catalyst.quantum.state. Replace with memref.alloc and a new
/// catalyst.quantum.state that uses the memory allocated by memref.alloc.
struct StateOpInterface
    : public mlir::bufferization::BufferizableOpInterface::ExternalModel<StateOpInterface,
                                                    catalyst::quantum::StateOp> {
  bool bufferizesToMemoryRead(mlir::Operation *op, mlir::OpOperand &opOperand,
                              const mlir::bufferization::AnalysisState &state) const {
    return true;
  }

  bool bufferizesToMemoryWrite(mlir::Operation *op, mlir::OpOperand &opOperand,
                               const mlir::bufferization::AnalysisState &state) const {
    return false;
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

/// Bufferization of catalyst.quantum.probs. Replace with memref.alloc and a new
/// catalyst.quantum.probs that uses the memory allocated by memref.alloc.
struct ProbsOpInterface
    : public mlir::bufferization::BufferizableOpInterface::ExternalModel<ProbsOpInterface,
                                                    catalyst::quantum::ProbsOp> {
  bool bufferizesToMemoryRead(mlir::Operation *op, mlir::OpOperand &opOperand,
                              const mlir::bufferization::AnalysisState &state) const {
    return true;
  }

  bool bufferizesToMemoryWrite(mlir::Operation *op, mlir::OpOperand &opOperand,
                               const mlir::bufferization::AnalysisState &state) const {
    return false;
  }

  mlir::bufferization::AliasingValueList getAliasingValues(mlir::Operation *op,
                                      mlir::OpOperand &opOperand,
                                      const mlir::bufferization::AnalysisState &state) const {
    return {};
  }

  LogicalResult bufferize(mlir::Operation *op, RewriterBase &rewriter,
                          const mlir::bufferization::BufferizationOptions &options) const {
    auto probsOp = cast<ProbsOp>(op);
    Location loc = op->getLoc();
    auto tensorType = cast<RankedTensorType>(probsOp.getProbabilities().getType());
    MemRefType resultType = MemRefType::get(tensorType.getShape(), tensorType.getElementType());

    Value allocVal = rewriter.create<memref::AllocOp>(loc, resultType);
    rewriter.create<ProbsOp>(loc, TypeRange{}, ValueRange{probsOp.getObs(), allocVal});
    mlir::bufferization::replaceOpWithBufferizedValues(rewriter, op, allocVal);

    return success();
  }
};

} // namespace

void catalyst::quantum::registerBufferizableOpInterfaceExternalModels(
    DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *ctx, catalyst::quantum::QuantumDialect *dialect) {
    StateOp::attachInterface<StateOpInterface>(*ctx);
    ProbsOp::attachInterface<ProbsOpInterface>(*ctx);
  });
}