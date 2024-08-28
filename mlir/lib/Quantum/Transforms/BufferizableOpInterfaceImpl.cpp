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

/// Bufferization of catalyst.quantum.counts. Replace with memref.alloc and a new
/// catalyst.quantum.counts that uses the memory allocated by memref.alloc.
struct CountsOpInterface
    : public mlir::bufferization::BufferizableOpInterface::ExternalModel<CountsOpInterface,
                                                    catalyst::quantum::CountsOp> {
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
    auto countsOp = cast<CountsOp>(op);
    Location loc = op->getLoc();
    auto tensorType0 = cast<RankedTensorType>(countsOp.getEigvals().getType());
    auto tensorType1 = cast<RankedTensorType>(countsOp.getCounts().getType());
    MemRefType resultType0 = MemRefType::get(tensorType0.getShape(), tensorType0.getElementType());
    MemRefType resultType1 = MemRefType::get(tensorType1.getShape(), tensorType1.getElementType());

    Value allocVal0 = rewriter.create<memref::AllocOp>(loc, resultType0);
    Value allocVal1 = rewriter.create<memref::AllocOp>(loc, resultType1);
    rewriter.create<CountsOp>(loc, nullptr, nullptr, countsOp.getObs(), allocVal0, allocVal1,
                              countsOp.getShotsAttr());
    mlir::bufferization::replaceOpWithBufferizedValues(rewriter, op, ValueRange{allocVal0, allocVal1});

    return success();
  }
};

} // namespace

void catalyst::quantum::registerBufferizableOpInterfaceExternalModels(
    DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *ctx, catalyst::quantum::QuantumDialect *dialect) {
    StateOp::attachInterface<StateOpInterface>(*ctx);
    ProbsOp::attachInterface<ProbsOpInterface>(*ctx);
    CountsOp::attachInterface<CountsOpInterface>(*ctx);
  });
}