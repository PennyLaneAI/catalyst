#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
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
    : public bufferization::BufferizableOpInterface::ExternalModel<StateOpInterface,
                                                    catalyst::quantum::StateOp> {
  bool bufferizesToMemoryRead(Operation *op, OpOperand &opOperand,
                              const bufferization::AnalysisState &state) const {
    return false;
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
    auto stateOp = cast<StateOp>(op);
    Location loc = op->getLoc();
    auto tensorType = cast<RankedTensorType>(stateOp.getState().getType());
    MemRefType resultType = MemRefType::get(tensorType.getShape(), tensorType.getElementType());

    Value allocVal = rewriter.create<memref::AllocOp>(loc, resultType);
    rewriter.create<StateOp>(loc, TypeRange{}, ValueRange{stateOp.getObs(), allocVal});
    bufferization::replaceOpWithBufferizedValues(rewriter, op, allocVal);

    return success();
  }
};

/// Bufferization of catalyst.quantum.probs. Replace with memref.alloc and a new
/// catalyst.quantum.probs that uses the memory allocated by memref.alloc.
struct ProbsOpInterface
    : public bufferization::BufferizableOpInterface::ExternalModel<ProbsOpInterface,
                                                    catalyst::quantum::ProbsOp> {
  bool bufferizesToMemoryRead(Operation *op, OpOperand &opOperand,
                              const bufferization::AnalysisState &state) const {
    return false;
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
    auto probsOp = cast<ProbsOp>(op);
    Location loc = op->getLoc();
    auto tensorType = cast<RankedTensorType>(probsOp.getProbabilities().getType());
    MemRefType resultType = MemRefType::get(tensorType.getShape(), tensorType.getElementType());

    Value allocVal = rewriter.create<memref::AllocOp>(loc, resultType);
    rewriter.create<ProbsOp>(loc, TypeRange{}, ValueRange{probsOp.getObs(), allocVal});
    bufferization::replaceOpWithBufferizedValues(rewriter, op, allocVal);

    return success();
  }
};

/// Bufferization of catalyst.quantum.counts. Replace with memref.allocs and a new
/// catalyst.quantum.counts that uses the memory allocated by memref.allocs.
struct CountsOpInterface
    : public bufferization::BufferizableOpInterface::ExternalModel<CountsOpInterface,
                                                    catalyst::quantum::CountsOp> {
  bool bufferizesToMemoryRead(Operation *op, OpOperand &opOperand,
                              const bufferization::AnalysisState &state) const {
    return false;
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
    bufferization::replaceOpWithBufferizedValues(rewriter, op, ValueRange{allocVal0, allocVal1});

    return success();
  }
};

/// Bufferization of catalyst.quantum.set_state. Replace with bufferization::ToMemrefOp and
/// a new catalyst.quantum.set_state that uses the memref from bufferization::ToMemrefOp.
struct SetStateOpInterface
    : public bufferization::BufferizableOpInterface::ExternalModel<SetStateOpInterface,
                                                    catalyst::quantum::SetStateOp> {
  bool bufferizesToMemoryRead(Operation *op, OpOperand &opOperand,
                              const bufferization::AnalysisState &state) const {
    return false;
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
    auto setStateOp = cast<SetStateOp>(op);
    Location loc = op->getLoc();
    auto tensorType = cast<RankedTensorType>(setStateOp.getInState().getType());
    MemRefType memrefType = MemRefType::get(tensorType.getShape(), tensorType.getElementType());

    auto toMemrefOp = rewriter.create<bufferization::ToMemrefOp>(loc, memrefType,
                                                                 setStateOp.getInState());
    auto memref = toMemrefOp.getResult();
    auto newSetStateOp = rewriter.create<SetStateOp>(loc, setStateOp.getOutQubits().getTypes(),
                                                     memref, setStateOp.getInQubits());
    bufferization::replaceOpWithBufferizedValues(rewriter, op, newSetStateOp.getOutQubits());
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
    SetStateOp::attachInterface<SetStateOpInterface>(*ctx);
  });
}