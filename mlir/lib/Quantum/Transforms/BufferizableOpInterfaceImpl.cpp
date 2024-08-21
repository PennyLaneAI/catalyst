#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"

#include "Quantum/IR/QuantumOps.h"
#include "Quantum/Transforms/BufferizableOpInterfaceImpl.h"

using namespace mlir;
using namespace catalyst::quantum;

namespace {

/// Bufferization of tensor.extract. Replace with memref.load.
struct ExtractOpInterface
    : public mlir::bufferization::BufferizableOpInterface::ExternalModel<ExtractOpInterface,
                                                    catalyst::quantum::ExtractOp> {
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
    auto extractOp = cast<catalyst::quantum::ExtractOp>(op);

    return success();
  }
};

}

void catalyst::quantum::registerBufferizableOpInterfaceExternalModels(
    DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *ctx, catalyst::quantum::QuantumDialect *dialect) {
    ExtractOp::attachInterface<ExtractOpInterface>(*ctx);
  });
}