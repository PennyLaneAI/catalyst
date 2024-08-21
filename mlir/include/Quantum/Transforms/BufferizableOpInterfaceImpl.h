#ifndef MLIR_DIALECT_QUANTUM_BUFFERIZABLEOPINTERFACEIMPL_H
#define MLIR_DIALECT_QUANTUM_BUFFERIZABLEOPINTERFACEIMPL_H

namespace mlir {
class DialectRegistry;

namespace tensor {
void registerBufferizableOpInterfaceExternalModels(DialectRegistry &registry);
} // namespace tensor
} // namespace mlir

#endif // MLIR_DIALECT_QUANTUM_BUFFERIZABLEOPINTERFACEIMPL_H