add_mlir_dialect(TransportOps transport)
add_mlir_doc(TransportDialect TransportDialect Transport/ -gen-dialect-doc)
add_mlir_doc(TransportOps TransportOps Transport/ -gen-op-doc)

set(LLVM_TARGET_DEFINITIONS TransportOps.td)
mlir_tablegen(TransportEnums.h.inc -gen-enum-decls)
mlir_tablegen(TransportEnums.cpp.inc -gen-enum-defs)
add_public_tablegen_target(MLIRTransportEnumsIncGen)
