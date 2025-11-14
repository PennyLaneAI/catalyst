add_mlir_dialect(PauliFrameOps pauli_frame)
add_mlir_doc(PauliFrameDialect PauliFrameDialect PauliFrame/ -gen-dialect-doc -gen-op-doc)
add_mlir_doc(PauliFrameOps PauliFrameOps PauliFrame/ -gen-op-doc)

set(LLVM_TARGET_DEFINITIONS PauliFrameOps.td)
