add_mlir_dialect(QECDialect qec)
add_mlir_doc(QECDialect QECDialect QEC/ -gen-dialect-doc -gen-op-doc)

set(LLVM_TARGET_DEFINITIONS QECDialect.td)
