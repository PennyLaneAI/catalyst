add_mlir_dialect(QECDialect qec)
add_mlir_doc(QECDialect QECDialect QEC/ -gen-dialect-doc -gen-op-doc)

set(LLVM_TARGET_DEFINITIONS QECDialect.td)
mlir_tablegen(QECEnums.h.inc -gen-enum-decls)
mlir_tablegen(QECEnums.cpp.inc -gen-enum-defs)
mlir_tablegen(QECAttributes.h.inc -gen-attrdef-decls -attrdefs-dialect=qec)
mlir_tablegen(QECAttributes.cpp.inc -gen-attrdef-defs -attrdefs-dialect=qec)
add_public_tablegen_target(MLIRQECEnumsIncGen)
