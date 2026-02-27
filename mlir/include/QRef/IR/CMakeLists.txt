add_mlir_dialect(QRefOps qref)
add_mlir_interface(QRefInterfaces)
add_mlir_doc(QRefDialect QRefDialect QRef/ -gen-dialect-doc -gen-op-doc)
add_mlir_doc(QRefOps QRefOps QRef/ -gen-op-doc)
add_mlir_doc(QRefInterfaces QRefInterfaces QRef/ -gen-op-interface-docs)

set(LLVM_TARGET_DEFINITIONS QRefOps.td)
mlir_tablegen(QRefAttributes.h.inc -gen-attrdef-decls -attrdefs-dialect=qref)
mlir_tablegen(QRefAttributes.cpp.inc -gen-attrdef-defs -attrdefs-dialect=qref)
