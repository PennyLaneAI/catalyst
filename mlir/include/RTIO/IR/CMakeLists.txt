add_mlir_dialect(RTIOOps rtio)
add_mlir_doc(RTIODialect RTIODialect RTIO/ -gen-dialect-doc)
add_mlir_doc(RTIOOps RTIOOps RTIO/ -gen-op-doc)

set(LLVM_TARGET_DEFINITIONS RTIOOps.td)
mlir_tablegen(RTIOAttributes.h.inc -gen-attrdef-decls -attrdefs-dialect=rtio)
mlir_tablegen(RTIOAttributes.cpp.inc -gen-attrdef-defs -attrdefs-dialect=rtio)
add_public_tablegen_target(MLIRRTIOAttributesIncGen)
