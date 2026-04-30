add_mlir_dialect(CatalystOps catalyst)
add_mlir_doc(CatalystDialect CatalystDialect Catalyst/ -gen-dialect-doc)
add_mlir_doc(CatalystOps CatalystOps Catalyst/ -gen-op-doc)
add_mlir_doc(CatalystAttributes CatalystAttributes Catalyst/ -gen-attrdef-doc)

set(LLVM_TARGET_DEFINITIONS CatalystAttributes.td)
mlir_tablegen(CatalystAttributes.h.inc -gen-attrdef-decls -attrdefs-dialect=catalyst)
mlir_tablegen(CatalystAttributes.cpp.inc -gen-attrdef-defs -attrdefs-dialect=catalyst)
add_public_tablegen_target(MLIRCatalystAttributesIncGen)
