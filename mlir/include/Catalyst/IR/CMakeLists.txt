add_mlir_dialect(CatalystOps catalyst)
add_mlir_doc(CatalystDialect CatalystDialect Catalyst/ -gen-dialect-doc)
add_mlir_doc(CatalystOps CatalystOps Catalyst/ -gen-op-doc)

set(LLVM_TARGET_DEFINITIONS CatalystOps.td)
mlir_tablegen(CatalystEnums.h.inc -gen-enum-decls)
mlir_tablegen(CatalystEnums.cpp.inc -gen-enum-defs)
mlir_tablegen(CatalystAttributes.h.inc -gen-attrdef-decls -attrdefs-dialect=catalyst)
mlir_tablegen(CatalystAttributes.cpp.inc -gen-attrdef-defs -attrdefs-dialect=catalyst)
add_public_tablegen_target(MLIRCatalystEnumsIncGen)

