add_mlir_dialect(MitigationOps mitigation)
add_mlir_doc(MitigationDialect MitigationDialect Mitigation/ -gen-dialect-doc)
add_mlir_doc(MitigationOps MitigationOps Mitigation/ -gen-op-doc)

set(LLVM_TARGET_DEFINITIONS MitigationOps.td)
mlir_tablegen(MitigationEnums.h.inc -gen-enum-decls)
mlir_tablegen(MitigationEnums.cpp.inc -gen-enum-defs)
mlir_tablegen(MitigationAttributes.h.inc -gen-attrdef-decls -attrdefs-dialect=mitigation)
mlir_tablegen(MitigationAttributes.cpp.inc -gen-attrdef-defs -attrdefs-dialect=mitigation)
add_public_tablegen_target(MLIRMitigationEnumsIncGen)
