add_mlir_dialect(GradientOps gradient)
add_mlir_interface(GradientInterfaces)
add_mlir_doc(GradientDialect GradientDialect Gradient/ -gen-dialect-doc)
add_mlir_doc(GradientOps GradientOps Gradient/ -gen-op-doc)
add_mlir_doc(GradientInterfaces GradientInterfaces Gradient/ -gen-op-interface-docs)
add_mlir_doc(GradientAttrDefs GradientAttributes Gradient/ -gen-attrdef-doc)

set(LLVM_TARGET_DEFINITIONS GradientAttrDefs.td)
mlir_tablegen(GradientEnums.h.inc -gen-enum-decls)
mlir_tablegen(GradientEnums.cpp.inc -gen-enum-defs)
mlir_tablegen(GradientAttributes.h.inc -gen-attrdef-decls -attrdefs-dialect=gradient)
mlir_tablegen(GradientAttributes.cpp.inc -gen-attrdef-defs -attrdefs-dialect=gradient)
add_public_tablegen_target(MLIRGradientAttrDefsIncGen)

