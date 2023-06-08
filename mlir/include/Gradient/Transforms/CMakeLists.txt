set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name gradient)
add_public_tablegen_target(MLIRGradientPassIncGen)
add_mlir_doc(Passes GradientPasses ./ -gen-pass-doc)
