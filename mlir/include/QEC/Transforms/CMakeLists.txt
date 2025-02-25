set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name qec)
add_public_tablegen_target(MLIRQECPassIncGen)
add_mlir_doc(Passes QECPasses ./ -gen-pass-doc)
