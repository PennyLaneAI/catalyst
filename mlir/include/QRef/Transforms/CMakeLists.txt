set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name QRef)
add_public_tablegen_target(MLIRQRefPassIncGen)
add_mlir_doc(Passes QRefPasses ./ -gen-pass-doc)
