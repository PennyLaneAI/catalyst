set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name QecPhysical)
add_public_tablegen_target(MLIRQecPhysicalPassIncGen)
add_mlir_doc(Passes QecPhysicalPasses ./ -gen-pass-doc)
