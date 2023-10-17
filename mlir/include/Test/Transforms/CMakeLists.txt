set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name test)
add_public_tablegen_target(MLIRCatalystTestIncGen)
add_mlir_doc(Passes TestPasses ./ -gen-pass-doc)

