set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name catalyst)
add_public_tablegen_target(MLIRCatalystPassIncGen)
add_mlir_doc(Passes CatalystPasses ./ -gen-pass-doc)
