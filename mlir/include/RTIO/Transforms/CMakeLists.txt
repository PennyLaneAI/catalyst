set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name RTIO)
add_public_tablegen_target(MLIRRTIOPassIncGen)
add_mlir_doc(Passes RTIOPasses ./ -gen-pass-doc)
