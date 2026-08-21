set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name Transport)
add_public_tablegen_target(MLIRTransportPassIncGen)
add_mlir_doc(Passes TransportPasses ./ -gen-pass-doc)
