set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name MBQC)
add_public_tablegen_target(MLIRMBQCPassIncGen)
add_mlir_doc(Passes MBQCPasses ./ -gen-pass-doc)
