set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name openqasm)
add_public_tablegen_target(MLIROpenQASMPassIncGen)
add_mlir_doc(Passes OpenQASMPasses ./ -gen-pass-doc)
