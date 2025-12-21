set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name RefQuantum)
add_public_tablegen_target(MLIRRefQuantumPassIncGen)
add_mlir_doc(Passes RefQuantumPasses ./ -gen-pass-doc)
