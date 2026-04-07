set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name Quantum)
add_public_tablegen_target(MLIRQuantumPassIncGen)
add_mlir_doc(Passes QuantumPasses ./ -gen-pass-doc)
