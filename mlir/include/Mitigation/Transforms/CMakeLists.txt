set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name mitigation)
add_public_tablegen_target(MLIRMitigationPassIncGen)
add_mlir_doc(Passes MitigationPasses ./ -gen-pass-doc)
