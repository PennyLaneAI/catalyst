set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name Ion)
add_public_tablegen_target(MLIRIonPassIncGen)
add_mlir_doc(Passes IonPasses ./ -gen-pass-doc)
