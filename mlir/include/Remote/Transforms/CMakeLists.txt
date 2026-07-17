set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name Remote)
add_public_tablegen_target(MLIRRemotePassIncGen)
add_mlir_doc(Passes RemotePasses ./ -gen-pass-doc)
