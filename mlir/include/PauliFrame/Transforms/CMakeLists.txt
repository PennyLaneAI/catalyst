set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name PauliFrame)
add_public_tablegen_target(MLIRPauliFramePassIncGen)
add_mlir_doc(Passes PauliFramePasses ./ -gen-pass-doc)
