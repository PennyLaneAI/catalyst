set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name Executor)
add_public_tablegen_target(MLIRExecutorPassIncGen)
add_mlir_doc(Passes ExecutorPasses ./ -gen-pass-doc)
