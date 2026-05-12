set(LLVM_TARGET_DEFINITIONS Passes.td)

#### Uncomment the following lines if your dialect defines any enums
# mlir_tablegen(PassesEnums.h.inc -gen-enum-decls)
# mlir_tablegen(PassesEnums.cpp.inc -gen-enum-defs)

mlir_tablegen(Passes.h.inc -gen-pass-decls -name @@@NewDialect@@@)
add_public_tablegen_target(MLIR@@@NewDialect@@@PassIncGen)
add_mlir_doc(Passes @@@NewDialect@@@Passes ./ -gen-pass-doc)
