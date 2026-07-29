add_mlir_dialect(@@@NewDialect@@@Dialect @@@new_dialect@@@)
add_mlir_doc(@@@NewDialect@@@Dialect @@@NewDialect@@@Dialect @@@NewDialect@@@/ -gen-dialect-doc -gen-op-doc)

set(LLVM_TARGET_DEFINITIONS @@@NewDialect@@@Dialect.td)

#### Uncomment the following lines if your dialect defines any enums
# mlir_tablegen(@@@NewDialect@@@Enums.h.inc -gen-enum-decls)
# mlir_tablegen(@@@NewDialect@@@Enums.cpp.inc -gen-enum-defs)

#### Uncomment the following lines if your dialect defines any attributes
# mlir_tablegen(@@@NewDialect@@@Attributes.h.inc -gen-attrdef-decls)
# mlir_tablegen(@@@NewDialect@@@Attributes.cpp.inc -gen-attrdef-defs)

add_public_tablegen_target(MLIR@@@NewDialect@@@EnumsIncGen)
