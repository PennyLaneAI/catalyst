
get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(translation_libs GLOBAL PROPERTY MLIR_TRANSLATION_LIBS)

set(LIBS
    ${dialect_libs}
    ${translation_libs}
    MLIRQuantum
    MlirToQasm3Translation
)

add_mlir_tool(quantum-translate quantum-translate.cpp DEPENDS ${LIBS})
target_link_libraries(quantum-translate PRIVATE ${LIBS})
llvm_update_compile_flags(quantum-translate)
mlir_check_all_link_libraries(quantum-translate)
