get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
set(LIBS
    ${dialect_libs}
    ${conversion_libs}
    MLIRLspServerLib
    MLIRCatalyst
    MLIRQuantum
    MLIRGradient
    MLIRMitigation
    MLIRIon
    MhloRegisterDialects
    StablehloRegister
)

add_llvm_executable(quantum-lsp-server quantum-lsp-server.cpp)
target_link_libraries(quantum-lsp-server PRIVATE ${LIBS})
llvm_update_compile_flags(quantum-lsp-server)
mlir_check_all_link_libraries(quantum-lsp-server)
