get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
set(LIBS
    ${dialect_libs}
    ${conversion_libs}
    MLIROptLib
    MLIRQuantum
    quantum-transforms
    MLIRGradient
    gradient-transforms
)

add_llvm_executable(quantum-opt quantum-opt.cpp)
target_link_libraries(quantum-opt PRIVATE ${LIBS})
llvm_update_compile_flags(quantum-opt)
mlir_check_all_link_libraries(quantum-opt)
