get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
get_property(extension_libs GLOBAL PROPERTY MLIR_EXTENSION_LIBS)
set(LIBS
    ${dialect_libs}
    ${conversion_libs}
    ${extension_libs}
    MLIROptLib
    MLIRCatalyst
    MLIRCatalystTransforms
    MLIRQuantum
    quantum-transforms
    MLIRGradient
    gradient-transforms
    MLIRMitigation
    mitigation-transforms
    MhloRegisterDialects
    StablehloRegister
    MLIRCatalystTest
    ${ALL_MHLO_PASSES}
)

add_llvm_executable(quantum-opt quantum-opt.cpp)
target_link_libraries(quantum-opt PRIVATE ${LIBS})
llvm_update_compile_flags(quantum-opt)
mlir_check_all_link_libraries(quantum-opt)
