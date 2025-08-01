get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
get_property(extension_libs GLOBAL PROPERTY MLIR_EXTENSION_LIBS)
set(LIBS
    ${dialect_libs}
    ${conversion_libs}
    ${extension_libs}
    MLIROptLib
    MLIRCatalyst
    catalyst-transforms
    catalyst-mhlo-transforms
    MLIRQuantum
    quantum-transforms
    MLIRQEC
    qec-transforms
    MLIRGradient
    gradient-transforms
    MLIRMBQC
    mbqc-transforms
    MLIRMitigation
    mitigation-transforms
    MLIRIon
    ion-transforms
    MhloRegisterDialects
    StablehloRegister
    MLIRCatalystTest
    MLIRCatalystUtils
    MLIRTestDialect
    ${ALL_MHLO_PASSES}
)

add_mlir_tool(quantum-opt quantum-opt.cpp DEPENDS ${LIBS} SUPPORT_PLUGINS)
target_link_libraries(quantum-opt PRIVATE ${LIBS})
llvm_update_compile_flags(quantum-opt)
mlir_check_all_link_libraries(quantum-opt)
export_executable_symbols_for_plugins(quantum-opt)
