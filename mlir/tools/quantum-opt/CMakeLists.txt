get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
get_property(extension_libs GLOBAL PROPERTY MLIR_EXTENSION_LIBS)
set(LIBS
    ${dialect_libs}
    ${conversion_libs}
    ${extension_libs}
    ExternalStablehloLib
    MLIROptLib
    MLIRCatalyst
    catalyst-transforms
    catalyst-stablehlo-transforms
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
    MLIRCatalystTest
    MLIRCatalystUtils
    MLIRTestDialect
)

set(STANDALONE_SOURCE_DIR ${PROJECT_SOURCE_DIR}/standalone)
set(STANDALONE_BUILD_DIR ${PROJECT_SOURCE_DIR}/standalone/build)
include_directories(${STANDALONE_SOURCE_DIR}/include)
include_directories(${STANDALONE_BUILD_DIR}/include)

add_mlir_tool(quantum-opt quantum-opt.cpp
    ${STANDALONE_SOURCE_DIR}/lib/Standalone/StandaloneDialect.cpp
    ${STANDALONE_SOURCE_DIR}/lib/Standalone/StandaloneOps.cpp
    ${STANDALONE_SOURCE_DIR}/lib/Standalone/StandalonePasses.cpp
    ${STANDALONE_SOURCE_DIR}/lib/Standalone/StandaloneTypes.cpp
    DEPENDS ${LIBS} SUPPORT_PLUGINS)
target_link_libraries(quantum-opt PRIVATE ${LIBS})
llvm_update_compile_flags(quantum-opt)
mlir_check_all_link_libraries(quantum-opt)
export_executable_symbols_for_plugins(quantum-opt)
