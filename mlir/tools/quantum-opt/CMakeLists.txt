get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
get_property(extension_libs GLOBAL PROPERTY MLIR_EXTENSION_LIBS)

find_package(Python3 COMPONENTS Development REQUIRED)

set(LIBS
    ${dialect_libs}
    ${conversion_libs}
    ${extension_libs}
    ExternalStablehloLib
    MLIROptLib
    MLIRRegisterAllDialects
    MLIRRegisterAllPasses
    MLIRCatalyst
    catalyst-analysis
    catalyst-transforms
    catalyst-stablehlo-transforms
    MLIRQecLogical
    qecp-transforms
    MLIRQuantum
    quantum-transforms
    MLIRQRef
    qref-transforms
    MLIRPBC
    pbc-transforms
    MLIRGradient
    gradient-transforms
    MLIRMBQC
    mbqc-transforms
    MLIRMitigation
    mitigation-transforms
    MLIRPauliFrame
    pauli-frame-transforms
    MLIRIon
    ion-transforms
    MLIRRTIO
    rtio-transforms
    MLIRQecLogical
    MLIRQecPhysical
    MLIRCatalystTest
    MLIRCatalystUtils
    MLIRTestDialect
    QuantumPythonCallbacks
    pybind11::embed
    Python3::Python
)

add_mlir_tool(quantum-opt quantum-opt.cpp DEPENDS ${LIBS} SUPPORT_PLUGINS)
target_link_libraries(quantum-opt PRIVATE ${LIBS})
llvm_update_compile_flags(quantum-opt)
mlir_check_all_link_libraries(quantum-opt)
export_executable_symbols_for_plugins(quantum-opt)
