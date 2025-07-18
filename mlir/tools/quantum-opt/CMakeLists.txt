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
    MLIRTestDialect
    ${ALL_MHLO_PASSES}
)

add_mlir_tool(quantum-opt quantum-opt.cpp DEPENDS ${LIBS} SUPPORT_PLUGINS)
target_link_libraries(quantum-opt PRIVATE ${LIBS})
llvm_update_compile_flags(quantum-opt)
mlir_check_all_link_libraries(quantum-opt)
export_executable_symbols_for_plugins(quantum-opt)

# Read catalyst/frontend/catalyst/_version.py, extract the version, and write it out to a C++ header file
function(make_version_hpp_file input_file output_file)
    file(READ ${input_file} input_content)
    if (input_content MATCHES "__version__ = \"([^\"]+)\"")
        set(version_number "${CMAKE_MATCH_1}")
    else()
        message(WARNING "Could not find version regex in ${input_file}")
        set(version_number "unknown")
    endif()
    set(header_content "const char* CATALYST_VERSION = \"${version_number}\";")
    file(WRITE ${output_file} "${header_content}")
endfunction(make_version_hpp_file)

make_version_hpp_file(
    "${PROJECT_SOURCE_DIR}/../frontend/catalyst/_version.py"
    "${PROJECT_BINARY_DIR}/include/frontend_catalyst_version_py.hpp"
)