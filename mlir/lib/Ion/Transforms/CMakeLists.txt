set(LIBRARY_NAME ion-transforms)

file(GLOB SRC
    ion-to-llvm.cpp
    ion-to-rtio.cpp
    IonToRTIOPatterns.cpp
    ConversionPatterns.cpp
    gates_to_pulses.cpp
    GatesToPulsesPatterns.cpp
)

include(CheckCXXCompilerFlag)

set(ION_WARNING_FLAGS "")

check_cxx_compiler_flag("-Wcovered-switch-default" HAS_COVERED_SWITCH)
if(HAS_COVERED_SWITCH)
    list(APPEND ION_WARNING_FLAGS "-Wno-covered-switch-default")
endif()

check_cxx_compiler_flag("-Wdeprecated-literal-operator" HAS_DEPRECATED_LITERAL)
if(HAS_DEPRECATED_LITERAL)
    list(APPEND ION_WARNING_FLAGS "-Wno-deprecated-literal-operator")
endif()

if(ION_WARNING_FLAGS)
    set_source_files_properties(${SRC} PROPERTIES 
        COMPILE_OPTIONS "${ION_WARNING_FLAGS}"
    )
endif()

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
set(LIBS
    ${dialect_libs}
    ${conversion_libs}
    MLIRIon
    MLIRRTIO
)

set(DEPENDS
    MLIRIonPassIncGen
)

add_mlir_library(${LIBRARY_NAME} STATIC ${SRC} LINK_LIBS PRIVATE ${LIBS} DEPENDS ${DEPENDS})
target_compile_features(${LIBRARY_NAME} PUBLIC cxx_std_20)
target_include_directories(${LIBRARY_NAME} PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

target_link_libraries(${LIBRARY_NAME} PRIVATE
    tomlplusplus::tomlplusplus
    nlohmann_json::nlohmann_json
)
