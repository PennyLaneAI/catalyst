set(LIBRARY_NAME ion-transforms)

file(GLOB SRC
    ion-to-llvm.cpp
    ion-to-rtio.cpp
    IonToRTIOPatterns.cpp
    ConversionPatterns.cpp
    gates_to_pulses.cpp
    GatesToPulsesPatterns.cpp
)

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

if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    set_source_files_properties(${SRC} PROPERTIES 
        COMPILE_OPTIONS "-Wno-deprecated-literal-operator;-Wno-covered-switch-default"
    )
endif()

add_mlir_library(${LIBRARY_NAME} STATIC ${SRC} LINK_LIBS PRIVATE ${LIBS} DEPENDS ${DEPENDS})

target_compile_features(${LIBRARY_NAME} PUBLIC cxx_std_20)
target_include_directories(${LIBRARY_NAME} PUBLIC
                           .
                           ${PROJECT_SOURCE_DIR}/include
                           ${CMAKE_BINARY_DIR}/include)

target_link_libraries(${LIBRARY_NAME} PRIVATE
    tomlplusplus::tomlplusplus
    nlohmann_json::nlohmann_json
)
