set(LIBRARY_NAME catalyst-stablehlo-transforms)

file(GLOB SRC
    stablehlo_legalize_control_flow.cpp
    stablehlo_legalize_sort.cpp
    stablehlo_legalize_to_std.cpp
)

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
set(LIBS
    ${dialect_libs}
    ${conversion_libs}
    StablehloPasses
    StablehloOps
)

set(DEPENDS
    MLIRCatalystPassIncGen
    STABLEHLOCatalystPassIncGen
    MLIRStablehloLegalizeToStandardIncGen
    StablehloBaseIncGen
    StablehloOpsIncGen
)

add_mlir_library(${LIBRARY_NAME} STATIC ${SRC} LINK_LIBS PRIVATE ${LIBS} DEPENDS ${DEPENDS})
target_compile_features(${LIBRARY_NAME} PUBLIC cxx_std_20)
target_include_directories(${LIBRARY_NAME} PUBLIC
                           .
                           ${PROJECT_SOURCE_DIR}/include
                           ${CMAKE_BINARY_DIR}/include)
