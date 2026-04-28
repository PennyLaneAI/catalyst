set(LIBRARY_NAME qecp-transforms)

file(GLOB SRC
    decoder.cpp
    DecoderPatterns.cpp
)

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
set(LIBS
    ${dialect_libs}
    ${conversion_libs}
    MLIRQecPhysical
)

set(DEPENDS
    MLIRPBCPassIncGen
)

add_mlir_library(${LIBRARY_NAME} STATIC ${SRC} LINK_LIBS PRIVATE ${LIBS} DEPENDS ${DEPENDS})
target_compile_features(${LIBRARY_NAME} PUBLIC cxx_std_20)

target_link_libraries(${LIBRARY_NAME} 
    PRIVATE nlohmann_json::nlohmann_json
)

# Disable warning from json library
if(CMAKE_CXX_COMPILER_ID MATCHES ".*Clang")
set_source_files_properties(
    CountPPMSpecs.cpp
    COMPILE_FLAGS "-Wno-covered-switch-default"
)
endif()

target_include_directories(${LIBRARY_NAME} PUBLIC
                           .
                           ${PROJECT_SOURCE_DIR}/include
                           ${CMAKE_BINARY_DIR}/include)