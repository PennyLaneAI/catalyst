set(LIBRARY_NAME ion-transforms)

file(GLOB SRC
    ion-to-llvm.cpp
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
)

set(DEPENDS
    MLIRIonPassIncGen
)

add_mlir_library(${LIBRARY_NAME} STATIC ${SRC} LINK_LIBS PRIVATE ${LIBS} DEPENDS ${DEPENDS})
target_compile_features(${LIBRARY_NAME} PUBLIC cxx_std_20)
target_include_directories(${LIBRARY_NAME} PUBLIC
                           .
                           ${PROJECT_SOURCE_DIR}/include
                           ${CMAKE_BINARY_DIR}/include)

include(FetchContent)

FetchContent_Declare(tomlplusplus
    GIT_REPOSITORY https://github.com/marzer/tomlplusplus.git
    GIT_TAG v3.4.0
    SYSTEM
)
FetchContent_MakeAvailable(tomlplusplus)

FetchContent_Declare(json
    URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz
    URL_HASH  SHA256=d6c65aca6b1ed68e7a182f4757257b107ae403032760ed6ef121c9d55e81757d
    DOWNLOAD_EXTRACT_TIMESTAMP true
    SYSTEM
)
FetchContent_MakeAvailable(json)

target_link_libraries(${LIBRARY_NAME} PRIVATE
    tomlplusplus::tomlplusplus
    nlohmann_json::nlohmann_json
)

# toml++ source has a warning which we want to ignore. Surprisingly, the SYSTEM keyword does not
# seem to affect the headers, at least with this specific warning (only present on clang, not gcc).
if(CMAKE_CXX_COMPILER_ID MATCHES ".*Clang")
set_source_files_properties(
    ion-to-llvm.cpp
    ConversionPatterns.cpp
    gates_to_pulses.cpp
    GatesToPulsesPatterns.cpp
    PROPERTIES
    COMPILE_FLAGS "-Wno-covered-switch-default"
)
endif()
