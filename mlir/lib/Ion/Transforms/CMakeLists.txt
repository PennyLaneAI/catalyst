set(LIBRARY_NAME ion-transforms)

file(GLOB SRC
    ion-to-llvm.cpp
    ConversionPatterns.cpp
    quantum_to_ion.cpp
    QuantumToIonPatterns.cpp
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
target_compile_features(${LIBRARY_NAME} PUBLIC cxx_std_17)
target_include_directories(${LIBRARY_NAME} PUBLIC
                           .
                           ${PROJECT_SOURCE_DIR}/include
                           ${CMAKE_BINARY_DIR}/include)

include(FetchContent)
FetchContent_Declare(
    tomlplusplus
    GIT_REPOSITORY https://github.com/marzer/tomlplusplus.git
    GIT_TAG v3.4.0
)
FetchContent_MakeAvailable(tomlplusplus)

# Fetch json utilities
FetchContent_Declare(
   json
   URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz
   URL_HASH  SHA256=d6c65aca6b1ed68e7a182f4757257b107ae403032760ed6ef121c9d55e81757d
)
FetchContent_MakeAvailable(json)
target_link_libraries(ion-transforms
    PRIVATE nlohmann_json::nlohmann_json
)

target_link_libraries(${LIBRARY_NAME} PRIVATE tomlplusplus::tomlplusplus)

# tmol++ source has a warning which we want to ignore
# This is because toml++ ships source-only, not an executable, so we have to compile it
set_source_files_properties(
  ion-to-llvm.cpp
  ConversionPatterns.cpp
  quantum_to_ion.cpp
  QuantumToIonPatterns.cpp
  PROPERTIES
  COMPILE_FLAGS "-Wno-covered-switch-default"
)
