set(LIBRARY_NAME qec-transforms)


file(GLOB SRC
    CliffordTToPPR.cpp
    clifford_t_to_ppr.cpp
    CommuteCliffordTPPR.cpp
    commute_clifford_t_ppr.cpp
    CommuteCliffordPastPPM.cpp
    commute_clifford_past_ppm.cpp
    CountPPMSpecs.cpp
    decompose_non_clifford_ppr.cpp
    DecomposeNonCliffordPPR.cpp
    decompose_clifford_ppr.cpp
    DecomposeCliffordPPR.cpp
    PPRDecomposeUtils.cpp
    clifford_t_to_ppm.cpp
)

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
set(LIBS
    ${dialect_libs}
    ${conversion_libs}
    MLIRQEC
    QECUtils
)

set(DEPENDS
    MLIRQECPassIncGen
)

add_mlir_library(${LIBRARY_NAME} STATIC ${SRC} LINK_LIBS PRIVATE ${LIBS} DEPENDS ${DEPENDS})
target_compile_features(${LIBRARY_NAME} PUBLIC cxx_std_20)

include(FetchContent)
FetchContent_Declare(json
    URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz
    URL_HASH  SHA256=d6c65aca6b1ed68e7a182f4757257b107ae403032760ed6ef121c9d55e81757d
    DOWNLOAD_EXTRACT_TIMESTAMP true
    SYSTEM
)
FetchContent_MakeAvailable(json)
target_link_libraries(${LIBRARY_NAME} 
    PRIVATE nlohmann_json::nlohmann_json
)

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