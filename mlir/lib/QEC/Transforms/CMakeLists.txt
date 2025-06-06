set(LIBRARY_NAME qec-transforms)


file(GLOB SRC
    CliffordTToPPR.cpp
    clifford_t_to_ppr.cpp
    CommuteCliffordTPPR.cpp
    commute_clifford_t_ppr.cpp
    CommuteCliffordPastPPM.cpp
    commute_clifford_past_ppm.cpp
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
target_include_directories(${LIBRARY_NAME} PUBLIC
                           .
                           ${PROJECT_SOURCE_DIR}/include
                           ${CMAKE_BINARY_DIR}/include)
