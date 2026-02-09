set(LIBRARY_NAME quantum-transforms)

file(GLOB SRC
    BufferizableOpInterfaceImpl.cpp
    ConversionPatterns.cpp
    quantum_to_llvm.cpp
    emit_catalyst_pyface.cpp
    cp_global_buffers.cpp
    adjoint_lowering.cpp
    AdjointPatterns.cpp
    CancelInversesPatterns.cpp
    cancel_inverses.cpp
    SplitMultipleTapes.cpp
    split_to_single_terms.cpp
    merge_rotation.cpp
    MergeRotationsPatterns.cpp
    decompose_lowering.cpp
    DecomposeLoweringPatterns.cpp
    gridsynth.cpp
    GridsynthPatterns.cpp
    DisentangleSWAP.cpp
    DisentangleCNOT.cpp
    ions_decompositions.cpp
    IonsDecompositionPatterns.cpp
    loop_boundary_optimization.cpp
    LoopBoundaryOptimizationPatterns.cpp
    one_shot_mcm.cpp
)

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
set(LIBS
    ${dialect_libs}
    ${conversion_libs}
    MLIRQuantum
    ExternalStablehloLib
)

set(DEPENDS
    MLIRQuantumPassIncGen
)

add_mlir_library(${LIBRARY_NAME} STATIC ${SRC} LINK_LIBS PRIVATE ${LIBS} DEPENDS ${DEPENDS})
target_compile_features(${LIBRARY_NAME} PUBLIC cxx_std_20)
target_include_directories(${LIBRARY_NAME} PUBLIC
                           .
                           ${PROJECT_SOURCE_DIR}/include
                           ${CMAKE_BINARY_DIR}/include)
