set(LIBRARY_NAME quantum-transforms)

add_subdirectory(DecompGraphSolver)

file(GLOB SRC
    AdjointPatterns.cpp
    BufferizableOpInterfaceImpl.cpp
    CancelInversesPatterns.cpp
    ConversionPatterns.cpp
    DecomposeLoweringPatterns.cpp
    DisentangleCNOT.cpp
    DisentangleSWAP.cpp
    GridsynthPatterns.cpp
    IonsDecompositionPatterns.cpp
    LoopBoundaryOptimizationPatterns.cpp
    MergeRotationsPatterns.cpp
    PythonCallbackTest.cpp
    SplitMultipleTapes.cpp
    adjoint_lowering.cpp
    cancel_inverses.cpp
    cp_global_buffers.cpp
    decompose_lowering.cpp
    dynamic_one_shot.cpp
    emit_catalyst_pyface.cpp
    graph_decomposition.cpp
    gridsynth.cpp
    ions_decompositions.cpp
    loop_boundary_optimization.cpp
    merge_global_phase.cpp
    merge_rotation.cpp
    quantum_to_llvm.cpp
    split_non_commuting.cpp
    split_to_single_terms.cpp
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

add_mlir_library(${LIBRARY_NAME} STATIC
    ${SRC}
    LINK_LIBS
    PRIVATE
    ${LIBS}
    $<TARGET_OBJECTS:decompsolver>
    DEPENDS
    ${DEPENDS}
)
target_compile_features(${LIBRARY_NAME} PUBLIC cxx_std_20)
target_include_directories(${LIBRARY_NAME} PUBLIC
                           .
                           ${PROJECT_SOURCE_DIR}/include
                           ${CMAKE_BINARY_DIR}/include
                           DecompGraphSolver/)
