set(LIBRARY_NAME quantum-transforms)

file(GLOB SRC
    BufferizableOpInterfaceImpl.cpp
    ConversionPatterns.cpp
    quantum_to_llvm.cpp
    emit_catalyst_pyface.cpp
    cp_global_buffers.cpp
    adjoint_lowering.cpp
    annotate_function.cpp
    AdjointPatterns.cpp
    ChainedSelfInversePatterns.cpp
    remove_chained_self_inverse.cpp
    SplitMultipleTapes.cpp
    merge_rotation.cpp
    MergeRotationsPatterns.cpp
    DisentangleSWAP.cpp
    DisentangleCNOT.cpp
    ions_decompositions.cpp
    IonsDecompositionPatterns.cpp
    loop_boundary_optimization.cpp
    LoopBoundaryOptimizationPatterns.cpp
)

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
set(LIBS
    ${dialect_libs}
    ${conversion_libs}
    MLIRQuantum
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
