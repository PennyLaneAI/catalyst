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
    RouteCircuit.cpp
)

#add_compile_options(-Wno-covered-switch-default)
#Disable warning from json library
#if(CMAKE_CXX_COMPILER_ID MATCHES ".*Clang")
set_source_files_properties(
    RouteCircuit.cpp
    COMPILE_FLAGS "-Wno-covered-switch-default"
)
#endif()

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



include(FetchContent)
set(Z3_PATCH_FILE "${CMAKE_CURRENT_LIST_DIR}/z3.patch")

FetchContent_Declare(z3
        GIT_REPOSITORY https://github.com/Z3Prover/z3
        GIT_TAG        z3-4.15.1
        UPDATE_DISCONNECTED 1
)
FetchContent_Populate(z3)
execute_process(
    COMMAND ${GIT_EXECUTABLE} apply --check "${Z3_PATCH_FILE}"
    WORKING_DIRECTORY "${z3_SOURCE_DIR}"
    RESULT_VARIABLE CHECK_PATCH_RESULT
)

target_link_libraries(${LIBRARY_NAME} PRIVATE -Wl,-rpath,libz3)
