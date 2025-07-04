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
FetchContent_MakeAvailable(z3)

# Now, conditionally apply the patch after FetchContent has populated the source
if(EXISTS "${Z3_PATCH_FILE}")
    message(STATUS "Patch file found: ${Z3_PATCH_FILE}. Applying patch to Z3 source.")
    execute_process(
        COMMAND ${GIT_EXECUTABLE} apply "${Z3_PATCH_FILE}"
        WORKING_DIRECTORY "${z3_SOURCE_DIR}"
        RESULT_VARIABLE PATCH_RESULT
        OUTPUT_VARIABLE PATCH_OUTPUT
        ERROR_VARIABLE PATCH_ERROR
    )

    if(NOT PATCH_RESULT EQUAL 0)
        message(WARNING "Failed to apply Z3 patch. This might be expected if the patch was already 
                            applied or is not compatible. Error: ${PATCH_ERROR}")
    else()
        message(STATUS "Z3 patch applied successfully.")
    endif()
else()
    message(STATUS "Patch file not found: ${Z3_PATCH_FILE}. Skipping Z3 patch application.")
endif()

set(Z3_LIB_ACTUAL_DIR ${CMAKE_BINARY_DIR}/_deps/z3-build/libz3.dylib)
message(STATUS "hello: ${Z3_LIB_ACTUAL_DIR}")

include_directories( ${z3_SOURCE_DIR}/src/api)
# target_link_libraries(${LIBRARY_NAME} PRIVATE libz3)
target_link_libraries(${LIBRARY_NAME} PRIVATE -Wl,-rpath,libz3)
