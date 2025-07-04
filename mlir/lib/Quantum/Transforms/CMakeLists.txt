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


include(ExternalProject)
set(Z3_INSTALL_DIR ${CMAKE_BINARY_DIR}/z3)
set(Z3_LIB_PATH ${Z3_INSTALL_DIR}/lib/libz3.a) 

ExternalProject_Add(
  z3
  GIT_REPOSITORY https://github.com/Z3Prover/z3.git
  GIT_TAG z3-4.15.1
  PREFIX ${CMAKE_BINARY_DIR}/_deps/z3
  INSTALL_DIR ${Z3_INSTALL_DIR}
  CMAKE_ARGS
    -DCMAKE_INSTALL_PREFIX=${Z3_INSTALL_DIR}
    -DCMAKE_BUILD_TYPE=Release
    -DZ3_BUILD_PYTHON_BINDINGS=OFF
    -DBUILD_SHARED_LIBS=OFF
  INSTALL_COMMAND cmake --install .
  BUILD_BYPRODUCTS ${Z3_LIB_PATH}
)

# if(NOT EXISTS ${Z3_LIB_PATH})
#     message(STATUS "Z3 not found at ${Z3_LIB_PATH}, adding ExternalProject")
#     ExternalProject_Add(
#         z3
#         GIT_REPOSITORY https://github.com/Z3Prover/z3.git
#         GIT_TAG z3-4.15.1
#         UPDATE_DISCONNECTED TRUE
#         PREFIX ${CMAKE_BINARY_DIR}/_deps/z3
#         CMAKE_ARGS
#             -DCMAKE_INSTALL_PREFIX=${Z3_INSTALL_DIR}
#             -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
#             -DZ3_BUILD_PYTHON_BINDINGS=OFF
#         BUILD_BYPRODUCTS ${Z3_INSTALL_DIR}/lib/libz3.a
#         INSTALL_DIR ${Z3_INSTALL_DIR}
#     )
# else()
#     message(STATUS "Z3 is already built at ${Z3_LIB_PATH}")
# endif()

add_dependencies(${LIBRARY_NAME} z3)
target_include_directories(${LIBRARY_NAME} PRIVATE ${Z3_INSTALL_DIR}/include)
target_link_libraries(${LIBRARY_NAME} PRIVATE ${Z3_INSTALL_DIR}/lib/libz3.a)
