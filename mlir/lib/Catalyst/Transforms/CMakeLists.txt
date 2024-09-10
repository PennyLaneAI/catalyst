set(LIBRARY_NAME MLIRCatalystTransforms)

file(GLOB SRC
    ArrayListToMemRefPass.cpp
    scatter_lowering.cpp
    ScatterPatterns.cpp
    qnode_to_async_lowering.cpp
    QnodeToAsyncPatterns.cpp
    RegisterAllPasses.cpp
    BufferizationPatterns.cpp
    catalyst_bufferize.cpp
    catalyst_to_llvm.cpp
    hlo_custom_call_lowering.cpp
    HloCustomCallPatterns.cpp
    DetectQNodes.cpp
    AsyncUtils.cpp
    GEPInboundsPatterns.cpp
    GEPInboundsPass.cpp
    MemrefCopyToLinalgCopyPatterns.cpp
    MemrefCopyToLinalgCopyPass.cpp
    RegisterInactiveCallbackPass.cpp
    disable_assertion.cpp
    DisableAssertionPatterns.cpp
    ApplyTransformSequencePass.cpp
    SplitMultipleTapes.cpp
)

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
set(LIBS
    ${dialect_libs}
    ${conversion_libs}
)

set(DEPENDS
    MLIRCatalystPassIncGen
)

add_mlir_library(${LIBRARY_NAME} STATIC ${SRC} LINK_LIBS PRIVATE ${LIBS} DEPENDS ${DEPENDS})
target_compile_features(${LIBRARY_NAME} PUBLIC cxx_std_17)
target_include_directories(${LIBRARY_NAME} PUBLIC
                           .
                           ${PROJECT_SOURCE_DIR}/include
                           ${CMAKE_BINARY_DIR}/include)
