set(LIBRARY_NAME MLIRCatalystTransforms)

file(GLOB SRC
    ApplyTransformSequencePass.cpp
    ArrayListToMemRefPass.cpp
    AsyncUtils.cpp
    BufferizationPatterns.cpp
    catalyst_bufferize.cpp
    catalyst_to_llvm.cpp
    DetectQNodes.cpp
    DetensorizeSCFPass.cpp
    disable_assertion.cpp
    DisableAssertionPatterns.cpp
    GEPInboundsPass.cpp
    GEPInboundsPatterns.cpp
    hlo_custom_call_lowering.cpp
    HloCustomCallPatterns.cpp
    MemrefCopyToLinalgCopyPass.cpp
    MemrefCopyToLinalgCopyPatterns.cpp
    qnode_to_async_lowering.cpp
    QnodeToAsyncPatterns.cpp
    RegisterAllPasses.cpp
    RegisterInactiveCallbackPass.cpp
    scatter_lowering.cpp
    ScatterPatterns.cpp
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
