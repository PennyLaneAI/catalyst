set(LIBRARY_NAME remote-transforms)

set(LLVM_LINK_COMPONENTS
    AllTargetsAsmParsers
    AllTargetsCodeGens
)

file(GLOB SRC
    CrossCompileRemoteKernels.cpp
    DispatchRemoteTargets.cpp
    RemoteToLLVM.cpp
)

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
get_property(translation_libs GLOBAL PROPERTY MLIR_TRANSLATION_LIBS)
set(LIBS
    ${dialect_libs}
    ${conversion_libs}
    ${translation_libs}
    MLIRRemote
    MLIRCatalyst
)

set(DEPENDS
    MLIRRemotePassIncGen
)

add_mlir_library(${LIBRARY_NAME} STATIC ${SRC} LINK_LIBS PRIVATE ${LIBS} DEPENDS ${DEPENDS})
target_compile_features(${LIBRARY_NAME} PUBLIC cxx_std_20)
target_include_directories(${LIBRARY_NAME} PUBLIC
                           .
                           ${PROJECT_SOURCE_DIR}/include
                           ${CMAKE_BINARY_DIR}/include)
