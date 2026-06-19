set(LIBRARY_NAME mitigation-transforms)

file(GLOB SRC
    MitigationMethods/*.cpp
    mitigation_lowering.cpp
    LoweringPatterns.cpp
)

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
set(LIBS
    ${dialect_libs}
    ${conversion_libs}
    MLIRMitigation
)

set(DEPENDS
    MLIRMitigationPassIncGen
)

# MitigationMethods/Rem.cpp still uses the legacy `rewriter.create<OpTy>(...)`
# builder; upstream MLIR has deprecated it in favour of `OpTy::create(rewriter, ...)`.
# COMPILE_FLAGS rather than target_compile_options so the suppression also
# applies to the OBJECT library that `add_mlir_library` creates internally.
# Migration to the new builder is tracked separately.
set_source_files_properties(
    MitigationMethods/Rem.cpp
    PROPERTIES COMPILE_FLAGS "-Wno-deprecated-declarations"
)

add_mlir_library(${LIBRARY_NAME} STATIC ${SRC} LINK_LIBS PRIVATE ${LIBS} DEPENDS ${DEPENDS})
target_compile_features(${LIBRARY_NAME} PUBLIC cxx_std_20)
target_include_directories(${LIBRARY_NAME} PUBLIC
                           .
                           ${PROJECT_SOURCE_DIR}/include
                           ${CMAKE_BINARY_DIR}/include)
