set(LIBRARY_NAME ion-transforms)

file(GLOB SRC
    quantum_to_ion.cpp
    QuantumToIonPatterns.cpp
)

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)




find_package(Python 3
    REQUIRED COMPONENTS Interpreter Development.Module
    OPTIONAL_COMPONENTS Development.SABIModule)

# nanobind suggests including these lines to configure CMake to perform an optimized release build
# by default unless another build type is specified. Without this addition, binding code may run
# slowly and produce large binaries.
# See https://nanobind.readthedocs.io/en/latest/building.html#preliminaries

#if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
#    set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
#    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
#endif()

# Detect the installed nanobind package and import it into CMake
execute_process(
    COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
    OUTPUT_VARIABLE nanobind_ROOT OUTPUT_STRIP_TRAILING_WHITESPACE)

find_package(nanobind CONFIG REQUIRED)


include_directories(${Python3_INCLUDE_DIRS})

set(LIBS
    ${dialect_libs}
    ${conversion_libs}
    MLIRIon
    ${Python3_LIBRARIES}
)

set(DEPENDS
    MLIRIonPassIncGen
)

add_mlir_library(${LIBRARY_NAME} STATIC ${SRC} LINK_LIBS PRIVATE ${LIBS} DEPENDS ${DEPENDS})
target_compile_features(${LIBRARY_NAME} PUBLIC cxx_std_17)
target_include_directories(${LIBRARY_NAME} PUBLIC
                           .
                           ${PROJECT_SOURCE_DIR}/include
                           ${CMAKE_BINARY_DIR}/include
                           ${nanobind_ROOT}/../include
                           ${Python3_INCLUDE_DIRS})
