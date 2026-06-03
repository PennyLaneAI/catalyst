# Catch2-based unit tests for the DecompGraphSolver library.
#
# The DGSolver tests use Catch2 (not Google Test) because the library is
# a standalone C++ component with no MLIR/LLVM dependencies. Keeping Catch2
# avoids pulling in the full MLIR test infrastructure for a pure-algorithmic
# library.

include(FetchContent)

FetchContent_Declare(
Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG        v3.11.0
)

FetchContent_MakeAvailable(Catch2)

target_compile_options(Catch2 PRIVATE -Wno-error -w)

# Required for catch_discover_tests().
list(APPEND CMAKE_MODULE_PATH ${catch2_SOURCE_DIR}/extras)
include(CTest)
include(Catch)

add_executable(runner_tests_dgsolver
    Test_DecompGraphCore.cpp
    Test_DecompGraphSolver.cpp
)

target_link_libraries(runner_tests_dgsolver PRIVATE
    Catch2WithMain
    decompsolver
)

target_include_directories(runner_tests_dgsolver PRIVATE
    ${CMAKE_SOURCE_DIR}/lib/Quantum/Transforms/DecompGraphSolver
)

catch_discover_tests(runner_tests_dgsolver)
