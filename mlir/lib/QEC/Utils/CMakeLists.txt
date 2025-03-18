set(LLVM_REQUIRES_EH ON)

add_mlir_library(QECUtils
    PauliStringWrapper.cpp
)

include(FetchContent)
FetchContent_Declare(stim
        GIT_REPOSITORY https://github.com/quantumlib/stim.git
        GIT_TAG v1.14.0)
FetchContent_GetProperties(stim)
if(NOT stim_POPULATED)
    FetchContent_Populate(stim)
    add_subdirectory(${stim_SOURCE_DIR} ${CMAKE_BINARY_DIR}/_deps/stim-build)
endif()

target_link_libraries(QECUtils PRIVATE libstim)
