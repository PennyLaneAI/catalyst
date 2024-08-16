set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)

if(RUNTIME_CLANG_TIDY)
    if(NOT DEFINED CLANG_TIDY_BINARY)
        set(CLANG_TIDY_BINARY clang-tidy)
    endif()
    message(STATUS "Using CLANG_TIDY_BINARY=${CLANG_TIDY_BINARY}")
    set(CMAKE_CXX_CLANG_TIDY ${CLANG_TIDY_BINARY};
        -extra-arg=-std=c++20;
        --use-color;
    )
endif()

if(LIGHTNING_SRC_PATH)
    if(NOT IS_ABSOLUTE ${LIGHTNING_SRC_PATH})
        message(FATAL_ERROR " LIGHTNING_SRC_PATH=${LIGHTNING_SRC_PATH} must be set to an absolute path")
    endif()
    if(LIGHTNING_GIT_TAG)
        message(WARN " Setting `LIGHTNING_SRC_PATH=${LIGHTNING_SRC_PATH}` overrides `LIGHTNING_GIT_TAG=${LIGHTNING_GIT_TAG}`")
    endif()

    FetchContent_Declare(
        pennylane_lightning
        URL file://${LIGHTNING_SRC_PATH}
    )

    # Acquire local git hash and use for LIGHTNING_GIT_TAG
    execute_process(COMMAND git rev-parse --short HEAD
        WORKING_DIRECTORY ${LIGHTNING_SRC_PATH}
        OUTPUT_VARIABLE LIGHTNING_GIT_TAG
    )
    message(INFO " Building against Lightning commit ${LIGHTNING_GIT_TAG}")

else()
    if(NOT LIGHTNING_GIT_TAG)
        set(LIGHTNING_GIT_TAG "master" CACHE STRING "GIT_TAG value to build Lightning")
    endif()
    FetchContent_Declare(
        pennylane_lightning
        GIT_REPOSITORY https://github.com/PennyLaneAI/pennylane-lightning.git
        GIT_TAG      ${LIGHTNING_GIT_TAG}
    )
endif()
set(CMAKE_POSITION_INDEPENDENT_CODE ON) # build with -fPIC

if(ENABLE_LIGHTNING)
    set(PL_BACKEND "lightning_qubit" CACHE STRING "PennyLane Lightning backend")
endif()
set(ENABLE_PYTHON OFF CACHE BOOL "Enable compilation of the Python module")

FetchContent_MakeAvailable(pennylane_lightning)

set(src_files)

if(ENABLE_LIGHTNING)
    list(APPEND src_files
        lightning_dynamic/StateVectorLQubitDynamic.cpp
        lightning_dynamic/LightningSimulator.cpp
        )
endif()

add_library(rtd_lightning SHARED ${src_files})
add_dependencies(rtd_lightning pennylane_lightning)

target_include_directories(rtd_lightning PRIVATE .
    ${runtime_includes}
    ${backend_includes}
    )

target_link_libraries(rtd_lightning PRIVATE pennylane_lightning)

set_property(TARGET rtd_lightning PROPERTY POSITION_INDEPENDENT_CODE ON)
