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

set(LIGHTNING_GIT_TAG "master" CACHE STRING "GIT_TAG value to build Lightning")
FetchContent_Declare(
    pennylane_lightning
    GIT_REPOSITORY https://github.com/PennyLaneAI/pennylane-lightning.git
    GIT_TAG      ${LIGHTNING_GIT_TAG}
)
set(CMAKE_POSITION_INDEPENDENT_CODE ON) # build with -fPIC

if(ENABLE_LIGHTNING AND ENABLE_LIGHTNING_KOKKOS)
    set(PL_BACKEND "lightning_qubit;lightning_kokkos" CACHE STRING "PennyLane Lightning backend")
elseif(ENABLE_LIGHTNING)
    set(PL_BACKEND "lightning_qubit" CACHE STRING "PennyLane Lightning backend")
elseif(ENABLE_LIGHTNING_KOKKOS)
    set(PL_BACKEND "lightning_kokkos" CACHE STRING "PennyLane Lightning backend")
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
if(ENABLE_LIGHTNING_KOKKOS)
    list(APPEND src_files
    lightning_kokkos/LightningKokkosSimulator.cpp
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
