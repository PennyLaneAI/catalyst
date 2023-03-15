set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)

if(ENABLE_LIGHTNING_KOKKOS)
    set(CMAKE_CXX_EXTENSIONS OFF)

    set(LIGHTNING_GIT_TAG "main" CACHE STRING "GIT_TAG value to build Lightning-Kokkos")

    FetchContent_Declare(
        pennylane_lightning_kokkos
        GIT_REPOSITORY https://github.com/PennyLaneAI/pennylane-lightning-kokkos.git
        GIT_TAG        ${LIGHTNING_GIT_TAG}
    )
    set(Kokkos_ENABLE_OPENMP OFF CACHE BOOL "Build Lightning-Kokkos using OpenMP kernels")
    set(Kokkos_ENABLE_THREADS OFF CACHE BOOL "Build Lightning-Kokkos using Threads kernels")
    FetchContent_MakeAvailable(pennylane_lightning_kokkos)

    add_library(lightning_dynamic INTERFACE)

    target_link_libraries(lightning_dynamic INTERFACE
                                lightning_kokkos_algorithms
                                lightning_kokkos_simulator
                                lightning_kokkos_utils)
else()
    set(ENABLE_OPENMP ON)
    set(source_files "StateVectorDynamicCPU.cpp")

    set(LIGHTNING_GIT_TAG "master" CACHE STRING "GIT_TAG value to build Lightning")

    FetchContent_Declare(
        pennylane_lightning
        GIT_REPOSITORY https://github.com/PennyLaneAI/pennylane-lightning.git
        GIT_TAG      ${LIGHTNING_GIT_TAG}
    )
    set(CMAKE_POSITION_INDEPENDENT_CODE ON) # build with -fPIC
    set(ENABLE_KOKKOS OFF CACHE BOOL "Enable Kokkos")

    FetchContent_MakeAvailable(pennylane_lightning)

    add_library(lightning_dynamic INTERFACE ${source_files})

    target_link_libraries(lightning_dynamic INTERFACE
                                lightning_algorithms
                                lightning_gates
                                lightning_simulator
                                lightning_utils)
endif()

target_include_directories(lightning_dynamic INTERFACE .)
