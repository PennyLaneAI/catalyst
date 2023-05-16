set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)

set(LIGHTNING_GIT_TAG "master" CACHE STRING "GIT_TAG value to build Lightning")
set(source_files "StateVectorDynamicCPU.cpp")
FetchContent_Declare(
    pennylane_lightning
    GIT_REPOSITORY https://github.com/PennyLaneAI/pennylane-lightning.git
    GIT_TAG      ${LIGHTNING_GIT_TAG}
)
set(CMAKE_POSITION_INDEPENDENT_CODE ON) # build with -fPIC
set(ENABLE_KOKKOS OFF CACHE BOOL "Enable Kokkos")
FetchContent_MakeAvailable(pennylane_lightning)

add_library(lightning_interfaces INTERFACE)

target_link_libraries(lightning_interfaces INTERFACE pennylane_lightning)
target_include_directories(lightning_interfaces INTERFACE .)

set(LIGHTNING_KOKKOS_GIT_TAG "main" CACHE STRING "GIT_TAG value to build Lightning-Kokkos")

if(ENABLE_LIGHTNING_KOKKOS)
    FetchContent_Declare(
        pennylane_lightning_kokkos
        GIT_REPOSITORY https://github.com/PennyLaneAI/pennylane-lightning-kokkos.git
        GIT_TAG        ${LIGHTNING_KOKKOS_GIT_TAG}
    )

    FetchContent_MakeAvailable(pennylane_lightning_kokkos)
    
    target_link_libraries(lightning_interfaces INTERFACE pennylane_lightning_kokkos)
endif()
