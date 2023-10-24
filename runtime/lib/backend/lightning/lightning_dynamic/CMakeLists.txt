set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)

set(LIGHTNING_GIT_TAG "master" CACHE STRING "GIT_TAG value to build Lightning")
set(source_files "StateVectorLQubitDynamic.cpp")
FetchContent_Declare(
    pennylane_lightning
    GIT_REPOSITORY https://github.com/PennyLaneAI/pennylane-lightning.git
    GIT_TAG      ${LIGHTNING_GIT_TAG}
)
set(CMAKE_POSITION_INDEPENDENT_CODE ON) # build with -fPIC
set(ENABLE_KOKKOS OFF CACHE BOOL "Enable Kokkos")

if(ENABLE_LIGHTNING AND ENABLE_LIGHTNING_KOKKOS)
    set(PL_BACKEND "lightning_qubit;lightning_kokkos" CACHE STRING "PennyLane Lightning backend")
elseif(ENABLE_LIGHTNING)
    set(PL_BACKEND "lightning_qubit" CACHE STRING "PennyLane Lightning backend")
elseif(ENABLE_LIGHTNING_KOKKOS)
    set(PL_BACKEND "lightning_kokkos" CACHE STRING "PennyLane Lightning backend")
endif()

set(ENABLE_PYTHON OFF CACHE BOOL "Enable compilation of the Python module")
FetchContent_MakeAvailable(pennylane_lightning)

add_library(rt_interfaces INTERFACE)

target_link_libraries(rt_interfaces INTERFACE pennylane_lightning)
target_include_directories(rt_interfaces INTERFACE .)

set(LIGHTNING_KOKKOS_GIT_TAG "main" CACHE STRING "GIT_TAG value to build Lightning-Kokkos")


if(ENABLE_OPENQASM)
    find_package(pybind11 CONFIG)

    if(pybind11_FOUND)
        message(STATUS "Found pybind11")
    else()
        message(STATUS "Cound not find existing pybind11-dev package. Building from source.")
        set(CMAKE_POLICY_DEFAULT_CMP0127 NEW) # To suppress pybind11 CMP0127 warning

        # https://cmake.org/cmake/help/latest/module/FindPython.html
        find_package(Python COMPONENTS Interpreter Development)

        FetchContent_Declare(pybind11
        GIT_REPOSITORY https://github.com/pybind/pybind11.git
        GIT_TAG        v2.10.1
        )

        FetchContent_MakeAvailable(pybind11)
    endif()

    # To avoid link to libpython, we use pybind11::module interface library.
    target_link_libraries(rt_interfaces INTERFACE pybind11::module)

endif()
