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

list(APPEND source_files LightningSimulator.cpp)

if(ENABLE_LIGHTNING_KOKKOS)
    list(APPEND source_files LightningKokkosSimulator.cpp)
endif()

list(APPEND libs lightning_interfaces)

if(ENABLE_OPENQASM)
    find_package(pybind11)

    if(pybind11_FOUND)
        message(STATUS "Found pybind11")
    else()
        message(STATUS "Cound not find existing pybind11 package. Building from source.")
        set(CMAKE_POLICY_DEFAULT_CMP0127 NEW) # To suppress pybind11 CMP0127 warning

        # https://cmake.org/cmake/help/latest/module/FindPython.html
        find_package(Python COMPONENTS Interpreter Development)

        FetchContent_Declare(pybind11
        GIT_REPOSITORY https://github.com/pybind/pybind11.git
        GIT_TAG        v2.10.4
        )

        FetchContent_MakeAvailable(pybind11)
    endif()

    list(APPEND libs pybind11::embed)
    list(APPEND source_files openqasm/OpenQasmDevice.cpp)
endif()

add_library(rt_backend SHARED ${source_files})

target_include_directories(rt_backend PRIVATE   ${backend_includes}
                                                ${runtime_includes} 
                                                ${extensions_includes})

target_link_libraries(rt_backend PUBLIC ${CMAKE_DL_LIBS} ${libs})

set_property(TARGET rt_backend PROPERTY POSITION_INDEPENDENT_CODE ON)
