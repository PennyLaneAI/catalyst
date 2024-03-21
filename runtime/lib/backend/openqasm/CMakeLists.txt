
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

fetch_pybind11()

add_library(rtd_openqasm SHARED OpenQasmDevice.cpp)

target_include_directories(rtd_openqasm PRIVATE .
    ${runtime_includes}
    ${backend_includes}
    )

target_link_libraries(rtd_openqasm pybind11::module)

set_property(TARGET rtd_openqasm PROPERTY POSITION_INDEPENDENT_CODE ON)
