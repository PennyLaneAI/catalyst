
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

pybind11_add_module(openqasm_python_module SHARED openqasm_python_module.cpp)

add_dependencies(rtd_openqasm openqasm_python_module)
target_compile_definitions(rtd_openqasm PUBLIC -DOPENQASM_PY=\"$<TARGET_FILE_NAME:openqasm_python_module>\")
set_property(TARGET rtd_openqasm PROPERTY POSITION_INDEPENDENT_CODE ON)
set_property(TARGET rtd_openqasm APPEND PROPERTY BUILD_RPATH "$<TARGET_FILE_DIR:openqasm_python_module>")
