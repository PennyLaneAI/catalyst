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

if(ENABLE_LIGHTNING)
list(APPEND source_files lightning/LightningSimulator.cpp)
endif()

if(ENABLE_LIGHTNING_KOKKOS)
    list(APPEND source_files lightning-kokkos/LightningKokkosSimulator.cpp)
endif()

if(ENABLE_OPENQASM)
    list(APPEND source_files openqasm/OpenQasmDevice.cpp)
endif()

add_library(rt_backend SHARED ${source_files})

target_include_directories(rt_backend PRIVATE   ${backend_includes}
                                                ${runtime_includes} 
                                                ${extensions_includes})

target_link_libraries(rt_backend PUBLIC ${CMAKE_DL_LIBS} lightning_interfaces)

set_property(TARGET rt_backend PROPERTY POSITION_INDEPENDENT_CODE ON)
