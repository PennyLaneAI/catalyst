if(ENABLE_CLANG_TIDY)
if(NOT DEFINED CLANG_TIDY_BINARY)
    set(CLANG_TIDY_BINARY clang-tidy)
endif()
message(STATUS "Using CLANG_TIDY_BINARY=${CLANG_TIDY_BINARY}")
set(CMAKE_CXX_CLANG_TIDY ${CLANG_TIDY_BINARY};
                         -extra-arg=-std=c++20;
)
endif()

if(ENABLE_LIGHTNING_KOKKOS)
    set(source_files "LightningKokkosSimulator.cpp")
else()
    set(source_files "LightningSimulator.cpp")
endif()

add_library(rt_backend SHARED ${source_files})

target_include_directories(rt_backend PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
                                                ${runtime_includes} 
                                                ${extensions_includes})

target_link_libraries(rt_backend PUBLIC ${CMAKE_DL_LIBS} lightning_dynamic)

set_property(TARGET rt_backend PROPERTY POSITION_INDEPENDENT_CODE ON)
