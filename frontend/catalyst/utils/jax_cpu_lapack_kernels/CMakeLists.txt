project(catalyst_jax_lapack)

add_library(catalyst_lapack_kernels STATIC
    ${PROJECT_SOURCE_DIR}/lapack_kernels.cpp
    ${PROJECT_SOURCE_DIR}/lapack_kernels_using_lapack.cpp
)

set_property(TARGET catalyst_lapack_kernels PROPERTY POSITION_INDEPENDENT_CODE ON)

target_include_directories(catalyst_lapack_kernels PRIVATE
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>
  $<INSTALL_INTERFACE:include> 
)