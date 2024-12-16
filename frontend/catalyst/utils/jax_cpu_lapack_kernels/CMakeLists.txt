project(catalyst_jax_cpu_lapack_kernels)

# Create a static library for LAPACK kernels to be included into the
# custom_calls library in the parent scope.

add_library(catalyst_jax_cpu_lapack_kernels STATIC
    ${PROJECT_SOURCE_DIR}/lapack_kernels.cpp
    ${PROJECT_SOURCE_DIR}/lapack_kernels_using_lapack.cpp
)

# Ensure fPIC is set for static builds
set_property(TARGET catalyst_jax_cpu_lapack_kernels PROPERTY POSITION_INDEPENDENT_CODE ON)

# Headers are tracked into the target from the current project scope 
target_include_directories(catalyst_jax_cpu_lapack_kernels PUBLIC
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>
  $<INSTALL_INTERFACE:include> 
)

# If building on MacOS avoid the linker complaining about undefined symbols
if(APPLE)
target_link_options(catalyst_jax_cpu_lapack_kernels PUBLIC -undefined dynamic_lookup)
endif()
