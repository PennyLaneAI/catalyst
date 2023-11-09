##################################
# Object Lib catalyst_qir_qis_obj
##################################

add_library(catalyst_qir_qis_obj OBJECT RuntimeCAPI.cpp)

set(dep_libs)

if(ENABLE_LIGHTNING OR ENABLE_LIGHTNING_KOKKOS)
    list(APPEND dep_libs pennylane_lightning -lrtd_lightning)
endif()
if(ENABLE_OPENQASM)
  find_package(pybind11 CONFIG)
  list(APPEND dep_libs pybind11::module -lrtd_openqasm)
endif()

# link to rt_backend
target_link_libraries(catalyst_qir_qis_obj ${CMAKE_DL_LIBS}
    "-L${CMAKE_LIBRARY_OUTPUT_DIRECTORY}"
    ${dep_libs})

if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
  set(LINKER_FLAGS_1, "-Wl,-all_load")
  # We can technically use "-Wl,-noall_load", but the linker
  # raises a warning saying that it is a no-op.
  set(LINKER_FLAGS_2, "")
elseif(${CMAKE_SYSTEM_NAME} MATCHES "Linux")
  set(LINKER_FLAGS_1, "-Wl,--whole-archive")
  set(LINKER_FLAGS_2, "-Wl,--no-whole-archive")
endif()

target_link_libraries(catalyst_qir_qis_obj "-L${QIR_STDLIB_PATH}"
    ${LINKER_FLAGS_1}
    qir_stdlib
    ${LINKER_FLAGS_2}
    pthread
    dl)

target_include_directories(catalyst_qir_qis_obj PUBLIC .
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${QIR_STDLIB_PATH}
    ${runtime_includes}
    ${backend_includes}
    )

set_property(TARGET catalyst_qir_qis_obj PROPERTY POSITION_INDEPENDENT_CODE ON)

#####################
# Shared Lib rt_capi
#####################

add_library(rt_capi SHARED)

target_link_libraries(rt_capi ${CMAKE_DL_LIBS} catalyst_qir_qis_obj)

if(ENABLE_LIGHTNING OR ENABLE_LIGHTNING_KOKKOS)
    add_dependencies(rt_capi rtd_lightning)
endif()
if(ENABLE_OPENQASM)
    add_dependencies(rt_capi rtd_openqasm)
endif()

target_include_directories(rt_capi PUBLIC .
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${QIR_STDLIB_PATH}
    ${runtime_includes}
    ${capi_utils_includes}
    ${backend_includes}
    )

set_property(TARGET rt_capi PROPERTY POSITION_INDEPENDENT_CODE ON)
