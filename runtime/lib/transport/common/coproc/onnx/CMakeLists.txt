###############################################
# ONNX coprocessor function library           #
###############################################
# libcatalyst_onnx_coprocessor.so: a per-message CoprocessorFn that runs an ONNX model through
# onnxruntime, loaded at run time. Building it needs only onnxruntime's C API header: the copy of
# onnxruntime_c_api.h v1.22.0 in vendor/, or the one in ONNXRUNTIME_INCLUDE_DIR when set.

option(CATALYST_TRANSPORT_ONNX "Build the ONNX coprocessor function library" ON)
if(NOT CATALYST_TRANSPORT_ONNX)
    return()
endif()

set(ONNXRUNTIME_INCLUDE_DIR "$ENV{ONNXRUNTIME_INCLUDE_DIR}" CACHE PATH
    "Directory holding onnxruntime_c_api.h (the vendored copy when empty)")

if(ONNXRUNTIME_INCLUDE_DIR)
    set(_ort_include_dir ${ONNXRUNTIME_INCLUDE_DIR})
    if(NOT EXISTS ${_ort_include_dir}/onnxruntime_c_api.h)
        message(FATAL_ERROR "ONNXRUNTIME_INCLUDE_DIR has no onnxruntime_c_api.h: ${_ort_include_dir}")
    endif()
else()
    set(_ort_include_dir ${CMAKE_CURRENT_SOURCE_DIR}/vendor)
endif()

add_library(catalyst_onnx_coprocessor SHARED OnnxCoprocessor.cpp)
target_include_directories(catalyst_onnx_coprocessor SYSTEM PRIVATE ${_ort_include_dir})
target_link_libraries(catalyst_onnx_coprocessor PRIVATE catalyst_transport_common ${CMAKE_DL_LIBS})
set_target_properties(catalyst_onnx_coprocessor PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    CXX_VISIBILITY_PRESET hidden)
