###############################################
# ONNX coprocessor function library           #
###############################################
# libcatalyst_onnx_coprocessor.so: a per-message CoprocessorFn that runs an ONNX model through
# onnxruntime, loaded at run time. Building it needs only onnxruntime's C API header: the one in
# ONNXRUNTIME_INCLUDE_DIR when set, otherwise the pinned release header, fetched once and checked
# against its SHA-256.

set(ONNXRUNTIME_INCLUDE_DIR "$ENV{ONNXRUNTIME_INCLUDE_DIR}" CACHE PATH
    "Directory holding onnxruntime_c_api.h (fetched when empty)")
set(ONNXRUNTIME_C_API_VERSION 1.22.0)

if(ONNXRUNTIME_INCLUDE_DIR)
    set(_ort_include_dir ${ONNXRUNTIME_INCLUDE_DIR})
    if(NOT EXISTS ${_ort_include_dir}/onnxruntime_c_api.h)
        message(FATAL_ERROR "ONNXRUNTIME_INCLUDE_DIR has no onnxruntime_c_api.h: ${_ort_include_dir}")
    endif()
else()
    set(_ort_include_dir ${CMAKE_CURRENT_BINARY_DIR}/include)
    if(NOT EXISTS ${_ort_include_dir}/onnxruntime_c_api.h)
        file(DOWNLOAD
            https://raw.githubusercontent.com/microsoft/onnxruntime/v${ONNXRUNTIME_C_API_VERSION}/include/onnxruntime/core/session/onnxruntime_c_api.h
            ${_ort_include_dir}/onnxruntime_c_api.h
            EXPECTED_HASH SHA256=d683537d0fdc29e977b5520f7f15d87a0ac212ae6d94fa9be8893a52655621ae
            TLS_VERIFY ON)
    endif()
endif()

add_library(catalyst_onnx_coprocessor SHARED OnnxCoprocessor.cpp)
target_include_directories(catalyst_onnx_coprocessor SYSTEM PRIVATE ${_ort_include_dir})
target_link_libraries(catalyst_onnx_coprocessor PRIVATE ${CMAKE_DL_LIBS})
set_target_properties(catalyst_onnx_coprocessor PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    CXX_VISIBILITY_PRESET hidden)
