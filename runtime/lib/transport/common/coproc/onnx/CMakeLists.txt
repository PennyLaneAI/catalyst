###############################################
# ONNX coprocessor function library           #
###############################################
# libcatalyst_onnx_coprocessor.so: a per-message CoprocessorFn that runs an ONNX model through
# onnxruntime, loaded at run time. Building it needs only onnxruntime's C API header: the one in
# ONNXRUNTIME_INCLUDE_DIR when set, otherwise the pinned release header, downloaded once and checked
# against its SHA-256. When the download is not possible (an offline build), the library is skipped
# with a warning. A downloaded header whose checksum does not match is an error.

option(CATALYST_TRANSPORT_ONNX "Build the ONNX coprocessor function library" ON)
if(NOT CATALYST_TRANSPORT_ONNX)
    return()
endif()

set(ONNXRUNTIME_INCLUDE_DIR "$ENV{ONNXRUNTIME_INCLUDE_DIR}" CACHE PATH
    "Directory holding onnxruntime_c_api.h (downloaded when empty)")
set(ONNXRUNTIME_C_API_VERSION 1.22.0)
set(ONNXRUNTIME_C_API_SHA256 d683537d0fdc29e977b5520f7f15d87a0ac212ae6d94fa9be8893a52655621ae)

if(ONNXRUNTIME_INCLUDE_DIR)
    set(_ort_include_dir ${ONNXRUNTIME_INCLUDE_DIR})
    if(NOT EXISTS ${_ort_include_dir}/onnxruntime_c_api.h)
        message(FATAL_ERROR "ONNXRUNTIME_INCLUDE_DIR has no onnxruntime_c_api.h: ${_ort_include_dir}")
    endif()
else()
    set(_ort_include_dir ${CMAKE_CURRENT_BINARY_DIR}/include)
    set(_ort_header ${_ort_include_dir}/onnxruntime_c_api.h)
    if(NOT EXISTS ${_ort_header})
        set(_ort_download ${CMAKE_CURRENT_BINARY_DIR}/onnxruntime_c_api.h.download)
        file(DOWNLOAD
            https://raw.githubusercontent.com/microsoft/onnxruntime/v${ONNXRUNTIME_C_API_VERSION}/include/onnxruntime/core/session/onnxruntime_c_api.h
            ${_ort_download}
            STATUS _ort_status
            TLS_VERIFY ON)
        list(GET _ort_status 0 _ort_code)
        if(NOT _ort_code EQUAL 0)
            file(REMOVE ${_ort_download})
            message(WARNING "Skipping the ONNX coprocessor library: could not download "
                "onnxruntime_c_api.h (${_ort_status}). Set ONNXRUNTIME_INCLUDE_DIR to build it.")
            return()
        endif()
        file(SHA256 ${_ort_download} _ort_sha256)
        if(NOT _ort_sha256 STREQUAL ONNXRUNTIME_C_API_SHA256)
            file(REMOVE ${_ort_download})
            message(FATAL_ERROR "onnxruntime_c_api.h v${ONNXRUNTIME_C_API_VERSION} has SHA-256 "
                "${_ort_sha256}, expected ${ONNXRUNTIME_C_API_SHA256}")
        endif()
        file(MAKE_DIRECTORY ${_ort_include_dir})
        file(RENAME ${_ort_download} ${_ort_header})
    endif()
endif()

add_library(catalyst_onnx_coprocessor SHARED OnnxCoprocessor.cpp)
target_include_directories(catalyst_onnx_coprocessor SYSTEM PRIVATE ${_ort_include_dir})
target_link_libraries(catalyst_onnx_coprocessor PRIVATE catalyst_transport_common ${CMAKE_DL_LIBS})
set_target_properties(catalyst_onnx_coprocessor PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    CXX_VISIBILITY_PRESET hidden)
