// Copyright 2026 Xanadu Quantum Technologies Inc.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once
#include <cstddef>
#include <sstream>

#include "Error.hpp"

#include <hip/hip_runtime.h>

#if defined(__HIP_PLATFORM_NVIDIA__)
// For the driver API, which HIP's NVIDIA headers do not wrap. Included after hip_runtime.h, since
// that is what defines the platform macro.
#include <cuda.h>
#endif

/**
 * @brief Macro that throws `TransportError` if a HIP call does not return
 * hipSuccess, tagged with the source location of the call.
 */
#define HIP_CHECK(expression, what)                                                                \
    ::catalyst::transport::coproc::_hip_check((expression), (what), __FILE__, __LINE__, __func__)

#if defined(__HIP_PLATFORM_NVIDIA__)
/**
 * @brief Same, for a CUDA driver call, which returns `CUresult` rather than `hipError_t`.
 */
#define CU_CHECK(expression, what)                                                                 \
    ::catalyst::transport::coproc::_cu_check((expression), (what), __FILE__, __LINE__, __func__)
#endif

namespace catalyst::transport::coproc {

/**
 * @brief Throws a `TransportError` describing a failed HIP call.
 */
inline void _hip_check(hipError_t err, const char *what, const char *file_name, std::size_t line,
                       const char *function_name) {
    if (err == hipSuccess) {
        return;
    }
    std::stringstream sstream;
    sstream << "[" << file_name << ":" << line << "][Function:" << function_name
            << "] coproc: " << what << ": " << hipGetErrorString(err);
    throw ::catalyst::transport::common::TransportError(sstream.str());
}

#if defined(__HIP_PLATFORM_NVIDIA__)
/**
 * @brief Throws a `TransportError` describing a failed CUDA driver call.
 */
inline void _cu_check(CUresult err, const char *what, const char *file_name, std::size_t line,
                      const char *function_name) {
    if (err == CUDA_SUCCESS) {
        return;
    }
    // Both calls leave the pointer untouched on an unrecognized code, hence the defaults.
    const char *name = "CUDA_ERROR_UNKNOWN";
    const char *description = "unknown error";
    (void)cuGetErrorName(err, &name);
    (void)cuGetErrorString(err, &description);
    std::stringstream sstream;
    sstream << "[" << file_name << ":" << line << "][Function:" << function_name
            << "] coproc: " << what << ": " << name << ": " << description;
    throw ::catalyst::transport::common::TransportError(sstream.str());
}
#endif

} // namespace catalyst::transport::coproc
