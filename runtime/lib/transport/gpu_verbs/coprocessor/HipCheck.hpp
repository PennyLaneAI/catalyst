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
#include <stdexcept>

#include <hip/hip_runtime.h>

/**
 * @brief Macro that throws `std::runtime_error` if a HIP call does not return
 * hipSuccess, tagged with the source location of the call.
 */
#define HIP_CHECK(expression, what)                                                                \
    ::catalyst::transport::gpu_verbs::_hip_check((expression), (what), __FILE__, __LINE__, __func__)

namespace catalyst::transport::gpu_verbs {

/**
 * @brief Throws a `std::runtime_error` describing a failed HIP call.
 */
inline void _hip_check(hipError_t err, const char *what, const char *file_name, std::size_t line,
                       const char *function_name) {
    if (err == hipSuccess) {
        return;
    }
    std::stringstream sstream;
    sstream << "[" << file_name << ":" << line << "][Function:" << function_name
            << "] gpu_verbs: " << what << ": " << hipGetErrorString(err);
    throw std::runtime_error(sstream.str());
}

} // namespace catalyst::transport::gpu_verbs
