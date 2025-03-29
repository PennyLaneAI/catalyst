// Copyright 2022-2025 Xanadu Quantum Technologies Inc.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

/**
 * @file TestUtils.hpp
 * Helper methods for C++ Runtime Tests
 */
#pragma once

#include <string>

#include "ExecutionContext.hpp"
#include "QuantumDevice.hpp"

#define NO_MODIFIERS ((const Modifiers *)NULL)

inline auto get_dylib_ext() -> std::string
{
#ifdef __linux__
    return ".so";
#elif defined(__APPLE__)
    return ".dylib";
#endif
}

static inline Catalyst::Runtime::QuantumDevice *loadDevice(const std::string &device_name,
                                                           const std::string &filename)
{
    auto init_rtd_dylib = std::make_unique<Catalyst::Runtime::SharedLibraryManager>(filename);
    std::string factory_name{device_name + "Factory"};
    void *f_ptr = init_rtd_dylib->getSymbol(factory_name);

    // LCOV_EXCL_START
    return (f_ptr != nullptr)
               ? reinterpret_cast<decltype(Catalyst::Runtime::GenericDeviceFactory) *>(f_ptr)("")
               : nullptr;
    // LCOV_EXCL_STOP
}
