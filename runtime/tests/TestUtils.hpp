// Copyright 2022-2023 Xanadu Quantum Technologies Inc.

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
 * Helper methods for C++ Tests
 */
#pragma once

#include <string>
#include <tuple>
#include <vector>

#include "catch2/catch.hpp"

#include "ExecutionContext.hpp"
#include "QuantumDevice.hpp"

#include <iostream>

/// @cond DEV
namespace {
using namespace Catalyst::Runtime;
} // namespace
/// @endcond

/**
 * A tuple of available backend devices to be tested using TEMPLATE_LIST_TEST_CASE in Catch2
 */
#if __has_include("LightningSimulator.hpp")
#include "LightningSimulator.hpp"
using SimTypes = std::tuple<Catalyst::Runtime::Simulator::LightningSimulator>;
#endif

/**
 * Get available device names in the compatible format for `__catalyst__rt__device`
 *
 * This is a utility function used in Catch2 tests.
 *
 * @return `std::vector<std::pair<std::string, std::string>>`
 */
static inline auto getDevices() -> std::vector<std::tuple<std::string, std::string, std::string>>
{
    std::vector<std::tuple<std::string, std::string, std::string>> devices{
        {"lightning.qubit", "lightning.qubit", "{shots: 0}"}};
    return devices;
}

inline auto get_dylib_ext() -> std::string
{
#ifdef __linux__
    return ".so";
#elif defined(__APPLE__)
    return ".dylib";
#endif
}

#define NO_MODIFIERS ((const Modifiers *)NULL)

static inline MemRefT_CplxT_double_1d getState(size_t buffer_len)
{
    CplxT_double *buffer = new CplxT_double[buffer_len];
    MemRefT_CplxT_double_1d result = {buffer, buffer, 0, {buffer_len}, {1}};
    return result;
}

static inline void freeState(MemRefT_CplxT_double_1d &result) { delete[] result.data_allocated; }

static inline QuantumDevice *loadDevice(std::string device_name, std::string filename)
{
    std::unique_ptr<SharedLibraryManager> init_rtd_dylib =
        std::make_unique<SharedLibraryManager>(filename);
    std::string factory_name{device_name + "Factory"};
    void *f_ptr = init_rtd_dylib->getSymbol(factory_name);

    // LCOV_EXCL_START
    return (f_ptr != nullptr) ? reinterpret_cast<decltype(GenericDeviceFactory) *>(f_ptr)("")
                              : nullptr;
    // LCOV_EXCL_STOP
}
