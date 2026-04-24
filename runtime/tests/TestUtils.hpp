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
#include <vector>

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

template <class IntegerType> struct tanner_graph_steane {
    /* Tanner graph representation for the [[7, 1, 3]] Steane code
       The shape of dense matrix that the [[7, 1, 3]] Steane code is (10, 10).
       The first 7 columns represent data qubits, while the last 3 columns
       represent auxillary qubits. The full dense matrix is:
       | 0 0 0 0 0 0 0 1 0 0|
       | 0 0 0 0 0 0 0 1 1 0|
       | 0 0 0 0 0 0 0 1 1 1|
       | 0 0 0 0 0 0 0 1 0 1|
       | 0 0 0 0 0 0 0 0 1 0|
       | 0 0 0 0 0 0 0 0 1 1|
       | 0 0 0 0 0 0 0 0 0 1|
       | 1 1 1 1 0 0 0 0 0 0|
       | 0 1 1 0 1 1 0 0 0 0|
       | 0 0 1 1 0 1 1 0 0 0|
    */
    const size_t code_size = 7;
    const size_t code_distance = 3;
    const std::vector<IntegerType> row_idx = {7, 7, 8, 7, 8, 9, 7, 9, 8, 8, 9, 9,
                                              0, 1, 2, 3, 1, 2, 4, 5, 2, 3, 5, 6};
    const std::vector<IntegerType> col_ptr = {0, 1, 3, 6, 8, 9, 11, 12, 16, 20, 24};

    const std::vector<IntegerType> row_idx_parity_matrix_transpose = {0, 1, 2, 3, 1, 2,
                                                                      4, 5, 2, 3, 5, 6};
    const std::vector<IntegerType> col_ptr_parity_matrix_transpose = {0, 4, 8, 12};

    const std::unordered_map<std::string, std::vector<uint8_t>> lookup_table_syndrome_to_error = {
        {"000", std::vector<uint8_t>({0, 0, 0, 0, 0, 0, 0})},
        {"001", std::vector<uint8_t>({0, 0, 0, 0, 0, 0, 1})},
        {"010", std::vector<uint8_t>({0, 0, 0, 0, 1, 0, 0})},
        {"011", std::vector<uint8_t>({0, 0, 0, 0, 0, 1, 0})},
        {"100", std::vector<uint8_t>({1, 0, 0, 0, 0, 0, 0})},
        {"101", std::vector<uint8_t>({0, 0, 0, 1, 0, 0, 0})},
        {"110", std::vector<uint8_t>({0, 1, 0, 0, 0, 0, 0})},
        {"111", std::vector<uint8_t>({0, 0, 1, 0, 0, 0, 0})},

    };

    const std::unordered_map<int64_t, std::vector<int8_t>> lookup_table_error_idx_to_syndrome = {
        {-1, std::vector<int8_t>({0, 0, 0})}, {6, std::vector<int8_t>({0, 0, 1})},
        {4, std::vector<int8_t>({0, 1, 0})},  {5, std::vector<int8_t>({0, 1, 1})},
        {0, std::vector<int8_t>({1, 0, 0})},  {3, std::vector<int8_t>({1, 0, 1})},
        {1, std::vector<int8_t>({1, 1, 0})},  {2, std::vector<int8_t>({1, 1, 1})},

    };
};
