
// Copyright 2024 Xanadu Quantum Technologies Inc.

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

#include <cstring>
#include <string>
#include <vector>

#include "DynamicLibraryLoader.hpp"

namespace Catalyst::Runtime::Device {

/**
 * The OpenQASM circuit runner to execute an OpenQASM circuit on OQC devices thanks to
 * OQC qcaas client.
 */
struct OQCRunner {

    [[nodiscard]] auto Counts(const std::string &circuit, const std::string &qpu_id, size_t shots,
                              size_t num_qubits, const std::string &kwargs = "") const
        -> std::vector<size_t>
    {
        DynamicLibraryLoader libLoader(OQC_PY, RTLD_LAZY | RTLD_GLOBAL);

        using countsImpl_t = int (*)(const char *, const char *, size_t, size_t, const char *,
                                     void *, char *, size_t);
        auto countsImpl = libLoader.getSymbol<countsImpl_t>("counts");

        std::vector<size_t> results;
        char error_msg[256] = {0};

        int result_code = countsImpl(circuit.c_str(), qpu_id.c_str(), shots, num_qubits,
                                     kwargs.c_str(), &results, error_msg, sizeof(error_msg));

        RT_FAIL_IF(result_code, error_msg);

        return results;
    }
};

} // namespace Catalyst::Runtime::Device
