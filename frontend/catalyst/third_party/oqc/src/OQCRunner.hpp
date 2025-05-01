
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

#include <string>
#include <vector>

#include "DynamicLibraryLoader.hpp"

namespace Catalyst::Runtime::Device {

/**
 * The OpenQASM circuit runner to execute an OpenQASM circuit on OQC devices thanks to
 * OQC qcaas client.
 */
struct OQCRunner {

    [[nodiscard]] auto Counts(const std::string &circuit, const std::string &device, size_t shots,
                              size_t num_qubits, const std::string &kwargs = "") const
        -> std::vector<size_t>
    {
        DynamicLibraryLoader libLoader(OQC_PY);

        using countsImpl_t =
            std::vector<size_t> (*)(const char *, const char *, size_t, const char *);
        auto countsImpl = libLoader.getSymbol<countsImpl_t>("counts");

        return countsImpl(circuit.c_str(), device.c_str(), shots, kwargs.c_str());
    }
};

} // namespace Catalyst::Runtime::Device
