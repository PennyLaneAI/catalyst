
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

#include <complex>
#include <string>
#include <vector>

#include <dlfcn.h>

#include "Exception.hpp"

#ifdef INITIALIZE_PYTHON
#include <pybind11/embed.h>
#endif

namespace Catalyst::Runtime::Device {

/**
 * The OpenQasm circuit runner interface.
 */
struct OQCRunnerBase {
    explicit OQCRunnerBase() = default;
    virtual ~OQCRunnerBase() = default;

    [[nodiscard]] virtual auto runCircuit([[maybe_unused]] const std::string &circuit,
                                          [[maybe_unused]] const std::string &device,
                                          [[maybe_unused]] size_t shots,
                                          [[maybe_unused]] const std::string &kwargs = "") const
        -> std::string
    {
        RT_FAIL("Not implemented method");
        return {};
    }
    [[nodiscard]] virtual auto
    Probs([[maybe_unused]] const std::string &circuit, [[maybe_unused]] const std::string &device,
          [[maybe_unused]] size_t shots, [[maybe_unused]] size_t num_qubits,
          [[maybe_unused]] const std::string &kwargs = "") const -> std::vector<double>
    {
        RT_FAIL("Not implemented method");
        return {};
    }
    [[nodiscard]] virtual auto
    Sample([[maybe_unused]] const std::string &circuit, [[maybe_unused]] const std::string &device,
           [[maybe_unused]] size_t shots, [[maybe_unused]] size_t num_qubits,
           [[maybe_unused]] const std::string &kwargs = "") const -> std::vector<size_t>
    {
        RT_FAIL("Not implemented method");
        return {};
    }
    [[nodiscard]] virtual auto
    Counts([[maybe_unused]] const std::string &circuit, [[maybe_unused]] const std::string &device,
           [[maybe_unused]] size_t shots, [[maybe_unused]] size_t num_qubits,
           [[maybe_unused]] const std::string &kwargs = "") const -> std::vector<size_t>
    {
        RT_FAIL("Not implemented method");
        return {};
    }
    [[nodiscard]] virtual auto
    Expval([[maybe_unused]] const std::string &circuit, [[maybe_unused]] const std::string &device,
           [[maybe_unused]] size_t shots, [[maybe_unused]] const std::string &kwargs = "") const
        -> double
    {
        RT_FAIL("Not implemented method");
        return {};
    }
    [[nodiscard]] virtual auto Var([[maybe_unused]] const std::string &circuit,
                                   [[maybe_unused]] const std::string &device,
                                   [[maybe_unused]] size_t shots,
                                   [[maybe_unused]] const std::string &kwargs = "") const -> double
    {
        RT_FAIL("Not implemented method");
        return {};
    }
    [[nodiscard]] virtual auto
    State([[maybe_unused]] const std::string &circuit, [[maybe_unused]] const std::string &device,
          [[maybe_unused]] size_t shots, [[maybe_unused]] size_t num_qubits,
          [[maybe_unused]] const std::string &kwargs = "") const
        -> std::vector<std::complex<double>>
    {
        RT_FAIL("Not implemented method");
        return {};
    }
    [[nodiscard]] virtual auto Gradient([[maybe_unused]] const std::string &circuit,
                                        [[maybe_unused]] const std::string &device,
                                        [[maybe_unused]] size_t shots,
                                        [[maybe_unused]] size_t num_qubits,
                                        [[maybe_unused]] const std::string &kwargs = "") const
        -> std::vector<double>
    {
        RT_FAIL("Not implemented method");
        return {};
    }
};

/**
 * The OpenQASM circuit runner to execute an OpenQASM circuit on OQC devices thanks to
 * OQC qcaas client.
 */
struct OQCRunner : public OQCRunnerBase {

    [[nodiscard]] auto Counts(const std::string &circuit, const std::string &device, size_t shots,
                              size_t num_qubits, const std::string &kwargs = "") const
        -> std::vector<size_t>
    {

#ifdef INITIALIZE_PYTHON
        if (!Py_IsInitialized()) {
            pybind11::initialize_interpreter();
        }
#endif

        void *handle = dlopen(OQC_PY, RTLD_LAZY);
        if (!handle) {
            char *err_msg = dlerror();
            RT_FAIL(err_msg);
        }

        std::vector<size_t> (*countsImpl)(const char *, const char *, size_t, const char *);
        using countsImpl_t =
            std::vector<size_t> (*)(const char *, const char *, size_t, const char *);

        countsImpl = reinterpret_cast<countsImpl_t>(dlsym(handle, "counts"));
        if (!countsImpl) {
            char *err_msg = dlerror();
            RT_FAIL(err_msg);
        }

        return countsImpl(circuit.c_str(), device.c_str(), shots, kwargs.c_str());
    }
};

} // namespace Catalyst::Runtime::Device
