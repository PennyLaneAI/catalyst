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

#include "DynamicLibraryLoader.hpp"
#include "Exception.hpp"

namespace Catalyst::Runtime::Device::PLTape {

/**
 * @brief Runs serialized PennyLane tapes against a Python PennyLane device
 *        by calling into the companion nanobind shared library.
 *
 * This follows the exact same pattern as `BraketRunner`: the runner dlopen's
 * the companion `pennylane_python_module.so` shared library at each measurement
 * call, calls an extern "C" function, which acquires the Python GIL and executes
 * the tape on the specified PennyLane device.
 *
 * The shared library name is injected at compile time via the PLPYTHON_PY macro
 * (see CMakeLists.txt).
 */
struct PLPythonRunner {
    explicit PLPythonRunner() = default;
    ~PLPythonRunner() = default;

    /**
     * @brief Execute tape and return probabilities on `num_qubits` wires.
     */
    [[nodiscard]] auto Probs(const std::string &tape_json, const std::string &obs_json,
                             const std::string &device_kwargs, size_t shots,
                             size_t num_qubits) const -> std::vector<double>
    {
        DynamicLibraryLoader libLoader(PLPYTHON_PY);

        using fn_t = void (*)(const char *, const char *, const char *, size_t, size_t, void *);
        auto impl = libLoader.getSymbol<fn_t>("pl_probs");

        std::vector<double> result;
        impl(tape_json.c_str(), obs_json.c_str(), device_kwargs.c_str(), shots, num_qubits,
             &result);
        return result;
    }

    /**
     * @brief Execute tape and return partial probabilities on specified wires.
     */
    [[nodiscard]] auto PartialProbs(const std::string &tape_json, const std::string &obs_json,
                                    const std::string &device_kwargs, size_t shots,
                                    const std::vector<size_t> &wires) const -> std::vector<double>
    {
        DynamicLibraryLoader libLoader(PLPYTHON_PY);

        using fn_t =
            void (*)(const char *, const char *, const char *, size_t, const size_t *, size_t,
                     void *);
        auto impl = libLoader.getSymbol<fn_t>("pl_partial_probs");

        std::vector<double> result;
        impl(tape_json.c_str(), obs_json.c_str(), device_kwargs.c_str(), shots, wires.data(),
             wires.size(), &result);
        return result;
    }

    /**
     * @brief Execute tape and return samples (shots x num_qubits flat array).
     */
    [[nodiscard]] auto Sample(const std::string &tape_json, const std::string &obs_json,
                              const std::string &device_kwargs, size_t shots,
                              size_t num_qubits) const -> std::vector<size_t>
    {
        DynamicLibraryLoader libLoader(PLPYTHON_PY);

        using fn_t = void (*)(const char *, const char *, const char *, size_t, size_t, void *);
        auto impl = libLoader.getSymbol<fn_t>("pl_sample");

        std::vector<size_t> result;
        impl(tape_json.c_str(), obs_json.c_str(), device_kwargs.c_str(), shots, num_qubits,
             &result);
        return result;
    }

    /**
     * @brief Execute tape and return expectation value of observable at `obs_idx`.
     */
    [[nodiscard]] auto Expval(const std::string &tape_json, const std::string &obs_json,
                              const std::string &device_kwargs, size_t shots,
                              size_t obs_idx) const -> double
    {
        DynamicLibraryLoader libLoader(PLPYTHON_PY);

        using fn_t = double (*)(const char *, const char *, const char *, size_t, size_t);
        auto impl = libLoader.getSymbol<fn_t>("pl_expval");

        return impl(tape_json.c_str(), obs_json.c_str(), device_kwargs.c_str(), shots, obs_idx);
    }

    /**
     * @brief Execute tape and return variance of observable at `obs_idx`.
     */
    [[nodiscard]] auto Var(const std::string &tape_json, const std::string &obs_json,
                           const std::string &device_kwargs, size_t shots,
                           size_t obs_idx) const -> double
    {
        DynamicLibraryLoader libLoader(PLPYTHON_PY);

        using fn_t = double (*)(const char *, const char *, const char *, size_t, size_t);
        auto impl = libLoader.getSymbol<fn_t>("pl_var");

        return impl(tape_json.c_str(), obs_json.c_str(), device_kwargs.c_str(), shots, obs_idx);
    }

    /**
     * @brief Execute tape and return statevector.
     */
    [[nodiscard]] auto State(const std::string &tape_json, const std::string &obs_json,
                             const std::string &device_kwargs, size_t shots,
                             size_t num_qubits) const -> std::vector<std::complex<double>>
    {
        DynamicLibraryLoader libLoader(PLPYTHON_PY);

        using fn_t = void (*)(const char *, const char *, const char *, size_t, size_t, void *);
        auto impl = libLoader.getSymbol<fn_t>("pl_state");

        std::vector<std::complex<double>> result;
        impl(tape_json.c_str(), obs_json.c_str(), device_kwargs.c_str(), shots, num_qubits,
             &result);
        return result;
    }
};

} // namespace Catalyst::Runtime::Device::PLTape
