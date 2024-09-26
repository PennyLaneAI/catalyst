// Copyright 2023 Xanadu Quantum Technologies Inc.

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
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "Exception.hpp"

#include <pybind11/eval.h>
#include <pybind11/pybind11.h>
#ifdef INITIALIZE_PYTHON
#include <pybind11/embed.h>
#endif

#include <dlfcn.h>

namespace Catalyst::Runtime::Device::OpenQasm {

/**
 * The OpenQasm circuit runner interface.
 */
struct OpenQasmRunner {
    explicit OpenQasmRunner() = default;
    virtual ~OpenQasmRunner() = default;
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
 * The OpenQasm circuit runner to execute an OpenQasm circuit on Braket Devices backed by
 * Amazon Braket Python SDK.
 */
struct BraketRunner : public OpenQasmRunner {
    [[nodiscard]] auto runCircuit(const std::string &circuit, const std::string &device,
                                  size_t shots, const std::string &kwargs = "") const
        -> std::string override
    {
#ifdef INITIALIZE_PYTHON
        if (!Py_IsInitialized()) {
            pybind11::initialize_interpreter();
        }
#endif
        void *handle = dlopen(OPENQASM_PY, RTLD_LAZY);
        if (!handle) {
            char *err_msg = dlerror();
            RT_FAIL(err_msg);
        }

        char *(*runCircuitImpl)(const char *, const char *, size_t, const char *);
        typedef char *(*func_ptr_t)(const char *, const char *, size_t, const char *);

        runCircuitImpl = (func_ptr_t)dlsym(handle, "runCircuit");
        if (!runCircuitImpl) {
            char *err_msg = dlerror();
            RT_FAIL(err_msg);
        }

        char *message = runCircuitImpl(circuit.c_str(), device.c_str(), shots, kwargs.c_str());
        std::string messageStr(message);
        free(message);
        return messageStr;
    }

    [[nodiscard]] auto Probs(const std::string &circuit, const std::string &device, size_t shots,
                             size_t num_qubits, const std::string &kwargs = "") const
        -> std::vector<double> override
    {
#ifdef INITIALIZE_PYTHON
        if (!Py_IsInitialized()) {
            pybind11::initialize_interpreter();
        }
#endif
        std::vector<double> probs;

        void *handle = dlopen(OPENQASM_PY, RTLD_LAZY);
        if (!handle) {
            char *err_msg = dlerror();
            RT_FAIL(err_msg);
        }

        void *(*probsImpl)(const char *, const char *, size_t, size_t, const char *, void *);
        typedef void *(*probsImpl_t)(const char *, const char *, size_t, size_t, const char *,
                                     void *);

        probsImpl = (probsImpl_t)dlsym(handle, "probs");
        if (!probsImpl) {
            char *err_msg = dlerror();
            RT_FAIL(err_msg);
        }

        probsImpl(circuit.c_str(), device.c_str(), shots, num_qubits, kwargs.c_str(), &probs);

        return probs;
    }

    [[nodiscard]] auto Sample(const std::string &circuit, const std::string &device, size_t shots,
                              size_t num_qubits, const std::string &kwargs = "") const
        -> std::vector<size_t> override
    {

#ifdef INITIALIZE_PYTHON
        if (!Py_IsInitialized()) {
            pybind11::initialize_interpreter();
        }
#endif
        std::vector<size_t> samples;

        void *handle = dlopen(OPENQASM_PY, RTLD_LAZY);
        if (!handle) {
            char *err_msg = dlerror();
            RT_FAIL(err_msg);
        }

        void *(*samplesImpl)(const char *, const char *, size_t, size_t, const char *, void *);
        typedef void *(*samplesImpl_t)(const char *, const char *, size_t, size_t, const char *,
                                       void *);

        samplesImpl = (samplesImpl_t)dlsym(handle, "samples");
        if (!samplesImpl) {
            char *err_msg = dlerror();
            RT_FAIL(err_msg);
        }

        samplesImpl(circuit.c_str(), device.c_str(), shots, num_qubits, kwargs.c_str(), &samples);

        return samples;
    }

    [[nodiscard]] auto Expval(const std::string &circuit, const std::string &device, size_t shots,
                              const std::string &kwargs = "") const -> double override
    {
#ifdef INITIALIZE_PYTHON
        if (!Py_IsInitialized()) {
            pybind11::initialize_interpreter();
        }
#endif

        void *handle = dlopen(OPENQASM_PY, RTLD_LAZY);
        if (!handle) {
            char *err_msg = dlerror();
            RT_FAIL(err_msg);
        }

        double (*expvalImpl)(const char *, const char *, size_t, const char *);
        typedef double (*expvalImpl_t)(const char *, const char *, size_t, const char *);

        expvalImpl = (expvalImpl_t)dlsym(handle, "expval");
        if (!expvalImpl) {
            char *err_msg = dlerror();
            RT_FAIL(err_msg);
        }

        return expvalImpl(circuit.c_str(), device.c_str(), shots, kwargs.c_str());
    }

    [[nodiscard]] auto Var(const std::string &circuit, const std::string &device, size_t shots,
                           const std::string &kwargs = "") const -> double override
    {
#ifdef INITIALIZE_PYTHON
        if (!Py_IsInitialized()) {
            pybind11::initialize_interpreter();
        }
#endif

        void *handle = dlopen(OPENQASM_PY, RTLD_LAZY);
        if (!handle) {
            char *err_msg = dlerror();
            RT_FAIL(err_msg);
        }

        double (*varImpl)(const char *, const char *, size_t, const char *);
        typedef double (*varImpl_t)(const char *, const char *, size_t, const char *);

        varImpl = (varImpl_t)dlsym(handle, "expval");
        if (!varImpl) {
            char *err_msg = dlerror();
            RT_FAIL(err_msg);
        }

        return varImpl(circuit.c_str(), device.c_str(), shots, kwargs.c_str());
    }
};

} // namespace Catalyst::Runtime::Device::OpenQasm
