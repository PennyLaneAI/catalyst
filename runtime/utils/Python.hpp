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

#include <mutex>

#if __has_include("pybind11/embed.h")
#include <pybind11/embed.h>
#define __build_with_pybind11
#endif

struct PythonInterpreterGuard;
std::mutex &getPythonMutex();

/**
 * A (RAII) class for `pybind11::initialize_interpreter` and `pybind11::finalize_interpreter`.
 *
 * @note This is not copyable or movable and used in C++ tests and the ExecutionContext manager
 * of the runtime to solve the issue with re-initialization of the Python interpreter in `catch2`
 * tests which also enables the runtime to reuse the same interpreter in the scope of the global
 * quantum device unique pointer.
 *
 * @note This is only required for OpenQasmDevice and when CAPI is built with pybind11.
 */
#ifdef __build_with_pybind11
// LCOV_EXCL_START
struct PythonInterpreterGuard {
    // This ensures the guard scope to avoid Interpreter
    // conflicts with runtime calls from the frontend.
    bool _init_by_guard = false;

    PythonInterpreterGuard()
    {
        if (!Py_IsInitialized()) {
            pybind11::initialize_interpreter();
            _init_by_guard = true;
        }
    }
    ~PythonInterpreterGuard()
    {
        if (_init_by_guard) {
            pybind11::finalize_interpreter();
        }
    }

    PythonInterpreterGuard(const PythonInterpreterGuard &other) = delete;
    PythonInterpreterGuard(PythonInterpreterGuard &&other) = delete;
    PythonInterpreterGuard &operator=(const PythonInterpreterGuard &other) = delete;
    PythonInterpreterGuard &operator=(PythonInterpreterGuard &&other) = delete;
};
// LCOV_EXCL_STOP
#else
struct PythonInterpreterGuard {
    PythonInterpreterGuard() = default;
    ~PythonInterpreterGuard() = default;
    PythonInterpreterGuard(const PythonInterpreterGuard &other) = delete;
    PythonInterpreterGuard(PythonInterpreterGuard &&other) = delete;
    PythonInterpreterGuard &operator=(const PythonInterpreterGuard &other) = delete;
    PythonInterpreterGuard &operator=(PythonInterpreterGuard &&other) = delete;
};
#endif
