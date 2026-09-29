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

#include <cmath>
#include <complex>
#include <iomanip>
#include <limits>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>

#include "DynamicLibraryLoader.hpp"
#include "Exception.hpp"

namespace Catalyst::Runtime::Device::PLPython {

/**
 * @brief A minimal builder for the JSON payloads sent to the Python side of the bridge.
 */
class Payload {
  private:
    std::ostringstream oss;
    bool first = true;

    void key(const std::string &name)
    {
        oss << (first ? "" : ",") << "\"" << name << "\":";
        first = false;
    }

    void number(double value)
    {
        if (std::isnan(value)) {
            oss << "NaN";
        }
        else if (std::isinf(value)) {
            oss << (value > 0 ? "Infinity" : "-Infinity");
        }
        else {
            oss << value;
        }
    }

  public:
    Payload()
    {
        oss << std::setprecision(std::numeric_limits<double>::max_digits10);
        oss << "{";
    }

    auto add(const std::string &name, const std::string &value) -> Payload &
    {
        key(name);
        oss << "\"";
        for (char c : value) {
            if (c == '"' || c == '\\') {
                oss << '\\';
            }
            oss << c;
        }
        oss << "\"";
        return *this;
    }

    auto add(const std::string &name, const char *value) -> Payload &
    {
        return add(name, std::string(value));
    }

    auto add(const std::string &name, double value) -> Payload &
    {
        key(name);
        number(value);
        return *this;
    }

    auto add(const std::string &name, size_t value) -> Payload &
    {
        key(name);
        oss << value;
        return *this;
    }

    auto add(const std::string &name, bool value) -> Payload &
    {
        key(name);
        oss << (value ? "true" : "false");
        return *this;
    }

    template <typename T> auto add(const std::string &name, const std::vector<T> &values) -> Payload &
    {
        key(name);
        oss << "[";
        for (size_t i = 0; i < values.size(); i++) {
            oss << (i ? "," : "");
            if constexpr (std::is_same_v<T, bool>) {
                oss << (values[i] ? "true" : "false");
            }
            else if constexpr (std::is_same_v<T, std::string>) {
                oss << "\"" << values[i] << "\"";
            }
            else if constexpr (std::is_floating_point_v<T>) {
                number(values[i]);
            }
            else {
                oss << values[i];
            }
        }
        oss << "]";
        return *this;
    }

    auto addComplex(const std::string &re, const std::string &im,
                    const std::vector<std::complex<double>> &values) -> Payload &
    {
        std::vector<double> real(values.size());
        std::vector<double> imag(values.size());
        for (size_t i = 0; i < values.size(); i++) {
            real[i] = values[i].real();
            imag[i] = values[i].imag();
        }
        return add(re, real).add(im, imag);
    }

    auto addNull(const std::string &name) -> Payload &
    {
        key(name);
        oss << "null";
        return *this;
    }

    [[nodiscard]] auto str() -> std::string { return oss.str() + "}"; }
};

/**
 * @brief Calls into the Python side of the bridge (``catalyst.device.python_device``).
 *
 * The companion nanobind module (``pennylane_python_module.so``, injected at compile time via the
 * ``PLPYTHON_PY`` macro, see CMakeLists.txt) acquires the GIL and forwards each instruction to
 * ``catalyst.device.python_device.runtime_dispatch``, which records it on a PennyLane tape.
 */
class PLPythonRunner {
  private:
    using dispatch_t = bool (*)(const char *, const char *, void *, void *);
    DynamicLibraryLoader loader{PLPYTHON_PY};
    dispatch_t dispatch = loader.getSymbol<dispatch_t>("pl_dispatch");

  public:
    /**
     * @brief Send an instruction to the Python side and return its flattened results.
     */
    auto Call(const std::string &method, const std::string &payload) const -> std::vector<double>
    {
        std::vector<double> result;
        std::string error;
        if (!dispatch(method.c_str(), payload.c_str(), &result, &error)) {
            RT_FAIL(error.c_str());
        }
        return result;
    }
};

} // namespace Catalyst::Runtime::Device::PLPython
