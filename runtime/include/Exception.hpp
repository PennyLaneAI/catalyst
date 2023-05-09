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

#include <exception>
#include <iostream>

#include <sstream>
#include <string>
#include <type_traits>
#include <utility>

/**
 * @brief Macro that throws `RuntimeException` with given message.
 */
#define RT_FAIL(message) Catalyst::Runtime::_abort((message), __FILE__, __LINE__, __func__)

/**
 * @brief Macro that throws `RuntimeException` if expression evaluates
 * to true.
 */
#define RT_FAIL_IF(expression, message)                                                            \
    if ((expression)) {                                                                            \
        RT_FAIL(message);                                                                          \
    }

/**
 * @brief Macro that throws `RuntimeException` with the given expression
 * and source location if expression evaluates to false.
 */
#define RT_ASSERT(expression) RT_FAIL_IF(!(expression), "Assertion: " #expression)

namespace Catalyst::Runtime {

/**
 * @brief This is the general exception thrown by Catalyst for runtime errors
 * that is derived from `std::exception`.
 */
class RuntimeException : public std::exception {
  private:
    const std::string err_msg;

  public:
    explicit RuntimeException(std::string msg) noexcept
        : err_msg{std::move(msg)} {}        // LCOV_EXCL_LINE
    ~RuntimeException() override = default; // LCOV_EXCL_LINE

    RuntimeException(const RuntimeException &) = default;
    RuntimeException(RuntimeException &&) noexcept = default;

    RuntimeException &operator=(const RuntimeException &) = delete;
    RuntimeException &operator=(RuntimeException &&) = delete;

    [[nodiscard]] auto what() const noexcept -> const char * override
    {
        return err_msg.c_str();
    } // LCOV_EXCL_LINE
};

/**
 * @brief Throws a `RuntimeException` with the given error message.
 *
 * @note This is not supposed to be called directly.
 */
[[noreturn]] inline void _abort(const char *message, const char *file_name, size_t line,
                                const char *function_name)
{
    std::stringstream sstream;
    sstream << "[" << file_name << "][Line:" << line << "][Function:" << function_name
            << "] Error in Catalyst Runtime: " << message;

    throw RuntimeException(sstream.str());
} // LCOV_EXCL_LINE

} // namespace Catalyst::Runtime
