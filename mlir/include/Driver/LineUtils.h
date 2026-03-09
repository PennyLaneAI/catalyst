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

#include <string>

namespace catalyst::utils {

/**
 * LinesCount : A utility class to count the number of lines of embedded programs
 * in different compilation stages.
 *
 * You can dump the program-size embedded in an `Operation`, `ModuleOp`, or
 * `llvm::Module` using the static methods in this class.
 *
 * To display results, run the driver with the `ENABLE_DIAGNOSTICS=ON` variable.
 * To store results in YAML format, use `DIAGNOSTICS_RESULTS_PATH=/path/to/file.yml`
 * along with `ENABLE_DIAGNOSTICS=ON`.
 */
class LinesCount {
  public:
    template <class T> static void call(const T &op, const std::string &name = {})
    {
        LinesCount::impl<T>(op, name);
    }

  private:
    /**
     * @brief Convenience templated implementation function. Allows for ease-of-extension to newer
     * types, and preserves the default argument from `call`.
     *
     * @tparam T Operation type. Deduced using CTAD for most uses.
     * @param op Operation argument to print.
     * @param name
     */
    template <class T> static void impl(const T &op, const std::string &name);
};

} // namespace catalyst::utils
