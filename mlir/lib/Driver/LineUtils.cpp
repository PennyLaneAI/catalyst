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

#include <cassert>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/PassInstrumentation.h"
#include "llvm/IR/Module.h"

#include "Driver/LineUtils.hpp"

namespace {

/**
 * @brief Determine if the ENABLE_DIAGNOSTICS environment variable has been set. Valid true value is
 * `ON`, with all others as false.
 */
[[nodiscard]] inline static bool is_diagnostics_enabled()
{
    char *value = getenv("ENABLE_DIAGNOSTICS");
    return value && std::string(value) == "ON";
}

/**
 * @brief Outputs the program line count to `std::err`.
 *
 * @param opStrBuf String to use for line counting.
 * @param name Input to indicate previous operation name. Used only for diagnostics, and ignored if
 * empty.
 */
inline static void print(const std::string &opStrBuf, const std::string &name)
{
    const auto num_lines = std::count(opStrBuf.cbegin(), opStrBuf.cend(), '\n');
    if (!name.empty()) {
        std::cerr << "[DIAGNOSTICS] After " << std::setw(25) << std::left << name;
    }
    std::cerr << "\t" << std::fixed << "programsize: " << num_lines << std::fixed << " lines\n";
}

/**
 * @brief
 *
 * @param opStrBuf
 * @param name
 * @param file_path
 */
inline static void store(const std::string &opStrBuf, const std::string &name,
                         const std::filesystem::path &file_path)
{
    const auto num_lines = std::count(opStrBuf.cbegin(), opStrBuf.cend(), '\n');

    const std::string_view key_padding = "          ";
    const std::string_view val_padding = "              ";

    if (!std::filesystem::exists(file_path)) {
        std::ofstream ofile(file_path);
        assert(ofile.is_open() && "Invalid file to store timer results");
        if (!name.empty()) {
            ofile << key_padding << "- " << name << ":\n";
        }
        ofile << val_padding << "programsize: " << num_lines << "\n";
        ofile.close();
        return;
    }

    // Second, update the file
    std::ofstream ofile(file_path, std::ios::app);
    assert(ofile.is_open() && "Invalid file to store timer results");
    if (!name.empty()) {
        ofile << key_padding << "- " << name << ":\n";
    }
    ofile << val_padding << "programsize: " << num_lines << "\n";
    ofile.close();
}

/**
 * @brief
 *
 * @param opStrBuf
 * @param name
 */
inline static void dump(const std::string &opStrBuf, const std::string &name = {})
{
    char *file = getenv("DIAGNOSTICS_RESULTS_PATH");
    if (!file) {
        print(opStrBuf, name);
        return;
    }
    // else
    store(opStrBuf, name, std::filesystem::path{file});
}

} // namespace

namespace catalyst::utils {

/**
 * @brief Specialisation of `LinesCount` for an `llvm::Module` argument.
 *
 * @tparam const llvm::Module&
 * @param llvmModule
 * @param name
 */
template <>
void LinesCount::impl<llvm::Module>(const llvm::Module &llvmModule, const std::string &name)
{
    if (!is_diagnostics_enabled()) {
        return;
    }

    std::string modStrBef;
    llvm::raw_string_ostream rawStrBef{modStrBef};
    llvmModule.print(rawStrBef, nullptr);

    dump(modStrBef, name);
}

/**
 * @brief Specialisation of `LinesCount` for an `mlir::ModuleOp` argument.
 *
 * @tparam const mlir::ModuleOp&
 * @param op
 * @param name
 */
template <> void LinesCount::impl<mlir::ModuleOp>(const mlir::ModuleOp &op, const std::string &name)
{
    if (!is_diagnostics_enabled()) {
        return;
    }

    std::string modStrBef;
    llvm::raw_string_ostream rawStrBef{modStrBef};
    op->print(rawStrBef);

    dump(modStrBef, name);
}

/**
 * @brief Specialisation of `LinesCount` for an `mlir::Operation` argument.
 *
 * @tparam mlir::Operation *
 * @param op
 * @param name
 */
template <>
void LinesCount::impl<mlir::Operation>(const mlir::Operation &op, const std::string &name)
{
    if (!is_diagnostics_enabled()) {
        return;
    }

    std::string opStrBuf;
    llvm::raw_string_ostream rawStrBef{opStrBuf};
    rawStrBef << op;

    dump(opStrBuf, name);
}

} // namespace catalyst::utils
