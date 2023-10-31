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

#include <filesystem>
#include <string>

#include "mlir/Support/LogicalResult.h"
#include "llvm/Support/raw_ostream.h"

#include "CompilerDriver.h"

namespace catalyst {
namespace driver {

template <typename Obj>
void dumpToFile(const CompilerOptions &options, mlir::StringRef fileName, const Obj &obj)
{
    using std::filesystem::path;
    std::error_code errCode;
    std::string outFileName = path(options.workspace.str()) / path(fileName.str());

    CO_MSG(options, Verbosity::Debug, "Dumping '" << outFileName << "'\n");
    llvm::raw_fd_ostream outfile{outFileName, errCode};
    if (errCode) {
        CO_MSG(options, Verbosity::Urgent, "Unable to open file: " << errCode.message() << "\n");
        return;
    }
    outfile << obj;
    outfile.flush();
    if (errCode) {
        CO_MSG(options, Verbosity::Urgent,
               "Unable to write to file: " << errCode.message() << "\n");
    }
}
} // namespace driver
} // namespace catalyst
