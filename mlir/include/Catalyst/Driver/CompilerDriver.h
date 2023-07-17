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

#include "mlir/IR/MLIRContext.h"
#include "mlir/Support/LogicalResult.h"
#include <filesystem>

/// Run a given set of passes on an MLIR module.
///
/// The IR is supplied in textual form while the passes are expected in MLIR's command line
/// interface form.
mlir::FailureOr<std::string> RunPassPipeline(mlir::StringRef source, mlir::StringRef passes);

/// Data about the JIT function that is optionally inferred and returned to the caller.
///
/// This is important for calling a function when invoking the compiler on an MLIR or LLVM textual
/// representation intead of from Python.
struct FunctionAttributes {
    /// The name of the primary JIT entry point function.
    std::string functionName;
    /// The return type of the JIT entry point function.
    std::string returnType;
};

struct CompilerOptions {
    mlir::MLIRContext *ctx;
    /// The textual IR (MLIR or LLVM IR)
    mlir::StringRef source;
    /// The directory to place outputs (object file and intermediate results)
    mlir::StringRef workspace;
    /// The name of the module to compile. This is usually the same as the Python function.
    mlir::StringRef moduleName;
    /// The stream to output any error messages from MLIR/LLVM passes and translation.
    llvm::raw_ostream &diagnosticStream;
    /// If true, the driver will output the module at intermediate points.
    bool keepIntermediate;

    /// Get the destination of the object file at the end of compilation.
    std::string getObjectFile() const
    {
        using path = std::filesystem::path;
        return path(workspace.str()) / path(moduleName.str()).replace_extension(".o");
    }
};

/// Entry point to the MLIR portion of the compiler.
mlir::LogicalResult QuantumDriverMain(const CompilerOptions &options,
                                      std::optional<FunctionAttributes> &inferredData);
