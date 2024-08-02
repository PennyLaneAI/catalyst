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
#include <unordered_map>
#include <vector>

#include "mlir/IR/MLIRContext.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"

namespace catalyst {
namespace driver {

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

/// Verbosity level
// TODO: Adjust the number of levels according to our needs. MLIR seems to print few really
// low-level messages, we might want to hide these.
enum class Verbosity { Silent = 0, Urgent = 1, Debug = 2, All = 3 };

/// Helper verbose reporting macro.
#define CO_MSG(opt, level, op)                                                                     \
    do {                                                                                           \
        if ((opt).verbosity >= (level)) {                                                          \
            (opt).diagnosticStream << op;                                                          \
        }                                                                                          \
    } while (0)

/// Pipeline descriptor
struct Pipeline {
    using Name = std::string;
    using PassList = llvm::SmallVector<std::string>;
    Name name;
    PassList passes;
};

/// Optional parameters, for which we provide reasonable default values.
struct CompilerOptions {
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
    /// If true, the llvm.coroutine will be lowered.
    bool asyncQnodes;
    /// Sets the verbosity level to use when printing messages.
    Verbosity verbosity;
    /// Ordered list of named pipelines to execute, each pipeline is described by a list of MLIR
    /// passes it includes.
    std::vector<Pipeline> pipelinesCfg;
    /// Whether to assume that the pipelines output is a valid LLVM dialect and lower it to LLVM IR
    bool lowerToLLVM;
    /// Specify last entry point.
    std::string startingPoint;

    /// Get the destination of the object file at the end of compilation.
    std::string getObjectFile() const
    {
        using path = std::filesystem::path;
        return path(workspace.str()) / path(moduleName.str()).replace_extension(".o");
    }
};

struct CompilerOutput {
    typedef std::unordered_map<Pipeline::Name, std::string> PipelineOutputs;
    std::string objectFilename;
    std::string outIR;
    std::string diagnosticMessages;
    FunctionAttributes inferredAttributes;
    PipelineOutputs pipelineOutputs;
    size_t pipelineCounter = 0;
    /// if last entry point is reached.
    std::string reachStartingPoint;

    // Gets the next pipeline dump file name, prefixed with number.
    std::string nextPipelineDumpFilename(Pipeline::Name pipelineName, std::string ext = ".mlir")
    {
        return std::filesystem::path(std::to_string(this->pipelineCounter++) + "_" + pipelineName)
            .replace_extension(ext);
    };
};

}; // namespace driver
}; // namespace catalyst

/// Entry point to the MLIR portion of the compiler.
mlir::LogicalResult QuantumDriverMain(const catalyst::driver::CompilerOptions &options,
                                      catalyst::driver::CompilerOutput &output);

namespace llvm {

inline raw_ostream &operator<<(raw_ostream &oss, const catalyst::driver::Pipeline &p)
{
    oss << "Pipeline('" << p.name << "', [";
    bool first = true;
    for (const auto &i : p.passes) {
        oss << (first ? "" : ", ") << i;
        first = false;
    }
    oss << "])";
    return oss;
}

}; // namespace llvm
