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

#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/SmallVector.h"
#include <filesystem>
#include <vector>

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
typedef enum {
    CO_VERB_SILENT = 0,
    CO_VERB_URGENT = 1,
    CO_VERB_DEBUG = 2,
    CO_VERB_ALL = 3
} Verbosity;


/// Pipeline descriptor
struct Pipeline {
    typedef std::string Name;
    typedef llvm::SmallVector<std::string> PassList;
    std::string name;
    PassList passes;
};

/// Structure which defines the task for the driver to solve.
struct CompilerSpec {
    /// Ordered list of named pipelines to execute, each pipeline is described by a list of MLIR passes
    /// it includes.
    std::vector< Pipeline > pipelinesCfg;
    bool attemptLLVMLowering;
};

/// Optional parameters, for which we provide reasonable default values.
struct CompilerOptions {
    mlir::MLIRContext *ctx; // TODO: Move to Spec
    /// The textual IR (MLIR or LLVM IR)
    mlir::StringRef source; // TODO: Move to Spec
    /// The directory to place outputs (object file and intermediate results)
    mlir::StringRef workspace; // TODO: Move to Spec
    /// The name of the module to compile. This is usually the same as the Python function.
    mlir::StringRef moduleName; // TODO: Move to Spec
    /// The stream to output any error messages from MLIR/LLVM passes and translation.
    llvm::raw_ostream &diagnosticStream; // TODO: Move to Spec
    /// If true, the driver will output the module at intermediate points.
    bool keepIntermediate;
    /// Sets the verbosity level to use when printing messages.
    Verbosity verbosity;

    /// Get the destination of the object file at the end of compilation.
    /// TODO: Move to Spec
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
    FunctionAttributes inferredAttributes;
    PipelineOutputs pipelineOutputs;
};


/// Entry point to the MLIR portion of the compiler.
mlir::LogicalResult QuantumDriverMain(const CompilerSpec &spec,
                                      const CompilerOptions &options,
                                      CompilerOutput &output);

namespace llvm {

inline raw_ostream &operator<<(raw_ostream &oss, const Pipeline &p)
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

