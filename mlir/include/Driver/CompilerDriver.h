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
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"

namespace catalyst {
namespace driver {

/// Verbosity level
// TODO: Adjust the number of levels according to our needs. MLIR seems to print few really
// low-level messages, we might want to hide these.
enum class Verbosity { Silent = 0, Urgent = 1, Debug = 2, All = 3 };

enum SaveTemps { None, AfterPipeline, AfterPass };

enum Action { OPT, Translate, LLC, All };

enum InputType { MLIR, LLVMIR, OTHER };

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
    /// If specified, the driver will output the module after each pipeline or each pass.
    SaveTemps keepIntermediate;
    /// If true, the llvm.coroutine will be lowered.
    bool asyncQnodes;
    /// Sets the verbosity level to use when printing messages.
    Verbosity verbosity;
    /// Ordered list of named pipelines to execute, each pipeline is described by a list of MLIR
    /// passes it includes.
    std::vector<Pipeline> pipelinesCfg;
    /// Specify that the compiler should start after reaching the given pass.
    std::string checkpointStage;
    /// Specify the loweting action to perform
    Action loweringAction;
    /// If true, the compiler will dump the pass pipeline that will be run.
    bool dumpPassPipeline;

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
    PipelineOutputs pipelineOutputs;
    size_t pipelineCounter = 0;
    /// if the compiler reach the pass specified by startAfterPass.
    bool isCheckpointFound;

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
                                      catalyst::driver::CompilerOutput &output,
                                      mlir::DialectRegistry &registry);

int QuantumDriverMainFromCL(int argc, char **argv);
int QuantumDriverMainFromArgs(const std::string &source, const std::string &workspace,
                              const std::string &moduleName, bool keepIntermediate,
                              bool asyncQNodes, bool verbose, bool lowerToLLVM,
                              const std::vector<catalyst::driver::Pipeline> &passPipelines,
                              const std::string &checkpointStage,
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
