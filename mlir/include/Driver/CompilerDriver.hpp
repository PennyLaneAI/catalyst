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

#include <string>
#include <unordered_map>
#include <vector>

#include "llvm/ADT/StringRef.h"     // llvm::StringRef
#include "llvm/IR/Module.h"         // llvm::Module
#include "llvm/Support/SourceMgr.h" // llvm::SMDiagnostic
#include "llvm/Support/raw_ostream.h"

#include "mlir/IR/BuiltinOps.h"         // mlir::ModuleOp
#include "mlir/IR/DialectRegistry.h"    // mlir::DialectRegistry
#include "mlir/IR/MLIRContext.h"        // mlir::MLIRContext
#include "mlir/IR/OwningOpRef.h"        // mlir::OwningOpRef
#include "mlir/Pass/PassManager.h"      // mlir::PassManager
#include "mlir/Support/LogicalResult.h" // llvm::LogicalResult
#include "mlir/Support/Timing.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

#include "Driver/Pipelines.h"
#include "Driver/Timer.hpp"

namespace catalyst::driver {

/// Verbosity level
// TODO: Adjust the number of levels according to our needs. MLIR seems to print few really
// low-level messages, we might want to hide these.
enum class Verbosity { Silent = 0, Urgent = 1, Debug = 2, All = 3 };

/**
 * @brief Controls the stage for dumping the IR.
 *
 */
enum SaveTemps { None, AfterPipeline, AfterPassChanged, AfterPass };

/**
 * @brief Defines the different functional stages of the driver.
 *
 */
enum Action { OPT, Translate, LLC, All };

/**
 * @brief Defines the format of the input data.
 *
 */
enum InputType { MLIR, LLVMIR, OTHER };

/**
 * @brief Provides enum-defined error codes for the compiler driver.
 *
 */
enum class ErrorCode : int { Success = 0, Failure = 1 };

/// Helper verbose reporting macro.
#define CO_MSG(opt, level, op)                                                                     \
    do {                                                                                           \
        if ((opt).verbosity >= (level)) {                                                          \
            (opt).diagnosticStream << op;                                                          \
        }                                                                                          \
    } while (0)

/**
 * @brief Optional parameters for the compiler, for which we provide reasonable default values.
 *
 */
struct CompilerOptions {
    /// The textual IR (MLIR or LLVM IR)
    mlir::StringRef source;
    /// The directory to place outputs (object file and intermediate results)
    mlir::StringRef workspace;
    /// The name of the module to compile. This is usually the same as the Python function.
    mlir::StringRef moduleName;
    /// The stream to output any error messages from MLIR/LLVM passes and translation.
    llvm::raw_ostream &diagnosticStream;
    /// If specified, the driver will output the IR after each pipeline or each pass.
    SaveTemps keepIntermediate;
    /// If true, the compiler will dump the module scope when saving intermediate files.
    bool dumpModuleScope;
    /// Print SSA IDs using their name location, if provided, as prefix.
    bool useNameLocAsPrefix;
    /// If true, the llvm.coroutine will be lowered.
    bool asyncQnodes;
    /// Sets the verbosity level to use when printing messages.
    Verbosity verbosity;
    /// Ordered list of named pipelines to execute, each pipeline is described by a list of MLIR
    /// passes it includes.
    std::vector<Pipeline> pipelinesCfg;
    /// Specify that the compiler should start after reaching the given pass.
    std::string checkpointStage;
    /// Specify the lowering action to perform
    Action loweringAction;
    /// If true, the compiler will dump the pass pipeline that will be run.
    bool dumpPassPipeline;

    /// Get the destination of the object file at the end of compilation.
    std::string getObjectFile() const;
};

/**
 * @brief Holds the output from the compiler, providing access to pass stage data during
 * compilation.
 *
 */
struct CompilerOutput {
    using PipelineOutputs = std::unordered_map<std::string, std::string>;
    std::string outputFilename;
    std::string outIR;
    std::string diagnosticMessages;
    PipelineOutputs pipelineOutputs;
    size_t globalPipelineCounter = 0; // Counter for root-level pipeline summary files
    size_t passCounter = 0;           // Counter for passes within a pipeline folder
    std::string currentStage = ".";   // Current compilation stage subdirectory
    /// if the compiler reach the pass specified by startAfterPass.
    bool isCheckpointFound;

    // Gets the next pass dump file name within a pipeline folder
    std::string nextPassDumpFilename(const std::string &pipelineName,
                                     const std::string &ext = ".mlir");

    // Gets the root-level pipeline summary file name
    std::string nextPipelineSummaryFilename(const std::string &pipelineName,
                                            const std::string &ext = ".mlir");

    // Set the current compilation stage for organizing output files
    void setStage(const std::string &stageName);
};

/**
 * @brief Parse an MLIR module given in textual ASM representation. Any errors during parsing will
 * be output to diagnosticStream.
 *
 */
mlir::OwningOpRef<mlir::ModuleOp> parseMLIRSource(mlir::MLIRContext *ctx,
                                                  const llvm::SourceMgr &sourceMgr);

/**
 * @brief Checks if the program contains gradient operations in the input MLIR module. Used to
 * identify validity of the program with given passes.
 *
 * @param moduleOp
 * @return true Gradient operations are present in the program.
 * @return false Gradient operations are not present in the program.
 */
bool containsGradients(mlir::ModuleOp moduleOp);

/**
 * @brief Parse an LLVM module given in textual representation. Any parse errors will be output to
 * the provided SMDiagnostic.
 *
 * @param context
 * @param source
 * @param moduleName
 * @param err
 * @return std::shared_ptr<llvm::Module>
 */
std::shared_ptr<llvm::Module> parseLLVMSource(llvm::LLVMContext &context, llvm::StringRef source,
                                              llvm::StringRef moduleName, llvm::SMDiagnostic &err);

/**
 * @brief Register all dialects required by the Catalyst compiler to the given MLIR dialect
 * registry.
 *
 * @param registry Reference to the given MLIR dialect registry. Will be modified in-place with the
 * defined dialects.
 */
void registerAllCatalystDialects(mlir::DialectRegistry &registry);

/**
 * @brief Determines if the compilation stage should be executed if a checkpointStage is provided to
 * the compiler options. This will ensure the compiler will execute only after reaching the given
 * checkpoint.
 *
 * @param options Compiler configuration options.
 * @param output Compiler output object. Modified in-place if checkpoint is not found to indicate if
 * the current stage matches the provided stage name.
 * @param stageName The name of the compiler stage to treat as a checkpoint.
 * @return true Indicates the compiler should run the given stage.
 * @return false Indicates the compiler should not run the given stage.
 */
bool shouldRunStage(const CompilerOptions &options, CompilerOutput &output,
                    const std::string &stageName);

/**
 * @brief Run LLVM passes defined for asynchronous QNode execution.
 * @details These passes are applied when making use of asynchronous QNodes when
 * `qjit(async_qnodes=True), allowing for independent execution of multiple QNodes defined in a
 * given program (where supported). This stage makes use of LLVM coroutine-specific passes, and
 * provides the most benefit when having multiple QNodes in the program that can be independently
 * executed.
 *
 * @param options Compiler configuration options.
 * @param llvmModule
 * @param output
 * @return llvm::LogicalResult
 */
llvm::LogicalResult runCoroLLVMPasses(const CompilerOptions &options,
                                      std::shared_ptr<llvm::Module> llvmModule,
                                      CompilerOutput &output);

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/**
 * @brief Parse input pipelines to return a compatible compilation pipeline stage.
 *
 * @param catalystPipeline List of the pipeline stages in textual format
 * @return std::vector<catalyst::driver::Pipeline> Output pipeline stage usable by the compiler.
 */
std::vector<catalyst::driver::Pipeline>
parsePipelines(const llvm::cl::list<std::string> &catalystPipeline);

/**
 * @brief Find the position of a closing parenthesis character ")" for a given input string
 * reference.
 *
 * @param str Input string reference.
 * @param openParenPos Index position of the opening "(" for the input string.
 * @return size_t Index position of the closing parenthesis character in the string. Returns `npos`
 * if not found.
 */
size_t findMatchingClosingParen(llvm::StringRef str, size_t openParenPos);

/**
 * @brief Ensure the compiler options and input data type match their semantic use.
 *
 * @param options Catalyst compiler options.
 * @param inType Denote the expected input type.
 * @return llvm::LogicalResult
 */
llvm::LogicalResult verifyInputType(const CompilerOptions &options, InputType inType);

/**
 * @brief Apply all MLIR lowering passes on the given module.
 *
 * @param options Compiler configuration options.
 * @param ctx MLIR context for all operations.
 * @param moduleOp Top-level MLIR module container.
 * @param output Compiler output object. Modified in-place.
 * @param timing Timer object for instrumentation of pass execution.
 * @return llvm::LogicalResult
 */
llvm::LogicalResult runLowering(const CompilerOptions &options, mlir::MLIRContext *ctx,
                                mlir::ModuleOp moduleOp, CompilerOutput &output,
                                mlir::TimingScope &timing);

/**
 * @brief Run the given compiler pipeline...
 *
 * @param pm
 * @param options Compiler configuration options.
 * @param output
 * @param pipeline
 * @param clHasManualPipeline
 * @param moduleOp
 * @return llvm::LogicalResult
 */
llvm::LogicalResult runPipeline(mlir::PassManager &pm, const CompilerOptions &options,
                                CompilerOutput &output, Pipeline &pipeline,
                                bool clHasManualPipeline, mlir::ModuleOp moduleOp);

/**
 * @brief
 *
 * @param pm
 * @param options Compiler configuration options.
 * @param pipeline
 * @param clHasManualPipeline
 * @return llvm::LogicalResult
 */
llvm::LogicalResult configurePipeline(mlir::PassManager &pm, const CompilerOptions &options,
                                      Pipeline &pipeline, bool clHasManualPipeline);

/**
 * @brief
 *
 * @param pm
 * @param options Compiler configuration options.
 * @param output
 * @param timer
 * @param timing
 * @return llvm::LogicalResult
 */
llvm::LogicalResult preparePassManager(mlir::PassManager &pm, const CompilerOptions &options,
                                       CompilerOutput &output, catalyst::utils::Timer<> &timer,
                                       mlir::TimingScope &timing);

/**
 * @brief
 *
 * @param filename
 * @return std::string
 */
std::string readInputFile(const std::string &filename);

/**
 * @brief
 *
 * @param options Compiler configuration options.
 * @param llvmModule
 * @param output
 * @return llvm::LogicalResult
 */
llvm::LogicalResult runEnzymePasses(const CompilerOptions &options,
                                    std::shared_ptr<llvm::Module> llvmModule,
                                    CompilerOutput &output);

/**
 * @brief Run optimization passes at the -O2 level on the program representation.
 *
 * @param options Compiler configuration options.
 * @param llvmModule
 * @param output
 * @return llvm::LogicalResult
 */
llvm::LogicalResult runO2LLVMPasses(const CompilerOptions &options,
                                    std::shared_ptr<llvm::Module> llvmModule,
                                    CompilerOutput &output);

}; // namespace catalyst::driver

/**
 * @brief Entry point to the MLIR portion of the compiler.
 *
 * @param options
 * @param output
 * @param registry
 * @return mlir::LogicalResult
 */
mlir::LogicalResult QuantumDriverMain(const catalyst::driver::CompilerOptions &options,
                                      catalyst::driver::CompilerOutput &output,
                                      mlir::DialectRegistry &registry);

/**
 * @brief Entry point to the MLIR portion of the compiler using the command-line interface.
 *
 * @param argc `main` function argc
 * @param argv `main` function argv
 * @return int Return code of execution.
 */
int QuantumDriverMainFromCL(int argc, char **argv);