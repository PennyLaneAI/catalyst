// Copyright 2024 Xanadu Quantum Technologies Inc.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "Driver/Pipelines.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassOptions.h"

#include "Driver/DefaultPipelines/DefaultPipelines.h"
#include "Quantum/Transforms/Passes.h"

#include <fmt/core.h>
#include <fmt/ranges.h>

using namespace mlir;

namespace catalyst {
namespace driver {

void parsePassPipeline(const PassNames &passNames, OpPassManager &pm) {
    std::string passNamesStr = fmt::format("{}", fmt::join(passNames, ","));
    if (failed(mlir::parsePassPipeline(passNamesStr, pm))) {
        llvm::errs() << fmt::format("Error: analysing {}\n", passNames);
    }
}

void createQuantumCompilationStage(OpPassManager &pm) {
    parsePassPipeline(getQuantumCompilationStage(), pm);
}

void createUserTransformStage(OpPassManager &pm) { parsePassPipeline(getUserTransformStage(), pm); }

struct EstimateTransformStageOptions : public PassPipelineOptions<EstimateTransformStageOptions> {
    Option<std::string> outputFname{
        *this, "output-fname",
        llvm::cl::desc("The file to write the levels to. Prints to stdout if empty.")};
    Option<bool> allowUnknownPasses{
        *this, "allow-unknown-passes",
        llvm::cl::desc(
            "Assume that a pass without a resource effect does not change the resources."),
        llvm::cl::init(false)};
};

// The user transform stage, with the transform sequence estimated instead of applied.
void createEstimateTransformStage(OpPassManager &pm, const EstimateTransformStageOptions &options) {
    PassNames passNames = getUserTransformStage();
    assert(passNames.back() == ApplyTransformSequencePassName &&
           "apply-transform-sequence must end the user transform stage");
    passNames.pop_back();
    parsePassPipeline(passNames, pm);

    quantum::EstimateTransformSequencePassOptions passOptions;
    passOptions.outputFname = options.outputFname;
    passOptions.allowUnknownPasses = options.allowUnknownPasses;
    pm.nest<ModuleOp>().addPass(quantum::createEstimateTransformSequencePass(passOptions));
}

void createHLOLoweringStage(OpPassManager &pm) { parsePassPipeline(getHLOLoweringStage(), pm); }

void createGradientLoweringStage(OpPassManager &pm) {
    parsePassPipeline(getGradientLoweringStage(), pm);
}

void createBufferizationStage(OpPassManager &pm) { parsePassPipeline(getBufferizationStage(), pm); }

void createLLVMDialectLoweringStage(OpPassManager &pm) {
    parsePassPipeline(getLLVMDialectLoweringStage(), pm);
}

void createDefaultCatalystPipeline(OpPassManager &pm) {
    createQuantumCompilationStage(pm);
    createHLOLoweringStage(pm);
    createGradientLoweringStage(pm);
    createBufferizationStage(pm);
    createLLVMDialectLoweringStage(pm);
}

void registerQuantumCompilationStage() {
    PassPipelineRegistration<>("quantum-compilation-stage",
                               "Register quantum compilation stage as a pass.",
                               createQuantumCompilationStage);
}

void registerUserTransformStage() {
    PassPipelineRegistration<>("user-transform-stage", "Register user transform stage as a pass.",
                               createUserTransformStage);
}

void registerEstimateTransformStage() {
    PassPipelineRegistration<EstimateTransformStageOptions>(
        "estimate-transform-stage",
        "Register the user transform stage, estimating the resources after each pass of the "
        "transform sequence instead of applying it, as a pass.",
        createEstimateTransformStage);
}

void registerHLOLoweringStage() {
    PassPipelineRegistration<>("hlo-lowering-stage", "Register HLO lowering stage as a pass.",
                               createHLOLoweringStage);
}

void registerGradientLoweringStage() {
    PassPipelineRegistration<>("gradient-lowering-stage",
                               "Register gradient lowering stage as a pass.",
                               createGradientLoweringStage);
}

void registerBufferizationStage() {
    PassPipelineRegistration<>("bufferization-stage", "Register bufferization stage as a pass.",
                               createBufferizationStage);
}

void registerLLVMDialectLoweringStage() {
    PassPipelineRegistration<>("llvm-dialect-lowering-stage",
                               "Register LLVM dialect lowering stage as a pass.",
                               createLLVMDialectLoweringStage);
}

void registerDefaultCatalystPipeline() {
    PassPipelineRegistration<>("default-catalyst-pipeline",
                               "Register full default catalyst pipeline as a pass.",
                               createDefaultCatalystPipeline);
}

void registerAllCatalystPipelines() {
    registerQuantumCompilationStage();
    registerUserTransformStage();
    registerEstimateTransformStage();
    registerHLOLoweringStage();
    registerGradientLoweringStage();
    registerBufferizationStage();
    registerLLVMDialectLoweringStage();
    registerDefaultCatalystPipeline();
}

std::vector<Pipeline> getDefaultPipeline() {
    using PipelineFunc = void (*)(mlir::OpPassManager &);
    std::vector<PipelineFunc> pipelineFuncs = {
        &createQuantumCompilationStage, &createHLOLoweringStage, &createGradientLoweringStage,
        &createBufferizationStage, &createLLVMDialectLoweringStage};

    llvm::SmallVector<std::string> defaultPipelineNames = {
        "QuantumCompilationStage", "HLOLoweringStage", "GradientLoweringStage",
        "BufferizationStage", "MLIRToLLVMDialectConversion"};

    std::vector<Pipeline> defaultPipelines(defaultPipelineNames.size());
    for (size_t i = 0; i < defaultPipelineNames.size(); ++i) {
        defaultPipelines[i].setRegisterFunc(pipelineFuncs[i]);
        defaultPipelines[i].setName(defaultPipelineNames[i]);
        defaultPipelines[i].addPass(defaultPipelineNames[i]);
    }
    return defaultPipelines;
}

} // namespace driver
} // namespace catalyst

namespace llvm {

raw_ostream &operator<<(raw_ostream &oss, const catalyst::driver::Pipeline &p) {
    oss << "Pipeline('" << p.getName() << "', [";
    bool first = true;
    for (const auto &i : p.getPasses()) {
        oss << (first ? "" : ", ") << i;
        first = false;
    }
    oss << "])";
    return oss;
}

}; // namespace llvm
