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

#include <memory>

#include <fmt/core.h>
#include <fmt/ranges.h>

#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/BufferizationToMemRef/BufferizationToMemRef.h"
#include "mlir/Conversion/ComplexToLLVM/ComplexToLLVM.h"
#include "mlir/Conversion/ComplexToStandard/ComplexToStandard.h"
#include "mlir/Conversion/IndexToLLVM/IndexToLLVM.h"
#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"
#include "mlir/Conversion/MathToLibm/MathToLibm.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Conversion/TensorToLinalg/TensorToLinalgPass.h"
#include "mlir/Dialect/Arith/Transforms/Passes.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "stablehlo/conversions/linalg/transforms/Passes.h"
#include "stablehlo/transforms/Passes.h"
#include "stablehlo/transforms/optimization/Passes.h"

#include "Catalyst/IR/CatalystDialect.h"
#include "Catalyst/Transforms/Passes.h"
#include "Driver/DefaultPipelines.h"
#include "Driver/Pipelines.h"
#include "Gradient/IR/GradientDialect.h"
#include "Gradient/Transforms/Passes.h"
#include "Mitigation/Transforms/Passes.h"
#include "Quantum/IR/QuantumDialect.h"
#include "Quantum/Transforms/Passes.h"
#include "hlo-extensions/Transforms/Passes.h"

using namespace mlir;

namespace catalyst {
namespace driver {

void createQuantumCompilationStage(OpPassManager &pm)
{
    const auto &passList = getEnforceRuntimeInvariantsPipeline();
    std::string passNames = fmt::format("{}", fmt::join(passList, ","));
    if (failed(mlir::parsePassPipeline(passNames, pm))) {
        llvm::errs() << fmt::format("Error: analysing {}\n", passNames);
    }
}

void createHLOLoweringPipeline(OpPassManager &pm)
{
    const auto &passList = getHLOLoweringPipeline();
    std::string passNames = fmt::format("{}", fmt::join(passList, ","));
    if (failed(mlir::parsePassPipeline(passNames, pm))) {
        llvm::errs() << fmt::format("Error: analysing {}\n", passNames);
    }
}

void createQuantumCompilationPipeline(OpPassManager &pm)
{
    const auto &passList = getQuantumCompilationPipeline();
    std::string passNames = fmt::format("{}", fmt::join(passList, ","));
    if (failed(mlir::parsePassPipeline(passNames, pm))) {
        llvm::errs() << fmt::format("Error: analysing {}\n", passNames);
    }
}

void createBufferizationPipeline(OpPassManager &pm)
{
    const auto &passList = getBufferizationPipeline();
    std::string passNames = fmt::format("{}", fmt::join(passList, ","));
    if (failed(mlir::parsePassPipeline(passNames, pm))) {
        llvm::errs() << fmt::format("Error: analysing {}\n", passNames);
    }
}

void createLLVMDialectLoweringPipeline(OpPassManager &pm)
{
    const auto &passList = getLLVMDialectLoweringPipeline();
    std::string passNames = fmt::format("{}", fmt::join(passList, ","));
    if (failed(mlir::parsePassPipeline(passNames, pm))) {
        llvm::errs() << fmt::format("Error: analysing {}\n", passNames);
    }
}

void createDefaultCatalystPipeline(OpPassManager &pm)
{
    createQuantumCompilationStage(pm);
    createHloLoweringStage(pm);
    createGradientLoweringStage(pm);
    createBufferizationStage(pm);
    createLLVMDialectLoweringStage(pm);
}

void registerQuantumCompilationStage()
{
    PassPipelineRegistration<>("quantum-compilation-stage",
                               "Register quantum compilation stage as a pass.",
                               createQuantumCompilationStage);
}

void registerHloLoweringStage()
{
    PassPipelineRegistration<>("hlo-lowering-stage", "Register HLO lowering stage as a pass.",
                               createHloLoweringStage);
}

void registerGradientLoweringStage()
{
    PassPipelineRegistration<>("gradient-lowering-stage",
                               "Register gradient lowering stage as a pass.",
                               createGradientLoweringStage);
}

void registerBufferizationStage()
{
    PassPipelineRegistration<>("bufferization-stage", "Register bufferization stage as a pass.",
                               createBufferizationStage);
}

void registerLLVMDialectLoweringStage()
{
    PassPipelineRegistration<>("llvm-dialect-lowering-stage",
                               "Register LLVM dialect lowering stage as a pass.",
                               createLLVMDialectLoweringStage);
}

void registerDefaultCatalystPipeline()
{
    PassPipelineRegistration<>("default-catalyst-pipeline",
                               "Register full default catalyst pipeline as a pass.",
                               createDefaultCatalystPipeline);
}

void registerAllCatalystPipelines()
{
    registerQuantumCompilationStage();
    registerHloLoweringStage();
    registerGradientLoweringStage();
    registerBufferizationStage();
    registerLLVMDialectLoweringStage();
    registerDefaultCatalystPipeline();
}

std::vector<Pipeline> getDefaultPipeline()
{
    using PipelineFunc = void (*)(mlir::OpPassManager &);
    std::vector<PipelineFunc> pipelineFuncs = {
        &createQuantumCompilationStage, &createHloLoweringStage, &createGradientLoweringStage,
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
