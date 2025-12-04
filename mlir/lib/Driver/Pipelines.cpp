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
    pm.addPass(catalyst::quantum::createSplitMultipleTapesPass());
    pm.addNestedPass<ModuleOp>(catalyst::createApplyTransformSequencePass());
    pm.addPass(catalyst::createInlineNestedModulePass());
    pm.addPass(catalyst::mitigation::createMitigationLoweringPass());
    pm.addPass(catalyst::quantum::createAdjointLoweringPass());
    pm.addPass(catalyst::createDisableAssertionPass());
}
void createHloLoweringStage(OpPassManager &pm)
{
    pm.addPass(mlir::createCanonicalizerPass());
    pm.addNestedPass<mlir::func::FuncOp>(stablehlo::createChloLegalizeToStablehloPass());
    pm.addNestedPass<mlir::func::FuncOp>(
        catalyst::hlo_extensions::createStablehloLegalizeControlFlowPass());
    stablehlo::StablehloAggressiveSimplificationPassOptions ASoptions;
    pm.addNestedPass<mlir::func::FuncOp>(
        stablehlo::createStablehloAggressiveSimplificationPass(ASoptions));
    pm.addNestedPass<mlir::func::FuncOp>(stablehlo::createStablehloLegalizeToLinalgPass());
    pm.addNestedPass<mlir::func::FuncOp>(
        catalyst::hlo_extensions::createStablehloLegalizeToStandardPass());
    pm.addNestedPass<mlir::func::FuncOp>(
        catalyst::hlo_extensions::createStablehloLegalizeSortPass());
    pm.addPass(stablehlo::createStablehloConvertToSignlessPass());
    pm.addPass(mlir::createCanonicalizerPass());
    pm.addPass(catalyst::hlo_extensions::createScatterLoweringPass());
    pm.addPass(catalyst::hlo_extensions::createHloCustomCallLoweringPass());
    pm.addPass(mlir::createCSEPass());
    mlir::LinalgDetensorizePassOptions LDoptions;
    LDoptions.aggressiveMode = true;
    pm.addNestedPass<mlir::func::FuncOp>(mlir::createLinalgDetensorizePass(LDoptions));
    pm.addPass(catalyst::createDetensorizeSCFPass());
    pm.addPass(catalyst::createDetensorizeFunctionBoundaryPass());
    pm.addPass(mlir::createCanonicalizerPass());
    pm.addPass(mlir::createSymbolDCEPass());
}
void createGradientLoweringStage(OpPassManager &pm)
{
    pm.addPass(catalyst::gradient::createAnnotateInvalidGradientFunctionsPass());
    pm.addPass(catalyst::gradient::createGradientLoweringPass());
}
void createBufferizationStage(OpPassManager &pm)
{
    pm.addPass(mlir::createInlinerPass());
    pm.addPass(mlir::createConvertTensorToLinalgPass());
    pm.addPass(mlir::createConvertElementwiseToLinalgPass());
    pm.addPass(catalyst::gradient::createGradientPreprocessingPass());
    pm.addPass(mlir::bufferization::createEmptyTensorEliminationPass());
    ///////////
    mlir::bufferization::OneShotBufferizePassOptions options;
    options.bufferizeFunctionBoundaries = true;
    options.allowReturnAllocsFromLoops = true;
    options.functionBoundaryTypeConversion =
        mlir::bufferization::LayoutMapOption::IdentityLayoutMap;
    options.unknownTypeConversion = mlir::bufferization::LayoutMapOption::IdentityLayoutMap;
    pm.addPass(mlir::bufferization::createOneShotBufferizePass(options));
    //////////////
    pm.addPass(mlir::createCanonicalizerPass());
    pm.addPass(catalyst::gradient::createGradientPostprocessingPass());
    pm.addNestedPass<mlir::func::FuncOp>(mlir::bufferization::createBufferHoistingPass());
    pm.addNestedPass<mlir::func::FuncOp>(mlir::bufferization::createBufferLoopHoistingPass());
    pm.addNestedPass<mlir::func::FuncOp>(mlir::bufferization::createPromoteBuffersToStackPass());
    // TODO: migrate to new buffer deallocation "buffer-deallocation-pipeline"
    pm.addNestedPass<mlir::func::FuncOp>(catalyst::createBufferDeallocationPass());
    pm.addPass(catalyst::createArrayListToMemRefPass());
    pm.addPass(mlir::createConvertBufferizationToMemRefPass());
    pm.addPass(mlir::createCanonicalizerPass());
    pm.addPass(catalyst::quantum::createCopyGlobalMemRefPass());
}
void createLLVMDialectLoweringStage(OpPassManager &pm)
{
    pm.addPass(mlir::memref::createExpandReallocPass());
    pm.addPass(catalyst::gradient::createGradientConversionPass());
    pm.addPass(catalyst::createMemrefCopyToLinalgCopyPass());
    pm.addNestedPass<mlir::func::FuncOp>(mlir::createConvertLinalgToLoopsPass());
    pm.addPass(mlir::createSCFToControlFlowPass());
    pm.addPass(mlir::memref::createExpandStridedMetadataPass());
    pm.addPass(mlir::createLowerAffinePass());
    pm.addPass(mlir::arith::createArithExpandOpsPass());
    pm.addPass(mlir::createConvertComplexToStandardPass());
    pm.addPass(mlir::createConvertComplexToLLVMPass());
    pm.addPass(mlir::createConvertMathToLLVMPass());
    pm.addPass(mlir::createConvertMathToLibmPass());
    pm.addPass(mlir::createArithToLLVMConversionPass());
    pm.addPass(catalyst::createMemrefToLLVMWithTBAAPass());
    FinalizeMemRefToLLVMConversionPassOptions options;
    options.useGenericFunctions = true;
    pm.addPass(mlir::createFinalizeMemRefToLLVMConversionPass(options));
    pm.addPass(mlir::createConvertIndexToLLVMPass());
    pm.addPass(catalyst::createCatalystConversionPass());
    pm.addPass(catalyst::quantum::createQuantumConversionPass());
    pm.addPass(catalyst::createAddExceptionHandlingPass());
    pm.addPass(catalyst::quantum::createEmitCatalystPyInterfacePass());
    pm.addPass(mlir::createCanonicalizerPass());
    pm.addPass(mlir::createReconcileUnrealizedCastsPass());
    pm.addPass(catalyst::createGEPInboundsPass());
    pm.addPass(catalyst::createRegisterInactiveCallbackPass());
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
