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
#include "Catalyst/Transforms/Passes.h"
#include "Gradient/Transforms/Passes.h"
#include "Mitigation/Transforms/Passes.h"
#include "Quantum/Transforms/Passes.h"
#include "mhlo/transforms/passes.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

using namespace mlir;
namespace catalyst {
namespace driver {

void createEnforceRuntimeInvariantsPipeline(OpPassManager &pm)
{
    pm.addPass(catalyst::createSplitMultipleTapesPass());
    pm.addPass(catalyst::createApplyTransformSequencePass());
    pm.addPass(catalyst::createInlineNestedModulePass());
}
void createHloLoweringPipeline(OpPassManager &pm)
{
    pm.addPass(mlir::createCanonicalizerPass());
    pm.addNestedPass<mlir::func::FuncOp>(mhlo::createChloLegalizeToHloPass());
    pm.addPass(mlir::mhlo::createStablehloLegalizeToHloPass());
    pm.addNestedPass<mlir::func::FuncOp>(mhlo::createLegalizeControlFlowPass());
    pm.addNestedPass<mlir::func::FuncOp>(mhlo::createLegalizeHloToLinalgPass());
    pm.addNestedPass<mlir::func::FuncOp>(mhlo::createLegalizeToStdPass());
    pm.addNestedPass<mlir::func::FuncOp>(mhlo::createLegalizeSortPass());
    pm.addPass(mlir::mhlo::createConvertToSignlessPass());
    pm.addPass(mlir::createCanonicalizerPass());
    pm.addPass(catalyst::createScatterLoweringPass());
    pm.addPass(catalyst::createHloCustomCallLoweringPass());
    pm.addPass(mlir::createCSEPass());
    mlir::LinalgDetensorizePassOptions options;
    options.aggressiveMode = true;
    pm.addNestedPass<mlir::func::FuncOp>(mlir::createLinalgDetensorizePass(options));
    pm.addPass(catalyst::createDetensorizeSCFPass());
    pm.addPass(mlir::createCanonicalizerPass());
}
void createQuantumCompilationPipeline(OpPassManager &pm)
{
    pm.addPass(catalyst::createAnnotateFunctionPass());
    pm.addPass(catalyst::createMitigationLoweringPass());
    pm.addPass(catalyst::createGradientLoweringPass());
    pm.addPass(catalyst::createAdjointLoweringPass());
    pm.addPass(catalyst::createDisableAssertionPass());
}
void createBufferizationPipeline(OpPassManager &pm)
{
    mlir::bufferization::OneShotBufferizationOptions options;
    options.opFilter.allowDialect<mlir::bufferization::BufferizationDialect>();
    pm.addPass(mlir::bufferization::createOneShotBufferizePass(options));
    pm.addPass(mlir::createInlinerPass());
    pm.addPass(catalyst::createGradientPreprocessingPass());
    pm.addPass(catalyst::createGradientBufferizationPass());
    pm.addPass(mlir::createSCFBufferizePass());
    pm.addPass(mlir::createConvertTensorToLinalgPass());
    pm.addPass(mlir::createConvertElementwiseToLinalgPass());
    pm.addPass(mlir::arith::createArithBufferizePass());
    pm.addPass(mlir::bufferization::createEmptyTensorToAllocTensorPass());
    pm.addNestedPass<mlir::func::FuncOp>(mlir::bufferization::createBufferizationBufferizePass());
    pm.addNestedPass<mlir::func::FuncOp>(mlir::tensor::createTensorBufferizePass());
    pm.addPass(catalyst::createCatalystBufferizationPass());
    pm.addNestedPass<mlir::func::FuncOp>(mlir::createLinalgBufferizePass());
    pm.addNestedPass<mlir::func::FuncOp>(mlir::tensor::createTensorBufferizePass());
    pm.addPass(catalyst::createQuantumBufferizationPass());
    pm.addPass(mlir::func::createFuncBufferizePass());
    pm.addNestedPass<mlir::func::FuncOp>(mlir::bufferization::createFinalizingBufferizePass());
    pm.addPass(mlir::createCanonicalizerPass());
    pm.addPass(catalyst::createGradientPostprocessingPass());
    pm.addNestedPass<mlir::func::FuncOp>(mlir::bufferization::createBufferHoistingPass());
    pm.addNestedPass<mlir::func::FuncOp>(mlir::bufferization::createBufferLoopHoistingPass());
    pm.addNestedPass<mlir::func::FuncOp>(mlir::bufferization::createBufferDeallocationPass());
    pm.addPass(catalyst::createArrayListToMemRefPass());
    pm.addPass(mlir::createBufferizationToMemRefPass());
    pm.addPass(mlir::createCanonicalizerPass());
    pm.addPass(catalyst::createCopyGlobalMemRefPass());
}
void createLLVMDialectLoweringPipeline(OpPassManager &pm)
{
    pm.addPass(mlir::memref::createExpandReallocPass());
    pm.addPass(catalyst::createGradientConversionPass());
    pm.addPass(catalyst::createMemrefCopyToLinalgCopyPass());
    pm.addNestedPass<mlir::func::FuncOp>(mlir::createConvertLinalgToLoopsPass());
    pm.addPass(mlir::createConvertSCFToCFPass());
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
    pm.addPass(catalyst::createQuantumConversionPass());
    pm.addPass(catalyst::createAddExceptionHandlingPass());
    pm.addPass(catalyst::createEmitCatalystPyInterfacePass());
    pm.addPass(mlir::createCanonicalizerPass());
    pm.addPass(mlir::createReconcileUnrealizedCastsPass());
    pm.addPass(catalyst::createGEPInboundsPass());
    pm.addPass(catalyst::createRegisterInactiveCallbackPass());
}

void createDefaultCatalystPipeline(OpPassManager &pm)
{
    createEnforceRuntimeInvariantsPipeline(pm);
    createHloLoweringPipeline(pm);
    createQuantumCompilationPipeline(pm);
    createBufferizationPipeline(pm);
    createLLVMDialectLoweringPipeline(pm);
}

void registerEnforceRuntimeInvariantsPipeline()
{
    PassPipelineRegistration<>("enforce-runtime-invariants-pipeline",
                               "Register enforce runtime invariants pipeline as a pass.",
                               createEnforceRuntimeInvariantsPipeline);
}
void registerHloLoweringPipeline()
{
    PassPipelineRegistration<>("hlo_lowering-pipeline", "Register HLO lowering pipeline as a pass.",
                               createHloLoweringPipeline);
}
void registerQuantumCompilationPipeline()
{
    PassPipelineRegistration<>("quantum-compilation-pipeline",
                               "Register quantum compilation pipeline as a pass.",
                               createQuantumCompilationPipeline);
}
void registerBufferizationPipeline()
{
    PassPipelineRegistration<>("bufferization-pipeline",
                               "Register bufferization pipeline as a pass.",
                               createBufferizationPipeline);
}
void registerLLVMDialectLoweringPipeline()
{
    PassPipelineRegistration<>("llvm-dialect-lowring-pipeline",
                               "Register LLVM dialect lowring pipeline as a pass.",
                               createLLVMDialectLoweringPipeline);
}

void registerDefaultCatalystPipeline()
{
    PassPipelineRegistration<>("default-catalyst-pipeline",
                               "Register full default catalyst pipeline as a pass.",
                               createDefaultCatalystPipeline);
}

void registerAllCatalystPipelines()
{
    registerEnforceRuntimeInvariantsPipeline();
    registerHloLoweringPipeline();
    registerQuantumCompilationPipeline();
    registerBufferizationPipeline();
    registerLLVMDialectLoweringPipeline();
    registerDefaultCatalystPipeline();
}

std::vector<Pipeline> getDefaultPipeline()
{
    using PipelineFunc = void (*)(mlir::OpPassManager &);
    std::vector<PipelineFunc> pipelineFuncs = {
        &createEnforceRuntimeInvariantsPipeline, &createHloLoweringPipeline,
        &createQuantumCompilationPipeline, &createBufferizationPipeline,
        &createLLVMDialectLoweringPipeline};

    llvm::SmallVector<std::string> defaultPipelineNames = {
        "enforce-runtime-invariants-pipeline", "hlo-lowering-pipeline",
        "quantum-compilation-pipeline", "bufferization-pipeline", "llvm-dialect-lowering-pipeline"};

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
