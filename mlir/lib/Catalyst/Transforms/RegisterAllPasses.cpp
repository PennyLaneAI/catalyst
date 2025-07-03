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

#include "Catalyst/Transforms/Passes.h"
#include "Gradient/Transforms/Passes.h"
#include "Ion/Transforms/Passes.h"
#include "MBQC/Transforms/Passes.h"
#include "Mitigation/Transforms/Passes.h"
#include "QEC/Transforms/Passes.h"
#include "Quantum/Transforms/Passes.h"
#include "Test/Transforms/Passes.h"
#include "mlir/Pass/PassRegistry.h"

void catalyst::registerAllCatalystPasses()
{
    mlir::registerPass(catalyst::createAddExceptionHandlingPass);
    mlir::registerPass(catalyst::createAdjointLoweringPass);
    mlir::registerPass(catalyst::createAnnotateFunctionPass);
    mlir::registerPass(catalyst::createApplyTransformSequencePass);
    mlir::registerPass(catalyst::createArrayListToMemRefPass);
    mlir::registerPass(catalyst::createBufferDeallocationPass);
    mlir::registerPass(catalyst::createCatalystConversionPass);
    mlir::registerPass(catalyst::createCopyGlobalMemRefPass);
    mlir::registerPass(catalyst::createCommutePPRPass);
    mlir::registerPass(catalyst::createCliffordTToPPRPass);
    mlir::registerPass(catalyst::createMergePPRIntoPPMPass);
    mlir::registerPass(catalyst::createPPMCompilationPass);
    mlir::registerPass(catalyst::createDecomposeNonCliffordPPRPass);
    mlir::registerPass(catalyst::createDecomposeCliffordPPRPass);
    mlir::registerPass(catalyst::createCountPPMSpecsPass);
    mlir::registerPass(catalyst::createDetensorizeSCFPass);
    mlir::registerPass(catalyst::createDisableAssertionPass);
    mlir::registerPass(catalyst::createDisentangleCNOTPass);
    mlir::registerPass(catalyst::createDisentangleSWAPPass);
    mlir::registerPass(catalyst::createEmitCatalystPyInterfacePass);
    mlir::registerPass(catalyst::createGEPInboundsPass);
    mlir::registerPass(catalyst::createGradientConversionPass);
    mlir::registerPass(catalyst::createGradientPreprocessingPass);
    mlir::registerPass(catalyst::createGradientPostprocessingPass);
    mlir::registerPass(catalyst::createGradientLoweringPass);
    mlir::registerPass(catalyst::createHloCustomCallLoweringPass);
    mlir::registerPass(catalyst::createInlineNestedModulePass);
    mlir::registerPass(catalyst::createIonConversionPass);
    mlir::registerPass(catalyst::createMemrefCopyToLinalgCopyPass);
    mlir::registerPass(catalyst::createMemrefToLLVMWithTBAAPass);
    mlir::registerPass(catalyst::createMitigationLoweringPass);
    mlir::registerPass(catalyst::createQnodeToAsyncLoweringPass);
    mlir::registerPass(catalyst::createQuantumConversionPass);
    mlir::registerPass(catalyst::createRegisterInactiveCallbackPass);
    mlir::registerPass(catalyst::createRemoveChainedSelfInversePass);
    mlir::registerPass(catalyst::createMergeRotationsPass);
    mlir::registerPass(catalyst::createScatterLoweringPass);
    mlir::registerPass(catalyst::createSplitMultipleTapesPass);
    mlir::registerPass(catalyst::createTestPass);
    mlir::registerPass(catalyst::createIonsDecompositionPass);
    mlir::registerPass(catalyst::createGatesToPulsesPass);
    mlir::registerPass(catalyst::createLoopBoundaryOptimizationPass);
    mlir::registerPass(catalyst::createMBQCConversionPass);
}
