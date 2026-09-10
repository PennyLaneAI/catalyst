// Copyright 2026 Xanadu Quantum Technologies Inc.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#define DEBUG_TYPE "remove-global-phases"

#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/Support/Debug.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "Quantum/Transforms/Passes.h"


using namespace mlir;
using namespace llvm;

namespace catalyst {
namespace quantum {

#define GEN_PASS_DEF_DEVICEBASEDDECOMPOSITIONPASS
#include "Quantum/Transforms/Passes.h.inc"
    
struct DeviceBasedDecompositionPass : public impl::DeviceBasedDecompositionPassBase<DeviceBasedDecompositionPass> {
    using impl::DeviceBasedDecompositionPassBase<DeviceBasedDecompositionPass>::DeviceBasedDecompositionPassBase;

    void runOnOperation() final {
        LLVM_DEBUG(dbgs() << "DeviceBasedDecompositionPass\n");

        // This pass runs AdjointLowering -> CtrlLowering -> GraphDecomposition
        // The options for this pass are handled in the frontend, and match the requirements
        // as per the target device toml file.

        // Run the AdjointLoweringPass
        OpPassManager adjointPM("builtin.module");
        adjointPM.addPass(createAdjointLoweringPass());
        if (failed(runPipeline(adjointPM, getOperation()))) {
            return signalPassFailure();
        }

        // Run the CtrlLoweringPass
        OpPassManager ctrlPM("builtin.module");
        ctrlPM.addPass(createCtrlLoweringPass());
        if (failed(runPipeline(ctrlPM, getOperation()))) {
            return signalPassFailure();
        }

        // Populate the options for the GraphDecompositionPass

        // DeviceBasedDecompositionPass's options are the same as GraphDecompositionPassOptions
        // Copy all the options
        GraphDecompositionPassOptions GDOptions;

        for(auto& targetGate : targetGateSetOption) {
            GDOptions.targetGateSetOption.push_back(targetGate);
        }

        for(auto& fixedDecomp : fixedDecompsOption) {
            GDOptions.fixedDecompsOption.push_back(fixedDecomp);
        }

        for(auto& altDecomp : altDecompsOption) {
            GDOptions.altDecompsOption.push_back(altDecomp);
        }

        GDOptions.bytecodeRulesFile = bytecodeRulesFile;
        GDOptions.libQPDPath = libQPDPath;
        GDOptions.libpythonPath = libpythonPath;

        // Run the GraphDecompositionPass
        OpPassManager gdPM("builtin.module");
        gdPM.addPass(createGraphDecompositionPass(GDOptions));
        if (failed(runPipeline(gdPM, getOperation()))) {
            return signalPassFailure();
        }
    }
};
    
} // namespace quantum
} // namespace catalyst
