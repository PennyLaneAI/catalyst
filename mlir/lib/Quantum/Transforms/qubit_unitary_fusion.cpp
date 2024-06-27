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

#define DEBUG_TYPE "qubit-unitary-fusion"

#include <memory>
#include <vector>

#include "llvm/Support/Debug.h"
#include "llvm/Support/Errc.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "Catalyst/IR/CatalystDialect.h"
#include "Quantum/IR/QuantumOps.h"
#include "Quantum/Transforms/Patterns.h"

using namespace llvm;
using namespace mlir;
using namespace catalyst::quantum;

namespace catalyst {
namespace quantum {

#define GEN_PASS_DEF_QubitUnitaryFusionPASS
#include "Quantum/Transforms/Passes.h.inc"

struct QubitUnitaryFusionPass
    : impl::QubitUnitaryFusionPassBase<QubitUnitaryFusionPass> {
    using QubitUnitaryFusionPassBase::QubitUnitaryFusionPassBase;

    void runOnOperation() {
        // Get the current operation being operated on.
        ModuleOp op = getOperation();
        MLIRContext *ctx = &getContext();

        // Define the set of patterns to use.
        RewritePatternSet quantumPatterns(ctx);
        quantumPatterns.add<QubitUnitaryFusion>(ctx);

        // Apply patterns in an iterative and greedy manner.
        if (failed(applyPatternsAndFoldGreedily(op, std::move(quantumPatterns)))) {
            return signalPassFailure();
        }
    }
};

} // namespace quantum

std::unique_ptr<Pass> createQubitUnitaryFusionPass()
{
    return std::make_unique<quantum::QubitUnitaryFusionPass>();
}

} // namespace catalyst
