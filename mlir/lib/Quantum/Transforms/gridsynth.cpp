// Copyright 2025 Xanadu Quantum Technologies Inc.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#define DEBUG_TYPE "gridsynth"

#include "llvm/Support/Debug.h"

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "PBC/IR/PBCDialect.h"
#include "Quantum/Transforms/Patterns.h"

using namespace llvm;
using namespace mlir;

namespace catalyst {
namespace quantum {

#define GEN_PASS_DECL_GRIDSYNTHPASS
#define GEN_PASS_DEF_GRIDSYNTHPASS
#include "Quantum/Transforms/Passes.h.inc"

struct GridsynthPass : impl::GridsynthPassBase<GridsynthPass> {
    using GridsynthPassBase::GridsynthPassBase;

    void runOnOperation() final
    {
        LLVM_DEBUG(llvm::dbgs() << "Running GridsynthPass\n");
        mlir::Operation *module = getOperation();
        mlir::MLIRContext *context = &getContext();
        RewritePatternSet patterns(context);

        populateGridsynthPatterns(patterns, epsilon, pprBasis);

        if (failed(applyPatternsGreedily(module, std::move(patterns)))) {
            signalPassFailure();
        }
    }
};

} // namespace quantum
} // namespace catalyst
