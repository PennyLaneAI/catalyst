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

#define DEBUG_TYPE "memrefcopytolinalgcopy"

#include "llvm/Support/Debug.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "Catalyst/IR/CatalystDialect.h"
#include "Catalyst/Transforms/Patterns.h"

using namespace llvm;
using namespace mlir;
using namespace catalyst;

namespace catalyst {
#define GEN_PASS_DEF_MEMREFCOPYTOLINALGCOPYPASS
#include "Catalyst/Transforms/Passes.h.inc"

struct MemrefCopyToLinalgCopyPass
    : impl::MemrefCopyToLinalgCopyPassBase<MemrefCopyToLinalgCopyPass> {
    using MemrefCopyToLinalgCopyPassBase::MemrefCopyToLinalgCopyPassBase;

    void runOnOperation() final
    {
        LLVM_DEBUG(dbgs() << "memref.copy  to linalg.copy pass"
                          << "\n");

        RewritePatternSet patterns(&getContext());

        populateMemrefCopyToLinalgCopyPatterns(patterns);
        if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns)))) {
            return signalPassFailure();
        }
    }
};

std::unique_ptr<Pass> createMemrefCopyToLinalgCopyPass()
{
    return std::make_unique<MemrefCopyToLinalgCopyPass>();
}

} // namespace catalyst