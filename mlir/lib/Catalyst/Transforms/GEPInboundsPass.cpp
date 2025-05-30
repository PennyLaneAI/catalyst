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

#define DEBUG_TYPE "gepinbounds"

#include "llvm/Support/Debug.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "Catalyst/IR/CatalystDialect.h"
#include "Catalyst/Transforms/Patterns.h"

using namespace llvm;
using namespace mlir;
using namespace catalyst;

namespace catalyst {
#define GEN_PASS_DEF_GEPINBOUNDSPASS
#include "Catalyst/Transforms/Passes.h.inc"

struct GEPInboundsPass : impl::GEPInboundsPassBase<GEPInboundsPass> {
    using GEPInboundsPassBase::GEPInboundsPassBase;

    void runOnOperation() final
    {
        LLVM_DEBUG(dbgs() << "GEP inbounds pass"
                          << "\n");

        RewritePatternSet patterns(&getContext());

        populateGEPInboundsPatterns(patterns);
        if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
            return signalPassFailure();
        }
    }
};

std::unique_ptr<Pass> createGEPInboundsPass() { return std::make_unique<GEPInboundsPass>(); }

} // namespace catalyst
