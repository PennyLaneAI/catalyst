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

#define DEBUG_TYPE "commute-clifford-t-ppr"

#include "llvm/Support/Debug.h"

#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "QEC/IR/QECDialect.h"
#include "QEC/Transforms/Patterns.h"

using namespace llvm;
using namespace mlir;
using namespace catalyst;
using namespace catalyst::qec;

namespace catalyst {
namespace qec {

#define GEN_PASS_DEF_COMMUTECLIFFORDTPPRPASS
#define GEN_PASS_DECL_COMMUTECLIFFORDTPPRPASS
#include "QEC/Transforms/Passes.h.inc"

struct CommuteCliffordTPPRPass : public impl::CommuteCliffordTPPRPassBase<CommuteCliffordTPPRPass> {
    using CommuteCliffordTPPRPassBase::CommuteCliffordTPPRPassBase;

    void runOnOperation() final
    {
        RewritePatternSet patterns(&getContext());

        populateCommuteCliffordTPPRPatterns(patterns);

        if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns)))) {
            return signalPassFailure();
        }
    }
};

} // namespace qec

std::unique_ptr<Pass> createCommuteCliffordTPPRPass()
{
    return std::make_unique<CommuteCliffordTPPRPass>();
}

} // namespace catalyst
