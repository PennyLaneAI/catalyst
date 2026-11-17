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

#include "mlir/Dialect/Complex/IR/Complex.h"
#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "Catalyst/IR/CatalystDialect.h"
#include "Quantum/Transforms/Patterns.h"

using namespace mlir;

namespace catalyst {
namespace quantum {

#define GEN_PASS_DEF_MODIFIERSLOWERINGPASS
#include "Quantum/Transforms/Passes.h.inc"

// Lower `quantum.ctrl` and `quantum.adjoint` regions to op-level modifiers in a single greedy
// fixpoint. Both lowering patterns defer (return failure) while their region still holds the other
// modifier, so the greedy worklist interleaves them: an inner adjoint is reduced, then its
// enclosing control lowers on a re-try, and vice versa. This resolves arbitrarily nested modifiers
// without manually alternating the two standalone passes.
struct ModifiersLoweringPass : impl::ModifiersLoweringPassBase<ModifiersLoweringPass> {
    using ModifiersLoweringPassBase::ModifiersLoweringPassBase;

    void runOnOperation() final {
        RewritePatternSet patterns(&getContext());
        populateAdjointLoweringPatterns(patterns);
        populateCtrlLoweringPatterns(patterns);

        if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
            return signalPassFailure();
        }
    }
};

} // namespace quantum
} // namespace catalyst
