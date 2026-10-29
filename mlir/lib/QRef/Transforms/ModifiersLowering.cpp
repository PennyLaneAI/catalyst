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
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "Catalyst/IR/CatalystDialect.h"
#include "QRef/Transforms/Passes.h"
#include "QRef/Transforms/Patterns.h"
#include "Quantum/IR/QuantumDialect.h"
#include "Quantum/Transforms/Passes.h"

using namespace mlir;

namespace catalyst {
namespace qref {

#define GEN_PASS_DEF_MODIFIERSLOWERINGPASS
#include "QRef/Transforms/Passes.h.inc"

// Lower `qref.ctrl` and `qref.adjoint` regions to op-level modifiers in a single greedy
// fixpoint. Both lowering patterns defer (return failure) while their region still holds the other
// modifier, so the greedy worklist interleaves them: an inner adjoint is reduced, then its
// enclosing control lowers on a re-try, and vice versa. This resolves arbitrarily nested modifiers
// without manually alternating the two standalone passes.
struct ModifiersLoweringPass : impl::ModifiersLoweringPassBase<ModifiersLoweringPass> {
    using ModifiersLoweringPassBase::ModifiersLoweringPassBase;

    void runOnOperation() final {
        Operation *op = getOperation();
        OpPassManager toReference(op->getName());
        toReference.addPass(quantum::createReferenceSemanticsConversionPass());
        if (failed(runPipeline(toReference, op))) {
            return signalPassFailure();
        }

        RewritePatternSet patterns(&getContext());
        bool loweringFailed = false;
        populateAdjointLoweringPatterns(patterns, loweringFailed);
        populateCtrlLoweringPatterns(patterns, loweringFailed);

        if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))) || loweringFailed) {
            return signalPassFailure();
        }

        OpPassManager toValue(op->getName());
        toValue.addPass(createValueSemanticsConversionPass());
        if (failed(runPipeline(toValue, op))) {
            return signalPassFailure();
        }
    }
};

} // namespace qref
} // namespace catalyst
