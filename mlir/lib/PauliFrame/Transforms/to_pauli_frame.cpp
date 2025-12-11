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

#define DEBUG_TYPE "to-pauli-frame"

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/WalkPatternRewriteDriver.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/LogicalResult.h"

#include "PauliFrame/IR/PauliFrameOps.h"
#include "PauliFrame/Transforms/Patterns.h"

using namespace llvm;
using namespace mlir;

namespace catalyst {
namespace pauli_frame {

#define GEN_PASS_DECL_CLIFFORDTTOPAULIFRAMEPASS
#define GEN_PASS_DEF_CLIFFORDTTOPAULIFRAMEPASS
#include "PauliFrame/Transforms/Passes.h.inc"

struct CliffordTToPauliFramePass : impl::CliffordTToPauliFramePassBase<CliffordTToPauliFramePass> {
    using CliffordTToPauliFramePassBase::CliffordTToPauliFramePassBase;

    void runOnOperation() final
    {
        LLVM_DEBUG(dbgs() << "Clifford+T to Pauli frame pass\n");

        Operation *module = getOperation();

        RewritePatternSet patterns(&getContext());
        populateCliffordTToPauliFramePatterns(patterns);

        // NOTE: We want the walk-based pattern rewrite driver here, and not a greedy rewriter like
        // applyPatternsGreedily, since we match on ops and do not replace them, therefore a greedy
        // rewriter would loop infinitely.
        walkAndApplyPatterns(module, std::move(patterns));
    }
};

} // namespace pauli_frame
} // namespace catalyst
