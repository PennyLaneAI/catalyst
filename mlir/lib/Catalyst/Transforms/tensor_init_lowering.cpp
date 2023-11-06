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

#define DEBUG_TYPE "tensor-init"

#include <vector>

#include "llvm/Support/Debug.h"

#include "mhlo/IR/hlo_ops.h"
#include "mhlo/transforms/passes.h"

#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Transforms/DialectConversion.h"

#include "Catalyst/Transforms/Patterns.h"

using namespace llvm;
using namespace mlir;
using namespace catalyst;

namespace catalyst {
#define GEN_PASS_DEF_TENSORINITLOWERINGPASS
#include "Catalyst/Transforms/Passes.h.inc"

struct TensorInitLoweringPass : impl::TensorInitLoweringPassBase<TensorInitLoweringPass> {
    using TensorInitLoweringPassBase::TensorInitLoweringPassBase;

    void runOnOperation() final
    {
        LLVM_DEBUG(dbgs() << "tensor init lowering pass"
                          << "\n");

        RewritePatternSet patterns(&getContext());
        populateTensorInitPatterns(patterns);
        if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns)))) {
            return signalPassFailure();
        }
    }
};

std::unique_ptr<Pass> createTensorInitLoweringPass()
{
    return std::make_unique<TensorInitLoweringPass>();
}

} // namespace catalyst
