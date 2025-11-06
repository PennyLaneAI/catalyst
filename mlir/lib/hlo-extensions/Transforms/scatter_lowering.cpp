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

#define DEBUG_TYPE "scatter"

#include <vector>

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/transforms/Passes.h"
#include "llvm/Support/Debug.h"

#include "hlo-extensions/Transforms/Patterns.h"

using namespace llvm;
using namespace mlir;
using namespace catalyst;

namespace catalyst {
namespace hlo_extensions {

#define GEN_PASS_DEF_SCATTERLOWERINGPASS
#include "hlo-extensions/Transforms/Passes.h.inc"

struct ScatterLoweringPass : impl::ScatterLoweringPassBase<ScatterLoweringPass> {
    using ScatterLoweringPassBase::ScatterLoweringPassBase;

    void runOnOperation() final
    {
        LLVM_DEBUG(dbgs() << "scatter lowering pass"
                          << "\n");

        RewritePatternSet patterns(&getContext());
        populateScatterPatterns(patterns);
        if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
            return signalPassFailure();
        }
    }
};

} // namespace hlo_extensions
} // namespace catalyst
