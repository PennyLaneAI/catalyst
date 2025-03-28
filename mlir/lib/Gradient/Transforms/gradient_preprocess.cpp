// Copyright 2022-2023 Xanadu Quantum Technologies Inc.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/Transforms/Bufferize.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "Gradient/IR/GradientOps.h"
#include "Gradient/Transforms/Passes.h"
#include "Gradient/Transforms/Patterns.h"

using namespace mlir;
using namespace catalyst::gradient;

namespace catalyst {
namespace gradient {

#define GEN_PASS_DEF_GRADIENTPREPROCESSINGPASS
#include "Gradient/Transforms/Passes.h.inc"

struct GradientPreprocessingPass : impl::GradientPreprocessingPassBase<GradientPreprocessingPass> {
    using GradientPreprocessingPassBase::GradientPreprocessingPassBase;

    void runOnOperation() final
    {
        RewritePatternSet patterns(&getContext());
        populatePreprocessingPatterns(patterns);

        if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns)))) {
            return signalPassFailure();
        }
    }
};

} // namespace gradient

std::unique_ptr<Pass> createGradientPreprocessingPass()
{
    return std::make_unique<gradient::GradientPreprocessingPass>();
}

} // namespace catalyst