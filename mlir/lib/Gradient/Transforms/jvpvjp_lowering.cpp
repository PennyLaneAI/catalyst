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

#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "Gradient/IR/GradientOps.h"
#include "Gradient/Transforms/Passes.h"

#include "GradMethods/JVPVJPPatterns.hpp"

using namespace mlir;
using namespace catalyst::gradient;

namespace catalyst {
namespace gradient {

struct JVPVJPLoweringPass : public PassWrapper<JVPVJPLoweringPass, OperationPass<ModuleOp>> {
    JVPVJPLoweringPass() {}

    StringRef getArgument() const override { return "lower-jvpvjp"; }

    StringRef getDescription() const override
    {
        return "Lower JVP/VJP operations down to grad and linalg.generic operations.";
    }

    void getDependentDialects(DialectRegistry &registry) const override
    {
        registry.insert<linalg::LinalgDialect>();
        registry.insert<bufferization::BufferizationDialect>();
        registry.insert<memref::MemRefDialect>();
    }

    void runOnOperation() final
    {
        ModuleOp op = getOperation();

        RewritePatternSet patterns(&getContext());
        patterns.add<JVPLoweringPattern>(patterns.getContext());
        patterns.add<VJPLoweringPattern>(patterns.getContext());

        if (failed(applyPatternsAndFoldGreedily(op, std::move(patterns)))) {
            return signalPassFailure();
        }
    }
};

} // namespace gradient

std::unique_ptr<Pass> createJVPVJPLoweringPass()
{
    return std::make_unique<gradient::JVPVJPLoweringPass>();
}

} // namespace catalyst
