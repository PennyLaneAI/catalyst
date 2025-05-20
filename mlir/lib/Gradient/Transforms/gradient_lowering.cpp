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

#include <memory>
#include <vector>

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "Catalyst/IR/CatalystDialect.h"
#include "Gradient/IR/GradientOps.h"
#include "Gradient/Transforms/Passes.h"
#include "Gradient/Transforms/Patterns.h"
#include "Quantum/IR/QuantumOps.h"
#include "Quantum/Utils/RemoveQuantum.h"

using namespace mlir;
using namespace catalyst::gradient;

namespace catalyst {
namespace gradient {

#define GEN_PASS_DECL_GRADIENTLOWERINGPASS
#define GEN_PASS_DEF_GRADIENTLOWERINGPASS
#include "Gradient/Transforms/Passes.h.inc"

struct GradientLoweringPass : impl::GradientLoweringPassBase<GradientLoweringPass> {
    using GradientLoweringPassBase::GradientLoweringPassBase;

    void runOnOperation() final
    {
        RewritePatternSet gradientPatterns(&getContext());
        populateLoweringPatterns(gradientPatterns);

        // This is required to remove qubit values returned by if/for ops in the
        // quantum gradient function of the parameter-shift pattern.
        scf::IfOp::getCanonicalizationPatterns(gradientPatterns, &getContext());
        scf::ForOp::getCanonicalizationPatterns(gradientPatterns, &getContext());
        catalyst::quantum::InsertOp::getCanonicalizationPatterns(gradientPatterns, &getContext());
        catalyst::quantum::DeallocOp::getCanonicalizationPatterns(gradientPatterns, &getContext());

        if (failed(applyPatternsGreedily(getOperation(), std::move(gradientPatterns)))) {
            return signalPassFailure();
        }

        // Guarantee that functions intended to be free of quantum ops are indeed so after folding.
        for (Region &region : getOperation()->getRegions()) {
            for (Operation &op : region.getOps()) {
                if (isa<func::FuncOp>(op) && op.hasAttr("QuantumFree"))
                    if (failed(quantum::verifyQuantumFree(cast<func::FuncOp>(op)))) {
                        op.emitOpError() << "cloned during the gradient pass is not free of "
                                            "quantum ops:\n"
                                         << op;
                        return signalPassFailure();
                    }
            }
        }
    }
};

} // namespace gradient

std::unique_ptr<Pass> createGradientLoweringPass()
{
    return std::make_unique<gradient::GradientLoweringPass>();
}

} // namespace catalyst
