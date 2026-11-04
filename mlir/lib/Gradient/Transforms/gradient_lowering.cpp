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
#include <utility> // std::move
#include <vector>

#include "mlir/Analysis/CallGraph.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Inliner.h"

#include "Catalyst/IR/CatalystDialect.h"
#include "Gradient/IR/GradientOps.h"
#include "Gradient/Transforms/Patterns.h"
#include "Gradient/Utils/DifferentialQNode.h"
#include "Quantum/IR/QuantumOps.h"
#include "Quantum/Utils/RemoveQuantum.h"

using namespace mlir;
using namespace catalyst::gradient;

namespace catalyst {
namespace gradient {

#define GEN_PASS_DEF_GRADIENTLOWERINGPASS
#include "Gradient/Transforms/Passes.h.inc"

struct GradientLoweringPass : impl::GradientLoweringPassBase<GradientLoweringPass> {
    using GradientLoweringPassBase::GradientLoweringPassBase;

    void runOnOperation() final {
        auto shouldInlineCall = [](const Inliner::ResolvedCall &call) {
            auto callOp = call.call;
            if (!isa<func::CallOp>(callOp.getOperation())) {
                return false;
            }
            auto caller = callOp->getParentOfType<func::FuncOp>();
            auto callee =
                dyn_cast<func::FuncOp>(call.targetNode->getCallableRegion()->getParentOp());
            if (!caller || !callee || isQNode(callee)) {
                return false;
            }
            StringRef method = getQNodeDiffMethod(caller);
            return method == "parameter-shift" || method == "adjoint";
        };

        // Expose decomposition gates before splitting QNodes into classical and quantum parts.
        // Keep calls to QNodes intact so their differentiation method is preserved.
        InlinerConfig config;
        Inliner inliner(
            getOperation(), getAnalysis<CallGraph>(), *this, getAnalysisManager(),
            [this](Pass &, OpPassManager &pipeline, Operation *op) {
                return runPipeline(pipeline, op);
            },
            config, shouldInlineCall);
        if (failed(inliner.doInlining())) {
            return signalPassFailure();
        }

        RewritePatternSet gradientPatterns(&getContext());
        populateLoweringPatterns(gradientPatterns);

        // This is required to remove qubit values returned by if/for ops in the
        // quantum gradient function of the parameter-shift pattern.
        scf::IfOp::getCanonicalizationPatterns(gradientPatterns, &getContext());
        scf::ForOp::getCanonicalizationPatterns(gradientPatterns, &getContext());
        catalyst::quantum::ExtractOp::getCanonicalizationPatterns(gradientPatterns, &getContext());
        catalyst::quantum::DeallocOp::getCanonicalizationPatterns(gradientPatterns, &getContext());
        catalyst::quantum::AllocOp::getCanonicalizationPatterns(gradientPatterns, &getContext());

        if (failed(applyPatternsGreedily(getOperation(), std::move(gradientPatterns)))) {
            return signalPassFailure();
        }

        // Guarantee that functions intended to be free of quantum ops are indeed so after folding.
        getOperation()->walk([&](func::FuncOp f) {
            if (f->hasAttr("QuantumFree")) {
                if (failed(quantum::verifyQuantumFree(f))) {
                    f->emitOpError() << "cloned during the gradient pass is not free of "
                                        "quantum ops:\n"
                                     << *f;
                    return signalPassFailure();
                }
            }
        });
    }
};

} // namespace gradient
} // namespace catalyst
