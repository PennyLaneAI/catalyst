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

#define DEBUG_TYPE "user-defined-decomposition"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/Debug.h"

#include "Quantum/IR/QuantumOps.h"
#include "Quantum/Transforms/Patterns.h"

using namespace mlir;
using namespace catalyst::quantum;

namespace catalyst {
namespace quantum {

#define GEN_PASS_DEF_USERDEFINEDDECOMPOSITIONPASS
#define GEN_PASS_DECL_USERDEFINEDDECOMPOSITIONPASS
#include "Quantum/Transforms/Passes.h.inc"

namespace DecompositionUtils {

bool isDecompositionFunction(func::FuncOp func) { return func->hasAttr("catalyst.decomposition"); }

StringRef getTargetOp(func::FuncOp func)
{
    if (auto target_op_attr = func->getAttrOfType<StringAttr>("catalyst.decomposition.target_op")) {
        return target_op_attr.getValue();
    }
    return StringRef{};
}

} // namespace DecompositionUtils

struct UserDefinedDecompositionPass
    : impl::UserDefinedDecompositionPassBase<UserDefinedDecompositionPass> {
    using UserDefinedDecompositionPassBase::UserDefinedDecompositionPassBase;

  private:
    llvm::StringMap<func::FuncOp> decompositionRegistry;

    // Function to discover and register decomposition functions from a module
    void discoverAndRegisterDecompositions(ModuleOp module,
                                           llvm::StringMap<func::FuncOp> &decompositionRegistry)
    {
        LLVM_DEBUG(llvm::dbgs()
                   << "========== [DECOMP REGISTRY] Starting decomposition discovery ==========\n");

        int decompFunctions = 0;
        int registeredMappings = 0;

        module.walk([&](func::FuncOp func) {
            if (!DecompositionUtils::isDecompositionFunction(func)) {
                return;
            }
            decompFunctions++;

            StringRef targetOp = DecompositionUtils::getTargetOp(func);
            StringRef funcName = func.getSymName();

            LLVM_DEBUG(llvm::dbgs() << "  [INFO] Decomposition name: '" << funcName << "'\n"
                                    << "  [INFO] Target operation: '" << targetOp << "'\n";);

            if (!targetOp.empty()) {
                LLVM_DEBUG(llvm::dbgs() << "  [REGISTER] Mapping gate '" << targetOp
                                        << "' -> function '" << funcName << "'\n");
                decompositionRegistry[targetOp] = func;
                registeredMappings++;
            }
            else {
                LLVM_DEBUG(llvm::dbgs() << "  [WARNING] No target operation specified for "
                                           "decomposition function\n";);
            }
        });

        LLVM_DEBUG(
            llvm::dbgs() << "  [REGISTRY SUMMARY] Discovery completed!\n"
                         << "    Decomposition functions found: " << decompFunctions << "\n"
                         << "    Total gate->function mappings registered: " << registeredMappings
                         << "\n"
                         << "========== [DECOMP REGISTRY] Discovery completed ==========\n";);
    }

  public:
    void runOnOperation() final
    {
        ModuleOp module = cast<ModuleOp>(getOperation());

        // Discover and register all decomposition functions in the module
        discoverAndRegisterDecompositions(module, decompositionRegistry);
        if (decompositionRegistry.empty()) {
            return;
        }

        RewritePatternSet patternsCanonicalization(&getContext());
        catalyst::quantum::CustomOp::getCanonicalizationPatterns(patternsCanonicalization,
                                                                 &getContext());
        if (failed(applyPatternsGreedily(module, std::move(patternsCanonicalization)))) {
            return signalPassFailure();
        }

        RewritePatternSet decompositionPatterns(&getContext());
        populateUserDefinedDecompositionPatterns(decompositionPatterns, decompositionRegistry);
        if (failed(applyPatternsGreedily(module, std::move(decompositionPatterns)))) {
            return signalPassFailure();
        }

        PassManager pm(&getContext());
        pm.addPass(createInlinerPass());
        if (failed(pm.run(module))) {
            return signalPassFailure();
        }
    }
};

} // namespace quantum

std::unique_ptr<Pass> createUserDefinedDecompositionPass()
{
    return std::make_unique<quantum::UserDefinedDecompositionPass>();
}

} // namespace catalyst
