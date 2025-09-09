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

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/AllocatorBase.h"

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

// Check if a function is a decomposition function
// It's expected that the decomposition function would have these two attributes:
// `catalyst.decomposition` and `catalyst.decomposition.target_op`
// And these are set by the `updateDecompositionAttributes` function
// The decomposition attributes are used to determine if a function is a decomposition function,
// and target_op is that the decomposition function want to replace
bool isDecompositionFunction(func::FuncOp func)
{
    return func->hasAttr("catalyst.decomposition") &&
           func->hasAttr("catalyst.decomposition.target_op");
}

// Update decomposition attributes if function name begins with <Gate>_rule
// It's name sensitive, will not work if the function name is not following <Gate>_rule pattern
void markDecompositionAttributes(func::FuncOp func, MLIRContext *context)
{
    StringRef funcName = func.getSymName();

    // Check if function name follows <Gate>_rule pattern
    size_t firstUnderscore = funcName.find('_');
    if (firstUnderscore == StringRef::npos) {
        return;
    }

    StringRef afterFirstUnderscore = funcName.substr(firstUnderscore + 1);
    if ((afterFirstUnderscore.size() >= 4) && (afterFirstUnderscore.substr(0, 4) == "rule")) {
        if (StringRef gateName = funcName.substr(0, firstUnderscore); !gateName.empty()) {
            // Set the decomposition attributes
            func->setAttr("catalyst.decomposition", UnitAttr::get(context));
            func->setAttr("catalyst.decomposition.target_op", StringAttr::get(context, gateName));
            return;
        }
    }
    return;
}

StringRef getTargetOp(func::FuncOp func)
{
    if (auto target_op_attr = func->getAttrOfType<StringAttr>("catalyst.decomposition.target_op")) {
        return target_op_attr.getValue();
    }
    return StringRef{};
}

} // namespace DecompositionUtils

/// A module pass that work through a module, register all decomposition functions, and apply the
/// decomposition patterns
struct UserDefinedDecompositionPass
    : impl::UserDefinedDecompositionPassBase<UserDefinedDecompositionPass> {
    using UserDefinedDecompositionPassBase::UserDefinedDecompositionPassBase;

    void getDependentDialects(DialectRegistry &registry) const override
    {
        registry.insert<arith::ArithDialect>();
        registry.insert<func::FuncDialect>();
        registry.insert<quantum::QuantumDialect>();
        registry.insert<tensor::TensorDialect>();
        registry.insert<ub::UBDialect>();
    }

  private:
    llvm::StringMap<func::FuncOp> decompositionRegistry;
    llvm::StringSet<llvm::MallocAllocator> targetGateSet;

    // Function to discover and register decomposition functions from a module
    void discoverAndRegisterDecompositions(ModuleOp module,
                                           llvm::StringMap<func::FuncOp> &decompositionRegistry)
    {
        module.walk([&](func::FuncOp func) {
            DecompositionUtils::markDecompositionAttributes(func, &getContext());
            if (!DecompositionUtils::isDecompositionFunction(func)) {
                return;
            }

            if (StringRef targetOp = DecompositionUtils::getTargetOp(func); !targetOp.empty()) {
                decompositionRegistry[targetOp] = func;
            }
        });
    }

    void findTargetGateSet(ModuleOp module, llvm::StringSet<llvm::MallocAllocator> &targetGateSet)
    {
        WalkResult walkResult = module.walk([&](func::FuncOp func) {
            if (auto gate_set_attr = func->getAttrOfType<ArrayAttr>("gate_set")) {
                for (auto gate : gate_set_attr.getValue()) {
                    StringRef gate_name = cast<StringAttr>(gate).getValue();
                    targetGateSet.insert(gate_name);
                }
                return WalkResult::interrupt();
            }
            return WalkResult::advance();
        });
        if (!walkResult.wasInterrupted()) {
        }
    }

    void removeDecompositionFunctions(ModuleOp module,
                                      llvm::StringMap<func::FuncOp> &decompositionRegistry)
    {
        module.walk([&](func::FuncOp func) {
            if (DecompositionUtils::isDecompositionFunction(func)) {
                func.erase();
            }
        });
    }

  public:
    void runOnOperation() final
    {
        ModuleOp module = cast<ModuleOp>(getOperation());

        // Step 1: Discover and register all decomposition functions in the module
        discoverAndRegisterDecompositions(module, decompositionRegistry);
        if (decompositionRegistry.empty()) {
            return;
        }

        // Step 1.1: Find the target gate set
        findTargetGateSet(module, targetGateSet);

        // Step 2: Canonicalize the module
        RewritePatternSet patternsCanonicalization(&getContext());
        catalyst::quantum::CustomOp::getCanonicalizationPatterns(patternsCanonicalization,
                                                                 &getContext());
        if (failed(applyPatternsGreedily(module, std::move(patternsCanonicalization)))) {
            return signalPassFailure();
        }

        // Step 3: Apply the decomposition patterns
        RewritePatternSet decompositionPatterns(&getContext());
        populateUserDefinedDecompositionPatterns(decompositionPatterns, decompositionRegistry,
                                                 targetGateSet);
        if (failed(applyPatternsGreedily(module, std::move(decompositionPatterns)))) {
            return signalPassFailure();
        }

        // Step 4: Inline and canonicalize/CSE the module again
        PassManager pm(&getContext());
        pm.addPass(createInlinerPass());
        pm.addPass(createCanonicalizerPass());
        pm.addPass(createCSEPass());
        if (failed(pm.run(module))) {
            return signalPassFailure();
        }

        // Step 5. Remove redundant decomposition functions
        removeDecompositionFunctions(module, decompositionRegistry);

        // Step 6. Canonicalize the extract/insert pair
        RewritePatternSet patternsInsertExtract(&getContext());
        catalyst::quantum::InsertOp::getCanonicalizationPatterns(patternsInsertExtract,
                                                                 &getContext());
        catalyst::quantum::ExtractOp::getCanonicalizationPatterns(patternsInsertExtract,
                                                                  &getContext());
        if (failed(applyPatternsGreedily(module, std::move(patternsInsertExtract)))) {
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
