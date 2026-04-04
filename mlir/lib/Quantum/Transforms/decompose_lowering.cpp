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

#define DEBUG_TYPE "decompose-lowering"

// When we read the decomposition rules module from file,
// StablehloDialect may not be registered from start.
#include "stablehlo/dialect/StablehloOps.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/AllocatorBase.h"

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

#include "Quantum/IR/QuantumOps.h"
#include "Quantum/Transforms/Patterns.h"

using namespace mlir;
using namespace catalyst::quantum;

namespace catalyst {
namespace quantum {
#define GEN_PASS_DEF_DECOMPOSELOWERINGPASS
#define GEN_PASS_DECL_DECOMPOSELOWERINGPASS
#include "Quantum/Transforms/Passes.h.inc"

namespace DecompUtils {

static constexpr StringRef decomposition_rule_attr_name = "decomposition_rule";
static constexpr StringRef target_gate_attr_name = "target_gate";
static constexpr StringRef decomp_gateset_attr_name = "decomp_gateset";

// Check if a function is a decomposition function
// It's expected that the decomposition function would have this attribute:
// `catalyst.decomposition.target_op` And this attribute is set by the `markDecompositionAttributes`
// functionq The decomposition attribute are used to determine if a function is a decomposition
// function, and target_op is that the decomposition function want to replace
bool isDecompositionFunction(func::FuncOp func)
{
    return func->hasAttr(decomposition_rule_attr_name);
}

StringRef getTargetGateName(func::FuncOp func)
{
    if (auto target_op_attr = func->getAttrOfType<StringAttr>(target_gate_attr_name)) {
        return target_op_attr.getValue();
    }
    return StringRef{};
}

uint64_t getNumWires(func::FuncOp func)
{
    if (auto num_wires_attr = func->getAttrOfType<IntegerAttr>("num_wires")) {
        return num_wires_attr.getValue().getZExtValue();
    }
    return 0;
}

} // namespace DecompUtils

/// A module pass that work through a module, register all decomposition functions, and apply the
/// decomposition patterns
struct DecomposeLoweringPass : impl::DecomposeLoweringPassBase<DecomposeLoweringPass> {
    using DecomposeLoweringPassBase::DecomposeLoweringPassBase;

    void getDependentDialects(DialectRegistry &registry) const override
    {
        registry.insert<arith::ArithDialect>();
        registry.insert<func::FuncDialect>();
        registry.insert<quantum::QuantumDialect>();
        registry.insert<mlir::stablehlo::StablehloDialect>();
        registry.insert<tensor::TensorDialect>();
        registry.insert<ub::UBDialect>();
    }

  private:
    llvm::StringMap<func::FuncOp> decompositionRegistry;
    llvm::StringSet<llvm::MallocAllocator> targetGateSet;

    // Function to discover and register decomposition functions from a module
    // It's bookkeeping the targetOp and the decomposition function that can decompose the targetOp
    void discoverAndRegisterDecompositions(ModuleOp module,
                                           llvm::StringMap<func::FuncOp> &decompositionRegistry)
    {
        module.walk([&](func::FuncOp func) {
            if (StringRef targetOp = DecompUtils::getTargetGateName(func); !targetOp.empty()) {
                if (targetOp == "MultiRZ") {
                    // Create a new target op name with the number of wires
                    // for MultiRZ, since it has multiple decomposition functions
                    // based on the number of target qubits
                    std::string newTargetOpStr =
                        targetOp.str() + "_" + std::to_string(DecompUtils::getNumWires(func));

                    decompositionRegistry[newTargetOpStr] = func;
                }
                else {
                    decompositionRegistry[targetOp] = func;
                }
            }
            // No need to walk into the function body
            return WalkResult::skip();
        });
    }

    // Find the target gate set from the module.It's expected that the decomposition function would
    // have this attribute: `decomp_gateset` And this attribute is set by the frontend, it contains
    // the target gate set that the circuit function want to finally decompose into. Since each
    // module only contains one circuit function, we can just find the target gate set from the
    // function with the `decomp_gateset` attribute
    void findTargetGateSet(ModuleOp module, llvm::StringSet<llvm::MallocAllocator> &targetGateSet)
    {
        module.walk([&](func::FuncOp func) {
            if (auto gate_set_attr =
                    func->getAttrOfType<ArrayAttr>(DecompUtils::decomp_gateset_attr_name)) {
                for (auto gate : gate_set_attr.getValue()) {
                    StringRef gate_name = cast<StringAttr>(gate).getValue();
                    targetGateSet.insert(gate_name);
                }
                return WalkResult::interrupt();
            }
            // No need to walk into the function body
            return WalkResult::skip();
        });
    }

    // Remove unused decomposition functions:
    // Since the decomposition functions are marked as public from the frontend,
    // there is no way to remove them with any DCE pass automatically.
    // So we need to manually remove them from the module
    void removeDecompositionFunctions(ModuleOp module,
                                      llvm::StringMap<func::FuncOp> &decompositionRegistry)
    {
        llvm::DenseSet<func::FuncOp> usedDecompositionFunctions;

        module.walk([&](func::CallOp callOp) {
            if (auto targetFunc = module.lookupSymbol<func::FuncOp>(callOp.getCallee())) {
                if (DecompUtils::isDecompositionFunction(targetFunc)) {
                    usedDecompositionFunctions.insert(targetFunc);
                }
            }
        });

        // remove unused decomposition functions
        module.walk([&](func::FuncOp func) {
            if (DecompUtils::isDecompositionFunction(func) &&
                !usedDecompositionFunctions.contains(func)) {
                func.erase();
            }
            return WalkResult::skip();
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
        populateDecomposeLoweringPatterns(decompositionPatterns, decompositionRegistry,
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
} // namespace catalyst
