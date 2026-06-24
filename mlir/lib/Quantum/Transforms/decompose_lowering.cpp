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

#include <cstdint>
#include <string>
#include <utility>

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/AllocatorBase.h"
#include "llvm/Support/DebugLog.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/WalkResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Inliner.h"
#include "mlir/Transforms/Passes.h"
#include "stablehlo/dialect/StablehloOps.h" // When we read the decomposition rules module from file, StablehloDialect may not be registered from start.

#include "Quantum/IR/QuantumDialect.h"
#include "Quantum/IR/QuantumOps.h"
#include "Quantum/Transforms/Patterns.h"

using namespace mlir;
using namespace catalyst::quantum;

#define DEBUG_TYPE "decompose-lowering"

namespace catalyst {
namespace quantum {

#define GEN_PASS_DEF_DECOMPOSELOWERINGPASS
#define GEN_PASS_DECL_DECOMPOSELOWERINGPASS
#include "Quantum/Transforms/Passes.h.inc"

namespace DecompUtils {

static constexpr StringRef target_gate_attr_name = "target_gate";
static constexpr StringRef decomp_gateset_attr_name = "decomp_gateset";

// Check if a function is a decomposition function
// It's expected that the decomposition function would have this attribute:
// `catalyst.decomposition.target_op` And this attribute is set by the `markDecompositionAttributes`
// functionq The decomposition attribute are used to determine if a function is a decomposition
// function, and target_op is that the decomposition function want to replace
bool isDecompositionFunction(func::FuncOp func) { return func->hasAttr(target_gate_attr_name); }

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
                                           llvm::StringMap<func::FuncOp> &decompositionRegistry,
                                           llvm::StringSet<> targetRules)
    {
        module.walk([&](func::FuncOp func) {
            // if targetRules is provided, only add requested rules
            if (!targetRules.empty() && !targetRules.contains(func.getName())) {
                return WalkResult::skip();
            }

            if (StringRef targetOp = DecompUtils::getTargetGateName(func); !targetOp.empty()) {
                removeUnusedFuncArgs(func);
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

    // Remove unused arguments on a decomposition function
    // This is because we have some assumptions on the decomp funcs' signature structure
    void removeUnusedFuncArgs(func::FuncOp f)
    {
        f.front().eraseArguments([](BlockArgument arg) { return arg.use_empty(); });

        f.setFunctionType(FunctionType::get(f->getContext(), f.front().getArgumentTypes(),
                                            f.front().getTerminator()->getOperandTypes()));
    }

  public:
    void runOnOperation() final
    {
        ModuleOp module = cast<ModuleOp>(getOperation());

        // Step 1: Discover and register all decomposition functions in the module
        llvm::StringSet<> targetRules;
        for (auto rule : targetRulesOption) {
            targetRules.insert(rule);
        }
        discoverAndRegisterDecompositions(module, decompositionRegistry, targetRules);
        if (decompositionRegistry.empty()) {
            return;
        }

        // Step 2: Find the target gate set
        findTargetGateSet(module, targetGateSet);

        // Step 3: Apply the decomposition patterns, canonicalizing the insert/extract pairs
        RewritePatternSet decompositionPatterns(&getContext());
        populateDecomposeLoweringPatterns(decompositionPatterns, decompositionRegistry,
                                          targetGateSet);
        catalyst::quantum::InsertOp::getCanonicalizationPatterns(decompositionPatterns,
                                                                 &getContext());
        catalyst::quantum::ExtractOp::getCanonicalizationPatterns(decompositionPatterns,
                                                                  &getContext());
        if (failed(applyPatternsGreedily(module, std::move(decompositionPatterns)))) {
            return signalPassFailure();
        }
    }
};

} // namespace quantum
} // namespace catalyst
