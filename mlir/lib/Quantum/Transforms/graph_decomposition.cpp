// Copyright 2026 Xanadu Quantum Technologies Inc.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#define DEBUG_TYPE "graph-decomposition"

#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

#include "Catalyst/Transforms/Passes.h"
#include "Quantum/IR/QuantumOps.h"
#include "Quantum/Transforms/Patterns.h"

#include "graph_decomp_solver.hpp"

using namespace mlir;
using namespace catalyst::quantum;

namespace catalyst {
namespace quantum {
#define GEN_PASS_DEF_GRAPHDECOMPOSITIONPASS
#define GEN_PASS_DECL_GRAPHDECOMPOSITIONPASS
#include "Quantum/Transforms/Passes.h.inc"

struct GraphDecompositionPass : impl::GraphDecompositionPassBase<GraphDecompositionPass> {
    using GraphDecompositionPassBase::GraphDecompositionPassBase;
    void runOnOperation() final
    {
        ModuleOp module = cast<ModuleOp>(getOperation());

        ///////////////////////////
        // Step 1: annotate the user-defined (custom) rules with resources
        // and register them for later use in the graph decomposition.
        registerCustomDecompositionRules(module, customRules);

        ///////////////////////////
        // Step 2: Get the target gateset for decomposition from the module attribute
        getDecompGateset(module, targetGateset);

        ///////////////////////////
        // Step 3: Get and convert operators in the module required for creating the graph
        getOperators(module, setOfOps);

        ///////////////////////////
        // Step 4: Get the resources for all the rules (both built-in and custom)
        // and convert them to RuleNodes for later use in the graph decomposition
        getRuleNodes(module, customRules, setOfResources);

        ///////////////////////////
        // Step 5: Build and solve the decomposition graph
        auto solution = GraphDecompositionSolver::Solve(setOfOps, setOfResources, targetGateset);

        ///////////////////////////
        // Step 6: Insert decomposition rules picked by the graph solver (solution) into the module
        // and then run the decompose-lowering patterns to apply the decomposition rules and
        // rewrite the quantum operations.

        ///////////////////////////
        // Step 7: Run decompose-lowering patterns to apply the decomposition rules
        // TODO TODO
    }

  private:
    // Registry of custom decomposition rules defined in the main module, mapping from target gate
    // name to the corresponding function.
    llvm::StringMap<func::FuncOp> customRules;

    // To sync with the graph solver branch, targetGateset should be DictionaryAttr
    // TODO: it requires updates to the frontend.
    DictionaryAttr targetGateset;

    // List of operators
    std::vector<OperatorNode> setOfOps = {};

    // List of all resources both built-in and custom rules
    std::vector<RuleNode> setOfResources = {};

    void loadBuiltInDecompositionRules([[maybe_unused]] ModuleOp module /*, ...*/) { return; }

    void registerCustomDecompositionRules(ModuleOp module,
                                          llvm::StringMap<func::FuncOp> &custom_rules)
    {
        PassManager pm(&getContext());
        pm.addPass(createRegisterDecompRuleResourcePass());
        if (failed(pm.run(module))) {
            return signalPassFailure();
        }

        module.walk([&](func::FuncOp func) {
            if (StringRef funcName = func.getName();
                func->getAttrOfType<StringAttr>("target_gate")) {
                // TODO: Update this to only register rules that are customly defined for this
                // specific qp.decompose HOW? it requires updates to the lowering patterns from the
                // frontend ...

                custom_rules[funcName] = func;
            }
            return WalkResult::skip();
        });
    }

    void getDecompGateset(ModuleOp module, DictionaryAttr &gateset)
    {
        // module.walk([&](func::FuncOp func) {
        //     if (auto gate_set_attr =
        //             func->getAttrOfType<ArrayAttr>("decomp_gateset")) {
        //         for (auto gate : gate_set_attr.getValue()) {
        //             StringRef gate_name = cast<StringAttr>(gate).getValue();
        //             gateset.insert(gate_name);
        //         }
        //         return WalkResult::interrupt();
        //     }
        //     return WalkResult::skip();
        // });
        return;
    }

    void getOperators([[maybe_unused]] ModuleOp module,
                      [[maybe_unused]] std::vector<OperatorNode> &operators)
    {

        module.walk([&](CustomOp op) {
            OperatorNode node;
            node.op = op;
            node.name = op->getName().getStringRef();
            node.weight = 1.0; // TODO: gates could be lowered by some weights
            operators.push_back(node);
        });
    }

    void getRuleNodes([[maybe_unused]] ModuleOp module,
                      [[maybe_unused]] const llvm::StringMap<func::FuncOp> &custom_rules,
                      [[maybe_unused]] std::vector<RuleNode> &rules)
    {
        return;
    }
};

} // namespace quantum
} // namespace catalyst
