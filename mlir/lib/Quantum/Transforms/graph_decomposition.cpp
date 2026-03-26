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

#include "llvm/ADT/StringExtras.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "stablehlo/dialect/StablehloOps.h"

#include "mlir/Pass/PassManager.h"

#include "Catalyst/Transforms/Passes.h"
#include "Quantum/IR/QuantumDialect.h"
#include "Quantum/IR/QuantumOps.h"
#include "Quantum/Transforms/Passes.h"
#include "Quantum/Utils/Decomp.h"

#include "DGBuilder.hpp"
#include "DGSolver.hpp"
#include "DGTypes.hpp"

using namespace mlir;
using namespace catalyst::quantum;
using namespace DecompGraph::Core;
using namespace DecompGraph::Solver;

namespace catalyst {
namespace quantum {
#define GEN_PASS_DEF_GRAPHDECOMPOSITIONPASS
#include "Quantum/Transforms/Passes.h.inc"

struct GraphDecompositionPass : public impl::GraphDecompositionPassBase<GraphDecompositionPass> {
    using GraphDecompositionPassBase::GraphDecompositionPassBase;

    void runOnOperation() final
    {
        llvm::errs() << "Parsed options from CLI:\n";
        llvm::errs() << "\tgate-set:\n";
        for (auto gate : targetGateSetOption) {
            llvm::errs() << "\t\t" << gate << "\n";
        }
        llvm::errs() << "\tfixed-decomps:\n";
        for (auto rule : fixedDecompsOption) {
            llvm::errs() << "\t\t" << rule << "\n";
        }
        llvm::errs() << "\talt-decomps:\n";
        for (auto rule : altDecompsOption) {
            llvm::errs() << "\t\t" << rule << "\n";
        }
        llvm::errs() << "\tbytecode-rules: " << bytecodeRulesFile << "\n";

        // Registry of custom decomposition rules defined in the main module, mapping from
        // target gate name to the corresponding function.
        llvm::StringMap<func::FuncOp> customRules;

        llvm::StringMap<llvm::StringRef> fixedDecomps;
        llvm::StringMap<llvm::StringRef> altDecomps;

        llvm::errs() << "fixed decomps:\n";
        for (const std::string &opRulePair : fixedDecompsOption) {
            llvm::StringRef pairRef(opRulePair);

            auto [opName, ruleName] = pairRef.split("=");
            fixedDecomps[opName] = ruleName;
            llvm::errs() << "\t" << opName << ": " << ruleName << ",\n";
        }

        llvm::errs() << "alt decomps:\n";
        for (const std::string &opRulePair : altDecompsOption) {
            llvm::StringRef pairRef(opRulePair);

            auto [opName, ruleName] = pairRef.split("=");
            altDecomps[opName] = ruleName;
            llvm::errs() << "\t" << opName << ": " << ruleName << ",\n";
        }

        // List of operators
        std::vector<OperatorNode> setOfOps = {};

        // List of all resources both built-in and custom rules
        std::vector<RuleNode> setOfResources = {};
        std::unordered_map<std::string, mlir::OwningOpRef<func::FuncOp>> ruleNameToFuncOp = {};

        ModuleOp module = cast<ModuleOp>(getOperation());

        ///////////////////////////
        // Step 1: annotate the user-defined (custom) rules with resources
        // and register them for later use in the graph decomposition.
        registerCustomDecompositionRules(module, customRules);
        llvm::errs() << "registered user rules\n";

        ///////////////////////////
        // Step 2: Get the target gateset for decomposition from the module attribute
        // To sync with the graph solver branch, targetGateset should be DictionaryAttr
        WeightedGateset targetGateSet;
        llvm::errs() << "target gate set:\n";
        for (const std::string &opCostPair : targetGateSetOption) {
            llvm::StringRef pairRef(opCostPair);

            auto [opName, cost] = pairRef.split("=");
            bool success = to_float(cost, targetGateSet.ops[OperatorNode{opName.str()}]);

            if (!success) {
                return signalPassFailure();
            }
            llvm::errs() << "\t" << opName << ": " << cost << ",\n";
        }

        ///////////////////////////
        // Step 3: Get and convert operators in the module required for creating the graph
        getOperators(module, setOfOps);
        llvm::errs() << "got ops for graph\n";

        ///////////////////////////
        // Step 4: Get the resources for all the rules (both built-in and custom)
        // and convert them to RuleNodes for later use in the graph decomposition
        // TODO get nodes from user rules
        getRuleNodes(module, bytecodeRulesFile, customRules, setOfResources, ruleNameToFuncOp);
        llvm::errs() << "got rules for graph\n";

        ///////////////////////////
        // Step 5: Build and solve the decomposition graph
        DecompositionGraph graph(setOfOps, targetGateSet, setOfResources);
        DecompositionSolver solver(graph);
        auto solution = solver.getSolvedMap();
        // auto solution = GraphDecompositionSolver::Solve(setOfOps, setOfResources,
        // targetGateToCost);
        llvm::errs() << "got solution from graph\n";

        ///////////////////////////
        // Step 6: Insert decomposition rules picked by the graph solver (solution) into the
        // module and then run the decompose-lowering patterns to apply the decomposition rules
        // and rewrite the quantum operations.
        insertChosenRules(module, solution, ruleNameToFuncOp);
        llvm::errs() << "inserted chosen rules\n";

        ///////////////////////////
        // Step 7: Run decompose-lowering patterns to apply the decomposition rules
        PassManager pm(&getContext());
        pm.addPass(createDecomposeLoweringPass());

        if (failed(pm.run(module))) {
            return signalPassFailure();
        }
        llvm::errs() << "ran decompose-lowering\n";
    }

  private:
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
                // specific qp.decompose HOW? it requires updates to the lowering patterns from
                // the frontend ...

                custom_rules[funcName] = func;
            }
            return WalkResult::skip();
        });
    }

    void getOperators([[maybe_unused]] ModuleOp module,
                      [[maybe_unused]] std::vector<OperatorNode> &operators)
    {

        module.walk([&](CustomOp op) {
            OperatorNode node;
            node.name = op.getGateName().str();
            auto inQubits = op.getInQubits();
            node.numWires = inQubits.size();
            auto inParams = op.getParams();
            node.numParams = inParams.size();
            node.adjoint = op.getAdjoint();
            operators.push_back(node);
        });
    }

    void
    getRuleNodes(ModuleOp module, llvm::StringRef filename,
                 const llvm::StringMap<func::FuncOp> &custom_rules, std::vector<RuleNode> &rules,
                 std::unordered_map<std::string, mlir::OwningOpRef<func::FuncOp>> &ruleNameToFuncOp)
    {

        // TODO user nodes

        std::vector<mlir::OwningOpRef<mlir::func::FuncOp>> builtinRules =
            getRulesFromBytecode(filename, module.getContext());

        llvm::errs() << "read bytecode\n";

        for (auto &ruleOpRef : builtinRules) {
            RuleNode ruleNode;
            mlir::func::FuncOp func = ruleOpRef.get(); // access the op
            ruleNode.name = func.getName().str();
            ruleNameToFuncOp[ruleNode.name] = std::move(ruleOpRef);

            // Set output OperatorNode
            if (auto outputGateAttr = func->getAttrOfType<StringAttr>("target_gate")) {
                OperatorNode outputNode;
                outputNode.name = outputGateAttr.getValue().str();
                if (outputNode.name.starts_with("Adjoint(") && outputNode.name.ends_with(")")) {
                    outputNode.adjoint = true;
                    outputNode.name =
                        llvm::StringRef(outputNode.name).drop_front(8).drop_back(1).str();
                }
                ruleNode.output = outputNode;
            }
            else {
                llvm::errs() << "Rule " << ruleNode.name
                             << " is missing 'target_gate' attribute. Skipping this rule.\n";
                continue; // skip this rule if target_gate attribute is missing
            }

            // Convert resources attribute
            if (auto resourcesAttr = func->getAttrOfType<DictionaryAttr>("resources")) {
                for (const auto &resource : resourcesAttr) {
                    RuleTerm term;
                    auto res_int = resource.getValue();
                    if (auto intAttr = mlir::dyn_cast<IntegerAttr>(res_int)) {
                        term.op.name = resource.getName().str();
                        if (term.op.name.starts_with("Adjoint(") && term.op.name.ends_with(")")) {
                            term.op.adjoint = true;
                            term.op.name =
                                llvm::StringRef(term.op.name).drop_front(8).drop_back(1).str();
                        }
                        term.multiplicity = intAttr.getInt();
                        ruleNode.inputs.push_back(term);
                    }
                    else {
                        llvm::errs()
                            << "Resource " << resource.getName() << " in rule " << ruleNode.name
                            << " has non-integer multiplicity. Skipping this resource.\n";
                        continue; // skip this resource if multiplicity is not an integer
                    }
                }
            }
            else {
                llvm::errs() << "Rule " << ruleNode.name
                             << " is missing 'resources' attribute. Skipping this rule.\n";
                continue; // skip this rule if resources attribute is missing
            }

            rules.push_back(std::move(ruleNode));
        }
        llvm::errs() << "registered rules\n";

        return;
    }

    /**
     * @brief Insert the decomposition rules picked by the graph solver into the module for later
     * use in the decompose-lowering patterns to apply the decomposition rules and rewrite the
     * quantum operations.
     *
     * @param module The MLIR module to insert the chosen decomposition rules into.
     * @param solution The chosen decomposition rules from the graph solver.
     * @param ruleNameToFuncOp A mapping from rule names to their corresponding function operations.
     */
    void insertChosenRules(
        mlir::ModuleOp module, DecompositionSolver::SolutionType &solution,
        std::unordered_map<std::string, mlir::OwningOpRef<func::FuncOp>> &ruleNameToFuncOp)
    {
        for (const auto &[_, ruleNode] : solution) {
            if (ruleNode.isBasis) {
                continue; // skip basis rules as they don't correspond to actual decomposition
                          // functions to insert
            }
            llvm::errs() << "inserting rule: " << ruleNode.ruleName << "\n";
            module.push_back(ruleNameToFuncOp.at(ruleNode.op.name).release());
        }
    };
};

} // namespace quantum
} // namespace catalyst
