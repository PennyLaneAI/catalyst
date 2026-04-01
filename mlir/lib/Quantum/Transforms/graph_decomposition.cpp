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
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Parser/Parser.h"
#include "stablehlo/dialect/StablehloOps.h"

#include "mlir/Pass/PassManager.h"

#include "Catalyst/Transforms/Passes.h"
#include "Quantum/IR/QuantumDialect.h"
#include "Quantum/IR/QuantumOps.h"
#include "Quantum/Transforms/Passes.h"

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
        ///////////////////////////
        // Step 1: Gather inputs for graph
        std::vector<OperatorNode> setOfOps;
        std::vector<RuleNode> setOfRules;
        llvm::StringMap<mlir::OwningOpRef<func::FuncOp>> ruleNameToFuncOp;
        llvm::SmallVector<std::string> userRuleNames;
        llvm::SmallVector<mlir::OwningOpRef<func::FuncOp>>
            allUserRules; // includes rules unused in this decomp
        llvm::StringMap<std::string> opToFixedDecompName;
        llvm::StringMap<llvm::SmallVector<std::string>> opToAltDecompNames;
        WeightedGateset targetGateSet;

        ModuleOp module = cast<ModuleOp>(getOperation());
        // get names for fixed and alt decomps
        parseFixedDecomps(opToFixedDecompName, userRuleNames);
        parseAltDecomps(opToAltDecompNames, userRuleNames);
        if (failed(parseGateset(targetGateSet))) {
            return signalPassFailure();
        }

        // NOTE: getOperators must be after getRuleNodes, which removes user rules from the module.
        // This prevents operators in user rules from being added to the graph.
        getRuleNodes(module, bytecodeRulesFile, setOfRules, userRuleNames, allUserRules,
                     ruleNameToFuncOp);
        getOperators(module, setOfOps);

        ///////////////////////////
        // Step 2: Build and solve the decomposition graph
        DecompositionGraph graph(setOfOps, targetGateSet, setOfRules);
        DecompositionSolver solver(graph);
        auto solution = solver.getSolvedMap();

        ///////////////////////////
        // Step 3: Insert decomposition rules picked by the graph solver (solution) into the
        // module
        insertChosenRules(module, solution, ruleNameToFuncOp);

        ///////////////////////////
        // Step 4: Run decompose-lowering to apply the decomposition rules
        PassManager pm(&getContext());
        pm.addPass(createDecomposeLoweringPass());

        if (failed(pm.run(module))) {
            return signalPassFailure();
        }

        ///////////////////////////
        // Step 5: Re-introduce (all) user rules for future decompositions
        for (auto &rule : allUserRules) {
            llvm::errs() << "re-adding user rules " << rule.get().getName() << "\n";
            module.getBody()->push_back(rule.release());
        }
    }

  private:
    void parseFixedDecomps(llvm::StringMap<std::string> &opToFixedDecompName,
                           llvm::SmallVectorImpl<std::string> &userRuleNames)
    {
        for (const std::string &opRulePair : fixedDecompsOption) {
            llvm::StringRef pairRef(opRulePair);

            auto [opName, ruleName] = pairRef.split("=");
            opName = opName.trim();
            ruleName = ruleName.trim();

            if (ruleName.empty()) {
                llvm::errs() << opName
                             << " was given in fixed-decomps, but no rule was specified. Skipping "
                                "this rule.\n";
                return;
            }
            llvm::errs() << "adding fixed decomp " << ruleName << " for " << opName << "\n";
            opToFixedDecompName[opName.str()] = ruleName.str();
            userRuleNames.push_back(ruleName.str());
        }
    }

    void parseAltDecomps(llvm::StringMap<llvm::SmallVector<std::string>> &opToAltDecompNames,
                         llvm::SmallVectorImpl<std::string> &userRuleNames)
    {
        for (const std::string &opRulesPair : altDecompsOption) {
            llvm::StringRef pairRef(opRulesPair);

            auto [opName, rulesRef] = pairRef.split("=");
            opName = opName.trim();
            llvm::SmallVector<llvm::StringRef> splitRulesRef;

            rulesRef.split(splitRulesRef, "|");
            auto &opRulesList = opToAltDecompNames[opName.str()];

            for (llvm::StringRef ruleNameRef : splitRulesRef) {
                ruleNameRef = ruleNameRef.trim();
                if (!ruleNameRef.empty()) {
                    opRulesList.push_back(ruleNameRef.str());
                    userRuleNames.push_back(ruleNameRef.str());
                    llvm::errs() << "adding alt decomp " << ruleNameRef << " for " << opName
                                 << "\n";
                }
            }
        }
    }

    LogicalResult parseGateset(WeightedGateset &targetGateSet)
    {
        for (const std::string &opCostPair : targetGateSetOption) {
            llvm::StringRef pairRef(opCostPair);

            auto [opNameRaw, costRaw] = pairRef.split("=");
            llvm::StringRef opName = opNameRaw.trim();
            llvm::StringRef cost = costRaw.trim();

            cost.consume_back(": f64");
            cost = cost.trim();

            bool success = to_float(cost, targetGateSet.ops[OperatorNode{opName.str()}]);

            if (!success) {
                return failure();
            }
        }
        return success();
    }

    void loadBuiltInDecompositionRules(
        llvm::StringRef filename, mlir::MLIRContext *context,
        llvm::SmallVector<mlir::OwningOpRef<mlir::func::FuncOp>> &ruleRegistry)
    {
        mlir::ParserConfig config(context);
        mlir::OwningOpRef<mlir::ModuleOp> moduleOp =
            mlir::parseSourceFile<mlir::ModuleOp>(filename, config);

        if (!moduleOp) {
            llvm::errs() << "failed to find module\n";
            return;
        }

        for (auto rule : llvm::make_early_inc_range(moduleOp.get().getOps<mlir::func::FuncOp>())) {
            rule->remove();
            ruleRegistry.push_back(mlir::OwningOpRef<mlir::func::FuncOp>(rule));
        }
        return;
    }

    /**
     * @brief Remove user rules from the module, loading into
     */
    void
    loadUserDecompositionRules(ModuleOp module, llvm::SmallVector<std::string> &userRuleNames,
                               llvm::SmallVector<mlir::OwningOpRef<mlir::func::FuncOp>> &graphRules,
                               llvm::SmallVector<mlir::OwningOpRef<mlir::func::FuncOp>> &rules)
    {
        if (userRuleNames.empty()) {
            return;
        }

        PassManager pm(&getContext());
        pm.addPass(createRegisterDecompRuleResourcePass());
        if (failed(pm.run(module))) {
            return signalPassFailure();
        }

        llvm::SmallVector<mlir::func::FuncOp> userRules;

        llvm::errs() << "scanning for user rules\n";

        module.walk([&](mlir::func::FuncOp func) {
            llvm::errs() << "walking module found function with name " << func.getName() << "\n";
            if (func->hasAttr("target_gate")) {
                userRules.push_back(func);
                if (std::find(userRuleNames.begin(), userRuleNames.end(), func.getName()) !=
                    userRuleNames.end()) {
                    graphRules.push_back(mlir::OwningOpRef<mlir::func::FuncOp>(func.clone()));
                    llvm::errs() << "found user rule with name " << func.getName() << "\n";
                }
            }
            return WalkResult::skip();
        });

        for (auto rule : llvm::make_early_inc_range(userRules)) {
            rule->remove();
            rules.push_back(mlir::OwningOpRef<mlir::func::FuncOp>(rule));
        }
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

    /**
     * @brief create RuleNodes for each rule available to be used in graph decomposition. These
     * rules are added to the rules parameter.
     */
    void getRuleNodes(ModuleOp module, llvm::StringRef filename, std::vector<RuleNode> &rules,
                      llvm::SmallVector<std::string> &userRuleNames,
                      llvm::SmallVector<mlir::OwningOpRef<func::FuncOp>> &userRules,
                      llvm::StringMap<mlir::OwningOpRef<func::FuncOp>> &ruleNameToFuncOp)
    {
        llvm::SmallVector<mlir::OwningOpRef<mlir::func::FuncOp>> graphRules;
        loadBuiltInDecompositionRules(filename, module.getContext(), graphRules);
        loadUserDecompositionRules(module, userRuleNames, graphRules, userRules);

        llvm::errs() << "writing rules\n";

        for (auto &ruleOpRef : graphRules) {
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
                continue; // skip this rule if target_gate attribute is missing
            }

            // Convert resources attribute
            if (auto resourcesAttr = func->getAttrOfType<DictionaryAttr>("resources")) {
                DictionaryAttr operations =
                    mlir::dyn_cast<DictionaryAttr>(resourcesAttr.get("operations"));
                for (const auto &operation : operations) {
                    RuleTerm term;
                    auto res_int = operation.getValue();
                    if (auto intAttr = mlir::dyn_cast<IntegerAttr>(res_int)) {
                        term.op.name = operation.getName().str();
                        if (term.op.name.find("(") != std::string::npos) {
                            term.op.name = term.op.name.substr(0, term.op.name.find("("));
                        }
                        term.multiplicity = intAttr.getInt();
                        ruleNode.inputs.push_back(term);
                    }
                    else {
                        llvm::errs() << "Resource " << operation.getName() << " in rule "
                                     << ruleNode.name << " has non-integer multiplicity "
                                     << operation.getValue() << ". Skipping this resource.\n";
                        continue; // skip this resource if multiplicity is not an integer
                    }
                }
            }
            else {
                llvm::errs() << "Rule " << ruleNode.name
                             << " is missing 'resources' attribute. Skipping this rule.\n";
                continue; // skip this rule if resources attribute is missing
            }
            llvm::errs() << ruleNode.name << " for " << ruleNode.output.name << ", ";
            rules.push_back(std::move(ruleNode));
        }
        llvm::errs() << "\nregistered rules\n";

        return;
    }

    /**
     * @brief Insert the decomposition rules picked by the graph solver into the module for
     * later use in the decompose-lowering patterns to apply the decomposition rules and rewrite
     * the quantum operations.
     *
     * @param module The MLIR module to insert the chosen decomposition rules into.
     * @param solution The chosen decomposition rules from the graph solver.
     * @param ruleNameToFuncOp A mapping from rule names to their corresponding function
     * operations.
     */
    void insertChosenRules(mlir::ModuleOp module, DecompositionSolver::SolutionType &solution,
                           llvm::StringMap<mlir::OwningOpRef<func::FuncOp>> &ruleNameToFuncOp)
    {
        for (const auto &[_, chosenRule] : solution) {
            if (chosenRule.isBasis) {
                continue; // skip basis rules as they don't correspond to actual decomposition
                          // functions to insert
            }
            llvm::errs() << "inserting rule: " << chosenRule.ruleName << " for op "
                         << chosenRule.op.name << "\n";

            auto it = ruleNameToFuncOp.find(chosenRule.ruleName);

            if (it == ruleNameToFuncOp.end()) {
                llvm::errs() << "Rule " << chosenRule.ruleName << " not found\n";
            }
            if (!it->second) {
                llvm::errs() << "Rule " << chosenRule.ruleName << " has already been added!\n";
            }
            module.push_back(it->second.release());
        }
    };
};

} // namespace quantum
} // namespace catalyst
