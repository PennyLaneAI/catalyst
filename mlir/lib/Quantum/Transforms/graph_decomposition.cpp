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
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "stablehlo/dialect/StablehloOps.h"

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
        // Debugging output for command-line options
        LLVM_DEBUG(llvm::dbgs() << "Running GraphDecompositionPass with options:\n");
        LLVM_DEBUG({
            llvm::dbgs() << "targetGateSetOption\n";
            for (auto item : targetGateSetOption) {
                llvm::dbgs() << "\t" << item << ",\n";
            }
            llvm::dbgs() << "\n";

            llvm::dbgs() << "fixedDecompsOption\n";
            for (auto item : fixedDecompsOption) {
                llvm::dbgs() << "\t" << item << ",\n";
            }
            llvm::dbgs() << "\n";

            llvm::dbgs() << "altDecompsOption\n";
            for (auto item : altDecompsOption) {
                llvm::dbgs() << "\t" << item << ",\n";
            }
            llvm::dbgs() << "\n";
        });

        ///////////////////////////
        // Step 1: Gather inputs for graph
        ModuleOp module = getOperation();
        std::vector<OperatorNode> setOfOps;
        std::vector<RuleNode> setOfRules;
        llvm::StringMap<mlir::OwningOpRef<func::FuncOp>> ruleNameToFuncOp;
        llvm::StringSet<> userRuleNames;
        llvm::SmallVector<mlir::OwningOpRef<func::FuncOp>>
            allUserRules; // includes rules unused in this decomp
        llvm::StringMap<std::string> opToFixedDecompName;
        llvm::StringMap<llvm::SmallVector<std::string>> opToAltDecompNames;
        WeightedGateset targetGateSet;

        // Index rules by name for O(1) lookup instead of scanning the vector
        // for every fixed-decomp entry.
        llvm::StringMap<const RuleNode *> rulesByName(setOfRules.size());
        for (const auto &rule : setOfRules) {
            rulesByName[rule.name] = &rule;
        }

        // get names for fixed and alt decomps
        parseFixedDecomps(opToFixedDecompName, userRuleNames);
        parseAltDecomps(opToAltDecompNames, userRuleNames);
        if (failed(parseGateset(targetGateSet))) {
            return signalPassFailure();
        }

        // NOTE: getOperators must be after getRuleNodes, which removes user rules from the module.
        // This prevents operators in user rules from being added to the graph.
        getRuleNodes(bytecodeRulesFile, setOfRules, userRuleNames, allUserRules, ruleNameToFuncOp);
        getOperators(setOfOps);

        ///////////////////////////
        // Step 2: Build and solve the decomposition graph
        FixedDecomps fixedDecomps = buildFixedDecomps(opToFixedDecompName, rulesByName);
        AltDecomps altDecomps = buildAltDecomps(opToAltDecompNames, rulesByName);
        DecompositionGraph graph(setOfOps, targetGateSet, setOfRules, std::move(fixedDecomps),
                                 std::move(altDecomps));
        DecompositionSolver solver(graph);
        auto solution = solver.solve();
        ///////////////////////////
        // Step 3: Insert decomposition rules picked by the graph solver (solution) into the
        // module
        insertChosenRules(solution, ruleNameToFuncOp);

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
            module.getBody()->push_back(rule.release());
        }

        // TODO: insert a "clean-up" pass after the last graph-decomposition pass to remove user
        // functions
    }

  private:
    void parseFixedDecomps(llvm::StringMap<std::string> &opToFixedDecompName,
                           llvm::StringSet<> &userRuleNames)
    {
        for (const std::string &opRulePair : fixedDecompsOption) {
            llvm::StringRef pairRef(opRulePair);

            auto [opName, ruleName] = pairRef.split("=");
            opName = opName.trim();
            ruleName = ruleName.trim();
            ruleName.consume_front("\"");
            ruleName.consume_back("\"");

            if (ruleName.empty()) {
                continue;
            }
            opToFixedDecompName[opName.str()] = ruleName.str();
            userRuleNames.insert(ruleName.str());
        }
    }

    void parseAltDecomps(llvm::StringMap<llvm::SmallVector<std::string>> &opToAltDecompNames,
                         llvm::StringSet<> &userRuleNames)
    {
        for (const std::string &opRulesPair : altDecompsOption) {
            llvm::StringRef pairRef(opRulesPair);

            auto [opName, rulesRef] = pairRef.split("=");
            opName = opName.trim();
            llvm::SmallVector<llvm::StringRef> splitRulesRef;

            rulesRef = rulesRef.trim();
            rulesRef.consume_front("[");
            rulesRef.consume_back("]");
            rulesRef.split(splitRulesRef, ",");
            auto &opRulesList = opToAltDecompNames[opName.str()];

            for (llvm::StringRef ruleNameRef : splitRulesRef) {
                ruleNameRef = ruleNameRef.trim();
                ruleNameRef.consume_front("\"");
                ruleNameRef.consume_back("\"");
                if (!ruleNameRef.empty()) {
                    opRulesList.push_back(ruleNameRef.str());
                    userRuleNames.insert(ruleNameRef.str());
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
        llvm::StringRef filename,
        llvm::SmallVector<mlir::OwningOpRef<mlir::func::FuncOp>> &ruleRegistry)
    {
        mlir::MLIRContext *context = &getContext();
        mlir::ParserConfig config(context);
        mlir::OwningOpRef<mlir::ModuleOp> moduleOp =
            mlir::parseSourceFile<mlir::ModuleOp>(filename, config);

        if (!moduleOp) {
            mlir::emitError(mlir::UnknownLoc::get(context))
                << "failed to load built-in decomposition rules from '" << filename
                << "': the rules file could not be parsed";
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
    LogicalResult
    loadUserDecompositionRules(llvm::StringSet<> &userRuleNames,
                               llvm::SmallVector<mlir::OwningOpRef<mlir::func::FuncOp>> &graphRules,
                               llvm::SmallVector<mlir::OwningOpRef<mlir::func::FuncOp>> &rules)
    {
        mlir::ModuleOp module = getOperation();
        if (userRuleNames.empty()) {
            return success();
        }

        PassManager pm(&getContext());
        pm.addPass(createRegisterDecompRuleResourcePass());
        if (failed(pm.run(module))) {
            module.emitError() << "failed to load user decomposition rules: unable to run resource "
                                  "annotation pass";
            return failure();
        }

        llvm::SmallVector<mlir::func::FuncOp> userRules;

        module.walk([&](mlir::func::FuncOp func) {
            if (func->hasAttr("target_gate")) {
                userRules.push_back(func);
                if (userRuleNames.contains(func.getName())) {
                    graphRules.push_back(mlir::OwningOpRef<mlir::func::FuncOp>(func.clone()));
                }
            }
            return WalkResult::skip();
        });

        for (auto rule : llvm::make_early_inc_range(userRules)) {
            rule->remove();
            rules.push_back(mlir::OwningOpRef<mlir::func::FuncOp>(rule));
        }
        return success();
    }

    void getOperators(std::vector<OperatorNode> &operators)
    {
        getOperation().walk([&](CustomOp op) {
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
     * @brief Helper to parse a gate name into an OperatorNode.
     * Handles patterns like "Adjoint(GateName)" and "GateName(metadata)".
     */
    OperatorNode parseOperator(llvm::StringRef raw)
    {
        OperatorNode node;

        // Unwrap "Adjoint(GateName)"
        if (raw.consume_front("Adjoint(")) {
            node.adjoint = true;
            auto closeIdx = raw.rfind(')');
            if (closeIdx == llvm::StringRef::npos) {
                node.name = raw.trim().str();
                return node;
            }
            node.name = raw.take_front(closeIdx).trim().str();
            raw = raw.drop_front(closeIdx + 1); // leftover: "(w,p)" or ""
        }
        else {
            auto openIdx = raw.find('(');
            if (openIdx == llvm::StringRef::npos) {
                node.name = raw.trim().str();
                return node;
            }
            node.name = raw.take_front(openIdx).trim().str();
            raw = raw.drop_front(openIdx); // leftover: "(w,p)" or "(w)"
        }

        // Parse "(w,p)" (new) or "(w)" (legacy) suffix.
        if (raw.consume_front("(") && raw.consume_back(")")) {
            llvm::StringRef wStr, pStr;
            std::tie(wStr, pStr) = raw.split(',');
            int w = -1, p = -1;
            if (!wStr.getAsInteger(10, w)) {
                node.numWires = w;
            }
            if (!pStr.empty() && !pStr.getAsInteger(10, p)) {
                node.numParams = p;
            }
            // If pStr is empty we were given the legacy "(w)" format; leave
            // numParams at the wildcard default so old bytecode keeps working.
        }

        return node;
    }

    /**
     * @brief Create RuleNodes for each rule available to be used in graph decomposition.
     */
    void getRuleNodes(llvm::StringRef filename, std::vector<RuleNode> &rules,
                      llvm::StringSet<> &userRuleNames,
                      llvm::SmallVector<mlir::OwningOpRef<func::FuncOp>> &userRules,
                      llvm::StringMap<mlir::OwningOpRef<func::FuncOp>> &ruleNameToFuncOp)
    {
        llvm::SmallVector<mlir::OwningOpRef<mlir::func::FuncOp>> graphRules;

        // Load rules from bytecode and user-defined passes
        loadBuiltInDecompositionRules(filename, graphRules);
        if (failed(loadUserDecompositionRules(userRuleNames, graphRules, userRules))) {
            return signalPassFailure();
        }

        for (auto &ruleOpRef : graphRules) {
            mlir::func::FuncOp func = ruleOpRef.get();
            llvm::StringRef ruleName = func.getName();

            // 1. Mandatory Attribute Check (Target Gate and Resources)
            auto targetGateAttr = func->getAttrOfType<StringAttr>("target_gate");
            auto resourcesAttr = func->getAttrOfType<DictionaryAttr>("resources");
            if (!targetGateAttr || !resourcesAttr)
                continue;

            // 2. Extract 'operations' dictionary from resources
            auto operations =
                mlir::dyn_cast_or_null<DictionaryAttr>(resourcesAttr.get("operations"));
            if (!operations)
                continue;

            // 3. Populate RuleNode
            RuleNode ruleNode;
            ruleNode.name = ruleName.str();
            ruleNode.output = parseOperator(targetGateAttr.getValue());

            for (const auto &namedAttr : operations) {
                if (auto intAttr = mlir::dyn_cast<IntegerAttr>(namedAttr.getValue())) {
                    ruleNode.inputs.push_back({parseOperator(namedAttr.getName().strref()),
                                               static_cast<uint32_t>(intAttr.getInt())});
                }
            }

            // 4. Finalize: move the OpRef to the map to keep IR alive and store the node
            ruleNameToFuncOp[ruleNode.name] = std::move(ruleOpRef);
            rules.push_back(std::move(ruleNode));
        }
    }

    /**
     * @brief Insert the decomposition rules picked by the graph solver into the module for
     * later use in the decompose-lowering patterns to apply the decomposition rules and rewrite
     * the quantum operations.
     *
     * @param solution The chosen decomposition rules from the graph solver.
     * @param ruleNameToFuncOp A mapping from rule names to their corresponding function
     * operations.
     */
    void insertChosenRules(GraphResult &solution,
                           llvm::StringMap<mlir::OwningOpRef<func::FuncOp>> &ruleNameToFuncOp)
    {
        mlir::ModuleOp module = getOperation();
        for (const auto &[_, chosenRule] : solution) {
            if (chosenRule.isBasis) {
                continue; // skip basis rules as they don't correspond to actual decomposition
                          // functions to insert
            }
            auto it = ruleNameToFuncOp.find(chosenRule.ruleName);

            if (it == ruleNameToFuncOp.end() || !it->second) {
                // skip if the rule is not found or
                // the function op is null or
                // it is already moved
                continue;
            }
            module.push_back(it->second.release());
        }
    }

    /**
     * @brief Convert the parsed fixed-decomposition mapping (op name → rule name)
     * into the Core::FixedDecomps type expected by the DecompositionGraph.
     *
     * For each entry, looks up the corresponding RuleNode in setOfRules by name.
     * Rules not found in setOfRules are skipped with a diagnostic.
     *
     * @param opToFixedDecompName  Parsed mapping from operator name to fixed-rule name.
     * @param setOfRules           The full list of available decomposition rules.
     * @return Core::FixedDecomps  Mapping from OperatorNode to its fixed RuleNode.
     */
    FixedDecomps buildFixedDecomps(const llvm::StringMap<std::string> &opToFixedDecompName,
                                   const llvm::StringMap<const RuleNode *> &rulesByName)
    {
        FixedDecomps fixedDecomps;
        fixedDecomps.reserve(opToFixedDecompName.size());

        for (const auto &[opName, ruleName] : opToFixedDecompName) {
            auto it = rulesByName.find(ruleName);
            if (it == rulesByName.end()) {
                continue;
            }

            OperatorNode opNode;
            opNode.name = opName.str();
            fixedDecomps.emplace(std::move(opNode), *(it->second));
        }
        return fixedDecomps;
    }

    /**
     * @brief Convert the parsed alternative-decomposition mapping
     * (op name → list of rule names) into the Core::AltDecomps type
     * expected by the DecompositionGraph.
     *
     * For each entry, looks up the corresponding RuleNodes in setOfRules by name.
     * Individual rules not found are skipped with a diagnostic.
     *
     * @param opToAltDecompNames  Parsed mapping from operator name to alternative-rule names.
     * @param setOfRules          The full list of available decomposition rules.
     * @return Core::AltDecomps   Mapping from OperatorNode to its alternative RuleNodes.
     */
    AltDecomps
    buildAltDecomps(const llvm::StringMap<llvm::SmallVector<std::string>> &opToAltDecompNames,
                    const llvm::StringMap<const RuleNode *> &rulesByName)
    {
        AltDecomps altDecomps;
        altDecomps.reserve(opToAltDecompNames.size());

        for (const auto &[opName, ruleNames] : opToAltDecompNames) {
            OperatorNode opNode;
            opNode.name = opName.str();

            std::vector<RuleNode> altRules;
            altRules.reserve(ruleNames.size());

            for (const auto &ruleName : ruleNames) {
                auto it = rulesByName.find(ruleName);
                if (it == rulesByName.end()) {
                    continue;
                }
                altRules.push_back(*(it->second));
            }

            if (!altRules.empty()) {
                altDecomps.emplace(std::move(opNode), std::move(altRules));
            }
        }
        return altDecomps;
    }
};

} // namespace quantum
} // namespace catalyst
