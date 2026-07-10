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

#include <cstdint>
#include <numeric>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/DebugLog.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Support/WalkResult.h"
#include "stablehlo/dialect/StablehloOps.h"

#include "Catalyst/Analysis/ResourceAnalysis.h"
#include "Catalyst/Analysis/ResourceResult.h"
#include "Catalyst/Transforms/Passes.h"
#include "QRef/Transforms/Passes.h"
#include "Quantum/IR/QuantumDialect.h"
#include "Quantum/IR/QuantumInterfaces.h"
#include "Quantum/IR/QuantumOps.h"
#include "Quantum/Transforms/Passes.h"
#include "Quantum/Transforms/QPDLoader.h"

#include "DGBuilder.hpp"
#include "DGSolver.hpp"
#include "DGTypes.hpp"

#define DEBUG_TYPE "graph-decomposition"

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
        std::vector<OperatorNode> setOfOps;
        std::vector<RuleNode> setOfRules;
        llvm::StringSet<> userRuleNames;
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
        if (failed(getRuleNodes(bytecodeRulesFile, setOfRules, userRuleNames))) {
            return signalPassFailure();
        }
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
        // Step 3: Convert python-decompositions from reference to value semantics and run
        // decompose-lowering to apply the chosen decomposition rules
        ModuleOp module = getOperation();
        OpPassManager pm("builtin.module");

        DecomposeLoweringPassOptions dlOptions;
        for (auto &[op, chosenRule] : solution) {
            dlOptions.targetRulesOption.push_back(chosenRule.ruleName);
        }

        pm.addPass(qref::createValueSemanticsConversionPass());
        pm.addPass(createDecomposeLoweringPass(dlOptions));

        if (failed(runPipeline(pm, module))) {
            return signalPassFailure();
        }
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

    LogicalResult addRuleNode(mlir::func::FuncOp rule, std::vector<RuleNode> &ruleNodes)
    {
        llvm::StringRef ruleName = rule.getName();

        // 1. Mandatory Attribute Check (Target Gate and Resources)
        auto targetGateAttr = rule->getAttrOfType<StringAttr>("target_gate");
        auto resourcesAttr = rule->getAttrOfType<DictionaryAttr>("resources");
        if (!targetGateAttr) {
            llvm::errs() << "Cannot parse decomposition rule " << ruleName
                         << " without the `target_gate` attribute.\n";
            LDBG() << rule;
            return failure();
        }

        // Try to generate resources if they're missing
        if (!resourcesAttr) {
            ResourceAnalysis analysis(rule);
            if (const ResourceResult *flat = analysis.getFlattenedResource(rule.getName())) {
                rule->setAttr("resources", buildResourceDict(&getContext(), *flat));
            }
            resourcesAttr = rule->getAttrOfType<DictionaryAttr>("resources");
        }

        // Fail if resources are missing
        if (!resourcesAttr) {
            llvm::errs() << "Decomposition rule " << ruleName
                         << " was provided without resources, and resources could not be generated "
                            "for it.\n";
            return failure();
        }

        // 2. Extract 'operations' dictionary from resources
        auto operations = mlir::dyn_cast_or_null<DictionaryAttr>(resourcesAttr.get("operations"));
        if (!operations) {
            llvm::errs() << "Cannot parse resource for decomposition rule " << ruleName
                         << " without `operations` attribute.\n";
            LDBG() << rule;
            return failure();
        }

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

        // 4. Add RuleNode
        ruleNodes.push_back(std::move(ruleNode));
        return success();
    }

    LogicalResult loadBuiltInDecompositionRules(llvm::StringRef filename,
                                                std::vector<RuleNode> &ruleNodes)
    {
        mlir::MLIRContext *context = &getContext();
        mlir::ModuleOp module = getOperation();
        mlir::ParserConfig config(context);
        mlir::OwningOpRef<mlir::ModuleOp> builtinModule =
            mlir::parseSourceFile<mlir::ModuleOp>(filename, config);

        SymbolTable symbolTable(module);

        if (!builtinModule) {
            llvm::errs() << "failed to load built-in decomposition rules from '" << filename
                         << "': the rules file could not be parsed\n";
            return failure();
        }

        for (auto rule :
             llvm::make_early_inc_range(builtinModule.get().getOps<mlir::func::FuncOp>())) {
            if (failed(addRuleNode(rule, ruleNodes))) {
                return failure();
            }
            // avoid double-insertion
            if (!symbolTable.lookup<mlir::func::FuncOp>(rule.getName())) {
                rule->remove();
                module.push_back(std::move(rule));
            }
        }
        return success();
    }

    /**
     * @brief Load the listed user rules into the set of RuleNodes for the graph.
     */
    LogicalResult loadUserDecompositionRules(llvm::StringSet<> &userRuleNames,
                                             std::vector<RuleNode> &ruleNodes)
    {
        mlir::ModuleOp module = getOperation();
        if (userRuleNames.empty()) {
            return success();
        }

        WalkResult walkResult = module.walk([&](mlir::func::FuncOp func) {
            if (func->hasAttr("target_gate")) {
                if (userRuleNames.contains(func.getName())) {
                    if (failed(addRuleNode(func, ruleNodes))) {
                        return WalkResult::interrupt();
                    }
                }
            }
            return WalkResult::skip();
        });

        if (walkResult.wasInterrupted()) {
            return failure();
        }

        return success();
    }

    /**
     * @brief
     * Use python to lower decomposition rules for all unhandled decomposable operations in the
     * circuit, annotating the lowered decomposition rules with resources and target gates.
     */
    mlir::LogicalResult loadPythonDecomps(std::vector<RuleNode> &ruleNodes)
    {
        mlir::ModuleOp module = getOperation();
        MLIRContext *context = &getContext();

        llvm::StringSet<> handledOpIds;
        // Add IDs from existing decomposable ops with decomposition rules
        // NOTE: we assume in general that if one decomposition rule for an op is available, then
        // all decomposition rules for that op are available. No system should introduce a subset of
        // the rules for an op.
        module.walk([&](mlir::func::FuncOp func) {
            // TODO: generalize this
            if (func->hasAttr("target_gate")) {
                handledOpIds.insert(func->getAttrOfType<StringAttr>("target_gate").str());
            }
        });

        llvm::SmallVector<quantum::DecomposableGate> decomposableOps;
        module.walk([&](quantum::DecomposableGate op) { decomposableOps.push_back(op); });

        if (!decomposableOps.empty()) {
            if (!loadQPD(libQPDPath, libpythonPath)) {
                llvm::errs() << "failed to load libQuantumPythonCallbacks\n";
                return failure();
            }
        }

        for (quantum::DecomposableGate op : decomposableOps) {
            std::string opId = op.getGraphOpId();

            if (handledOpIds.contains(opId)) {
                continue;
            }
            handledOpIds.insert(opId);

            std::string mlirText = pythonRuleLowering(op);

            mlir::ParserConfig config(context);
            auto moduleOp = mlir::parseSourceString(llvm::StringRef(mlirText), config);
            if (!moduleOp) {
                // If we fail to parse the lowered module this op will be left without a
                // decomposition, so we must fail here.
                llvm::errs() << "failed to parse MLIR from python-decomposition\n";
                return failure();
            }

            moduleOp->walk([&](mlir::func::FuncOp func) {
                if (func.getName().starts_with(opId)) {
                    mlir::OwningOpRef<mlir::func::FuncOp> outOp;
                    func->remove();
                    outOp = mlir::OwningOpRef<mlir::func::FuncOp>(func);
                    mlir::func::FuncOp funcOp = outOp.get();
                    funcOp->setAttr("target_gate", mlir::StringAttr::get(context, opId));

                    // if we fail to add one of the decomps, we still want to try for the rest
                    std::ignore = addRuleNode(funcOp, ruleNodes);
                    LDBG() << "adding rule " << funcOp.getName();
                    module.push_back(std::move(outOp.release()));
                }
                return mlir::WalkResult::advance();
            });
        }
        return success();
    }

    bool isInDecompRule(Operation *op)
    {
        while (auto parentOp = op->getParentOp()) {
            if (auto funcOp = dyn_cast<func::FuncOp>(parentOp)) {
                if (funcOp->hasAttr("target_gate")) {
                    return true;
                }
            }
            op = parentOp;
        }
        return false;
    }

    void getOperators(std::vector<OperatorNode> &operators)
    {
        // TODO: replace this with DecomposableGate interface. We will drop support for any other op
        // types once the interface has been implemented for the core operations in the quantum
        // dialect.
        // The interface will provide one unified way of generating operator nodes from operations,
        // with consistent getter methods for all relevant data fields.
        getOperation().walk([&](quantum::QuantumGate op) {
            if (isInDecompRule(op)) {
                return;
            }
            OperatorNode node;
            node.numWires = op.getNonCtrlQubitOperands().size();
            node.adjoint = op.getAdjointFlag();

            if (auto customOp = llvm::dyn_cast<quantum::CustomOp>(op.getOperation())) {
                node.name = customOp.getGateName().str();
            }
            // Name handling for non-custom ops
            else {
                std::string name = op->getName().stripDialect().str();
                if (name == "gphase") {
                    name = "GlobalPhase";
                }
                else if (name == "paulirot") {
                    name = "paulirot" + cast<quantum::PauliRotOp>(op.getOperation()).getPauliWord();
                }
                node.name = name;
            }

            if (auto paramOp =
                    llvm::dyn_cast<catalyst::quantum::ParametrizedGate>(op.getOperation())) {
                node.numParams = paramOp.getAllParams().size();
            }
            else {
                node.numParams = 0;
            }

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
    LogicalResult getRuleNodes(llvm::StringRef filename, std::vector<RuleNode> &rules,
                               llvm::StringSet<> &userRuleNames)
    {
        // Load pre-compiled rules (ignore failure, we can try to solve without)
        std::ignore = loadBuiltInDecompositionRules(filename, rules);

        // Lower and load compile-time rules
        if (failed(loadPythonDecomps(rules))) {
            return failure();
        }

        // Load user-rules
        if (failed(loadUserDecompositionRules(userRuleNames, rules))) {
            return failure();
        }
        return success();
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
     * @param opToAltDecompNames  Parsed mapping from operator name to alternative-rule
     * names.
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
