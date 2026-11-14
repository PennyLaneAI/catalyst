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

#include "GraphPrep.hpp"

#include <cmath>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/DebugLog.h"
#include "llvm/Support/LogicalResult.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/AsmParser/AsmParser.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Support/WalkResult.h"

#include "Catalyst/Analysis/ResourceAnalysis.h"
#include "Catalyst/Analysis/ResourceResult.h"
#include "Quantum/IR/QuantumInterfaces.h"
#include "Quantum/Transforms/QPDLoader.h"

#include "DGBuilder.hpp"
#include "DGTypes.hpp"
#include "DecompUtils.hpp"

#define DEBUG_TYPE "graph-decomposition"

using namespace mlir;
using namespace catalyst::quantum;
using namespace DecompGraph::Core;
using namespace DecompGraph::Solver;

namespace {

void parseFixedDecomps(ArrayRef<std::string> fixedDecompsOption,
                       llvm::StringMap<std::string> &opToFixedDecompName) {
    for (const std::string &opRulePair : fixedDecompsOption) {
        llvm::StringRef pairRef(opRulePair);

        auto [opName, ruleName] = pairRef.split("=");
        opName = opName.trim();
        opName.consume_front("\"");
        opName.consume_back("\"");
        ruleName = ruleName.trim();
        ruleName.consume_front("\"");
        ruleName.consume_back("\"");

        if (ruleName.empty()) {
            continue;
        }
        opToFixedDecompName[opName.str()] = ruleName.str();
    }
}

void parseExcludedDecomps(ArrayRef<std::string> excludeDecompsOption,
                          llvm::StringMap<llvm::StringSet<>> &opToExcludedDecompNames) {
    for (const std::string &opRulesPair : excludeDecompsOption) {
        llvm::StringRef pairRef(opRulesPair);

        auto [opName, rulesRef] = pairRef.split("=");
        opName = opName.trim();
        opName.consume_front("\"");
        opName.consume_back("\"");
        llvm::SmallVector<llvm::StringRef> splitRulesRef;

        rulesRef = rulesRef.trim();
        rulesRef.consume_front("[");
        rulesRef.consume_back("]");
        rulesRef.split(splitRulesRef, ",");
        auto &opRulesList = opToExcludedDecompNames[opName.str()];

        for (llvm::StringRef ruleNameRef : splitRulesRef) {
            ruleNameRef = ruleNameRef.trim();
            ruleNameRef.consume_front("\"");
            ruleNameRef.consume_back("\"");
            if (!ruleNameRef.empty()) {
                opRulesList.insert(ruleNameRef.str());
            }
        }
    }
}

LogicalResult parseGateset(ArrayRef<std::string> targetGateSetOption,
                           WeightedGateset &targetGateSet, ModuleOp module) {
    for (const std::string &opCostPair : targetGateSetOption) {
        llvm::StringRef pairRef(opCostPair);

        auto [opNameRaw, costRaw] = pairRef.split("=");
        llvm::StringRef opName = opNameRaw.trim();
        llvm::StringRef cost = costRaw.trim();

        // Note gate_set is now a DictionaryAttr which quotes any key that is
        // not an MLIR op (e.g. "Adjoint(TemporaryAND)").
        // As the result, we need to strip the surrounding quotes so the stored name
        // matches the op's graphOpId name following parseFixedDecomps / parseExcludedDecomps.
        opName.consume_front("\"");
        opName.consume_back("\"");

        double weight = 0.0;
        bool parsed = false;

        if (cost.ends_with(": f64")) {
            // Float attributes encode NaN and infinity as hexadecimal bit patterns.
            // Parsing these with strtod would instead produce finite, positive costs.
            if (auto attr =
                    dyn_cast_or_null<FloatAttr>(mlir::parseAttribute(cost, module.getContext()))) {
                weight = attr.getValueAsDouble();
                parsed = true;
            }

        } else { // fallback string parsing
            parsed = !cost.empty() && to_float(cost, weight);
        }

        // validation
        if (!parsed || std::isnan(weight) || weight < 0.0) {
            return module.emitError()
                   << "Invalid cost '" << cost << "' for gate '" << opName
                   << "' in the target gateset: expected a non-negative number (not NaN).";
        }
        targetGateSet.ops[opName.str()] = weight;
    }
    return success();
}

/**
 * @brief Parse a graphOpId string into an OperatorNode.
 *
 * The graphOpId format is "<name>{params}{wires}{static}[uid]", where <name> already
 * carries any name-wrapped op-level modifiers produced by `defaultGetGraphOpId`, e.g.
 * "C(Adjoint(RX)){0:[f64]}{wires:1}{}".
 */
OperatorNode parseOperator(llvm::StringRef id) {
    OperatorNode node;

    node.id = id.str();
    node.name = id.take_until([](char c) { return c == '[' || c == '{'; }).drop_until([](char c) {
        return !llvm::isDigit(c);
    });

    return node;
}

LogicalResult addRuleNode(func::FuncOp rule, std::vector<RuleNode> &ruleNodes) {
    llvm::StringRef ruleName = rule.getName();

    // 1. Mandatory Attribute Check (Target Gate and Resources)
    auto targetGateAttr = rule->getAttrOfType<StringAttr>(DecompUtils::target_gate_attr_name);
    auto resourcesAttr = rule->getAttrOfType<DictionaryAttr>("resources");
    if (!targetGateAttr) {
        llvm::errs() << "Cannot parse decomposition rule " << ruleName
                     << " without the `target_gate` attribute.\n";
        LDBG() << rule;
        return failure();
    }

    // Try to generate resources if they're missing
    if (!resourcesAttr) {
        catalyst::ResourceAnalysis analysis(rule, {}, /*collectDetailedOperations=*/true);
        if (const catalyst::ResourceResult *flat = analysis.getFlattenedResource(rule.getName())) {
            rule->setAttr("resources", buildResourceDict(rule.getContext(), *flat));
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
    auto operations = dyn_cast_or_null<DictionaryAttr>(resourcesAttr.get("operations"));
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
        if (auto intAttr = dyn_cast<IntegerAttr>(namedAttr.getValue())) {
            ruleNode.inputs.push_back({parseOperator(namedAttr.getName().strref()),
                                       static_cast<uint32_t>(intAttr.getInt())});
        }
    }

    // 4. Add RuleNode
    ruleNodes.push_back(std::move(ruleNode));
    return success();
}

OwningOpRef<ModuleOp> loadPrecompiledRules(ModuleOp module, llvm::StringRef filename) {
    ParserConfig config(module.getContext());
    OwningOpRef<ModuleOp> precompiledModule = parseSourceFile<ModuleOp>(filename, config);

    SymbolTable symbolTable(module);

    if (!precompiledModule) {
        llvm::errs() << "failed to load built-in decomposition rules from '" << filename
                     << "': the rules file could not be parsed\n";
        return {};
    }

    SymbolTable precompiledSymbolTable(precompiledModule.get());
    llvm::SmallVector<func::FuncOp> worklist;
    for (auto rule : llvm::make_early_inc_range(precompiledModule.get().getOps<func::FuncOp>())) {
        if (!rule->hasAttr(DecompUtils::target_gate_attr_name)) {
            continue;
        }
        auto existing = symbolTable.lookup<func::FuncOp>(rule.getName());
        if (existing && existing->getAttr("target_gate") == rule->getAttr("target_gate") &&
            existing->getAttr("frontend_name") == rule->getAttr("frontend_name") &&
            existing->getAttr("resources") == rule->getAttr("resources")) {
            // References to a reused rule will resolve to the existing circuit symbol when
            // dependent functions are materialized.
            precompiledSymbolTable.erase(rule);
            continue;
        }
        worklist.push_back(rule);
    }

    // Prepare rule functions and their dependencies, not the library's setup, teardown or
    // capture entry point. Helpers with the same name may have different signatures.
    llvm::SmallPtrSet<Operation *, 16> required;
    while (!worklist.empty()) {
        func::FuncOp function = worklist.pop_back_val();
        if (!required.insert(function).second) {
            continue;
        }
        auto uses = SymbolTable::getSymbolUses(function);
        if (!uses) {
            function.emitError("cannot inspect precompiled rule dependencies");
            return {};
        }
        for (const auto &use : *uses) {
            if (auto dependency = dyn_cast_or_null<func::FuncOp>(
                    SymbolTable::lookupSymbolIn(precompiledModule.get(), use.getSymbolRef()))) {
                worklist.push_back(dependency);
            }
        }
    }

    llvm::SmallVector<func::FuncOp> functions;
    for (auto function : precompiledModule.get().getOps<func::FuncOp>()) {
        if (required.contains(function)) {
            functions.push_back(function);
        }
    }

    // Rename inside the source symbol table so every call is updated before moving functions.
    for (auto function : functions) {
        if (symbolTable.lookup(function.getName()) &&
            failed(precompiledSymbolTable.renameToUnique(function, {&symbolTable}))) {
            function.emitError("cannot rename a precompiled rule dependency");
            return {};
        }
    }
    return precompiledModule;
}

/**
 * @brief Load the listed user rules into the set of RuleNodes for the graph.
 */
LogicalResult loadDecompositionRules(ModuleOp module,
                                     llvm::StringMap<std::string> &opToFixedDecompName,
                                     llvm::StringMap<llvm::StringSet<>> &opToExcludedDecompNames,
                                     std::vector<RuleNode> &ruleNodes) {

    WalkResult walkResult = module.walk([&](mlir::func::FuncOp func) {
        if (func->hasAttr(DecompUtils::target_gate_attr_name)) {
            // TODO: remove GOID inspection
            llvm::StringRef targetGate =
                func->getAttrOfType<mlir::StringAttr>(DecompUtils::target_gate_attr_name)
                    .getValue()
                    .take_until([](char c) { return c == '{'; }) // remove GOID data
                    .drop_while(llvm::isDigit);                  // remove <n>C numeric prefix

            auto fixed = opToFixedDecompName.find(targetGate);
            auto excluded = opToExcludedDecompNames.find(targetGate);
            bool hasExclusions =
                excluded != opToExcludedDecompNames.end() && !excluded->second.empty();
            if (fixed != opToFixedDecompName.end() || hasExclusions) {
                // Match frontend names, since lowering may rename the function symbol.
                auto nameAttr = func->getAttrOfType<mlir::StringAttr>("frontend_name");
                if (!nameAttr) {
                    func.emitError()
                        << "decomposition rule targets a gate with fixed/exclude-decomps, but "
                           "doesn't have the `frontend_name` attribute.";
                    return WalkResult::interrupt();
                }
                llvm::StringRef funcName = nameAttr.getValue();
                if ((hasExclusions && excluded->second.contains(funcName)) ||
                    (fixed != opToFixedDecompName.end() && fixed->second != funcName)) {
                    LDBG() << "Skipping decomposition rule " << func.getName()
                           << " due to fixed/exclude-decomps";
                    // Keep the function in the module for subsequent passes.
                    return WalkResult::skip();
                }
            }

            // standard case - targets an op with no restrictions
            if (failed(addRuleNode(func, ruleNodes))) {
                return WalkResult::interrupt();
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
 *
 * This only *materializes* the lowered rule funcs (as `__builtin`-prefixed funcs) into the
 * module; `loadUserDecompositionRules`, which runs afterwards, is what registers them as
 * RuleNodes. It therefore takes no `ruleNodes` output.
 */
[[maybe_unused]] LogicalResult loadPythonDecomps(ModuleOp module, llvm::StringRef libQPDPath,
                                                 llvm::StringRef libpythonPath) {
    MLIRContext *context = module.getContext();
    SymbolTable symbolTable(module);

    llvm::StringSet<> handledOpIds;
    // Add IDs from existing decomposable ops with decomposition rules
    // NOTE: we assume in general that if one decomposition rule for an op is available,
    // then all decomposition rules for that op are available. No system should introduce a
    // subset of the rules for an op.
    module.walk([&](mlir::func::FuncOp func) {
        if (func->hasAttr("target_gate")) {
            handledOpIds.insert(func->getAttrOfType<StringAttr>("target_gate").str());
        }
    });

    llvm::SmallVector<catalyst::quantum::DecomposableGate> decomposableOps;
    module.walk([&](catalyst::quantum::DecomposableGate op) {
        if (!DecompUtils::isInDecompRule(op)) {
            decomposableOps.push_back(op);
        }
    });

    if (!decomposableOps.empty()) {
        if (!loadQPD(libQPDPath.str(), libpythonPath.str())) {
            llvm::errs() << "failed to load libQuantumPythonCallbacks\n";
            return failure();
        }
    }

    for (catalyst::quantum::DecomposableGate op : decomposableOps) {
        std::string opId = op.getGraphOpId();

        if (handledOpIds.contains(opId)) {
            continue;
        }

        std::string mlirText = pythonRuleLowering(op);

        mlir::ParserConfig config(context);
        auto moduleOp = mlir::parseSourceString(llvm::StringRef(mlirText), config);
        if (!moduleOp) {
            // If we fail to parse the lowered module this op will be left without a
            // decomposition, so we must fail here.
            llvm::errs() << "failed to parse MLIR from python-decomposition\n";
            return failure();
        }

        // The Python wrapper returns the *whole reachable rule closure* for this op, not
        // just its direct rules: the loader does not recurse into a rule's resource ops, so
        // every rule on the path down to the gate set (e.g. `Adjoint(S)` ->
        // `Adjoint(PhaseShift)` -> `PhaseShift`) must arrive together. We only
        // *materialize* each rule func (named
        // `__builtin_...`) into the module here; `loadUserDecompositionRules` runs next and
        // is the single place that turns `__builtin`-prefixed funcs into RuleNodes (the
        // same path the eager frontend relies on for its pre-embedded rules). We must NOT
        // also call `addRuleNode` here, or every on-demand rule would be registered twice.
        // We still skip helper funcs (no `target_gate`) and any target already materialized
        // by an earlier op's closure or embedded in the module, to avoid duplicate symbols.
        llvm::SmallVector<llvm::StringRef> newlyHandled;
        moduleOp->walk([&](mlir::func::FuncOp func) {
            // Note: a single target gate may have several alternative rules, so we only
            // mark a target handled *after* walking the whole module, otherwise the second
            // alternative would be dropped.
            auto targetGate = func->getAttrOfType<StringAttr>("target_gate");
            if (!targetGate || handledOpIds.contains(targetGate.getValue())) {
                return mlir::WalkResult::advance();
            }
            mlir::OwningOpRef<mlir::func::FuncOp> outOp;
            func->remove();
            outOp = mlir::OwningOpRef<mlir::func::FuncOp>(func);
            LDBG() << "materializing rule " << outOp.get().getName();
            newlyHandled.push_back(targetGate.getValue());
            // Resolve name collisions with rules already embedded in the module.
            symbolTable.insert(outOp.release());
            return mlir::WalkResult::advance();
        });
        for (llvm::StringRef target : newlyHandled) {
            handledOpIds.insert(target);
        }
        // Mark this op handled even if its closure produced no rule (e.g. no decomposition
        // exists), so repeated instances of the same gate do not re-invoke the Python
        // loader.
        handledOpIds.insert(opId);
    }
    return success();
}

/**
 * @brief Create RuleNodes for each rule available to be used in graph decomposition.
 */
LogicalResult getRuleNodes(ModuleOp module, llvm::StringRef filename,
                           llvm::StringMap<std::string> &opToFixedDecompName,
                           llvm::StringMap<llvm::StringSet<>> &opToExcludedDecompNames,
                           [[maybe_unused]] llvm::StringRef libQPDPath,
                           [[maybe_unused]] llvm::StringRef libpythonPath,
                           std::vector<RuleNode> &rules, OwningOpRef<ModuleOp> &ruleLibrary) {

    // Read library candidates separately from the circuit. Ignore a missing library: rules
    // already captured in the circuit may still be sufficient to solve the graph.
    if (!filename.empty()) {
        ruleLibrary = loadPrecompiledRules(module, filename);
    }

    // Lower compile-time rules into the module
    // TODO: re-enable once the frontend for this has been updated
    // These need to be removed for now since they result in symbol conflicts and we have no
    // simple way of fixing this.
    // if (failed(loadPythonDecomps(module, libQPDPath, libpythonPath))) {
    //     return failure();
    // }

    // Load rules from the module into the set of rules used by the graph, filtering by
    // fixed- and exclude-decomps
    if (failed(
            loadDecompositionRules(module, opToFixedDecompName, opToExcludedDecompNames, rules))) {
        return failure();
    }
    if (ruleLibrary && failed(loadDecompositionRules(ruleLibrary.get(), opToFixedDecompName,
                                                     opToExcludedDecompNames, rules))) {
        return failure();
    }
    return success();
}

} // namespace

namespace catalyst {
namespace quantum {
namespace GraphPrep {

LogicalResult materializeSelectedLibraryRules(ModuleOp module, ModuleOp ruleLibrary,
                                              const llvm::StringSet<> &selectedRules) {
    if (!ruleLibrary) {
        return success();
    }

    llvm::SmallVector<func::FuncOp> worklist;
    for (auto function : ruleLibrary.getOps<func::FuncOp>()) {
        if (selectedRules.contains(function.getName())) {
            worklist.push_back(function);
        }
    }
    llvm::SmallPtrSet<Operation *, 16> required;
    while (!worklist.empty()) {
        auto function = worklist.pop_back_val();
        if (!required.insert(function).second) {
            continue;
        }
        auto uses = SymbolTable::getSymbolUses(function);
        if (!uses) {
            return function.emitError("cannot inspect selected rule dependencies");
        }
        for (const auto &use : *uses) {
            if (auto dependency = dyn_cast_or_null<func::FuncOp>(
                    SymbolTable::lookupSymbolIn(ruleLibrary, use.getSymbolRef()))) {
                worklist.push_back(dependency);
            }
        }
    }

    // Names and callers were resolved before graph construction. Preserve those names when
    // importing only the functions needed by the chosen decomposition.
    SymbolTable symbolTable(module);
    for (auto function : llvm::make_early_inc_range(ruleLibrary.getOps<func::FuncOp>())) {
        if (required.contains(function)) {
            function->remove();
            symbolTable.insert(function);
        }
    }
    return success();
}

std::unique_ptr<DecompositionGraph> prepareGraph(ModuleOp module, ArrayRef<std::string> opIDs,
                                                 const PrepOptions &opts,
                                                 OwningOpRef<ModuleOp> &ruleLibrary) {
    LLVM_DEBUG(llvm::dbgs() << "Running GraphPrep with options:\n");
    LLVM_DEBUG({
        llvm::dbgs() << "targetGateSetOption\n";
        for (auto item : opts.targetGateSetOption) {
            llvm::dbgs() << "\t" << item << ",\n";
        }
        llvm::dbgs() << "\n";

        llvm::dbgs() << "fixedDecompsOption\n";
        for (auto item : opts.fixedDecompsOption) {
            llvm::dbgs() << "\t" << item << ",\n";
        }
        llvm::dbgs() << "\n";

        llvm::dbgs() << "excludeDecompsOption\n";
        for (auto item : opts.excludeDecompsOption) {
            llvm::dbgs() << "\t" << item << ",\n";
        }
        llvm::dbgs() << "\n";
    });

    std::vector<OperatorNode> setOfOps;
    for (const std::string &opID : opIDs) {
        setOfOps.push_back(parseOperator(opID));
    }

    std::vector<RuleNode> setOfRules;
    llvm::StringMap<std::string> opToFixedDecompName;
    llvm::StringMap<llvm::StringSet<>> opToExcludedDecompNames;
    WeightedGateset targetGateSet;

    // Get names for fixed and excluded decompositions.
    parseFixedDecomps(opts.fixedDecompsOption, opToFixedDecompName);
    parseExcludedDecomps(opts.excludeDecompsOption, opToExcludedDecompNames);
    if (failed(parseGateset(opts.targetGateSetOption, targetGateSet, module))) {
        return nullptr;
    }

    if (failed(getRuleNodes(module, opts.bytecodeRulesFile, opToFixedDecompName,
                            opToExcludedDecompNames, opts.libQPDPath, opts.libpythonPath,
                            setOfRules, ruleLibrary))) {
        return nullptr;
    }

    // Fixed and excluded decompositions are handled by filtering during rule collection.
    return std::make_unique<DecompositionGraph>(std::move(setOfOps), std::move(targetGateSet),
                                                std::move(setOfRules));
}

} // namespace GraphPrep
} // namespace quantum
} // namespace catalyst
