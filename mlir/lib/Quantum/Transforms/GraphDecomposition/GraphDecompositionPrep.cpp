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

#include "GraphDecompositionPrep.hpp"

#include <cassert>
#include <cstdint>
#include <memory>
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
#include "llvm/Support/LogicalResult.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Support/WalkResult.h"

#include "Catalyst/Analysis/ResourceAnalysis.h"
#include "Catalyst/Analysis/ResourceResult.h"
#include "Driver/Timer.h"
#include "Quantum/IR/QuantumInterfaces.h"
#include "Quantum/IR/QuantumOps.h"
#include "Quantum/Transforms/QPDLoader.h"

#include "DGTypes.hpp"
#include "DecompUtils.hpp"

#define DEBUG_TYPE "graph-decomposition"

using namespace mlir;
using namespace catalyst::quantum;
using namespace DecompGraph::Core;
using namespace DecompGraph::Solver;

namespace {

class ScopedDiagnosticTimer {
  public:
    explicit ScopedDiagnosticTimer(std::string name) : name(std::move(name)) { timer.start(); }
    ~ScopedDiagnosticTimer() { timer.dump(name); }

    ScopedDiagnosticTimer(const ScopedDiagnosticTimer &) = delete;
    ScopedDiagnosticTimer &operator=(const ScopedDiagnosticTimer &) = delete;

  private:
    catalyst::utils::Timer<> timer;
    std::string name;
};

bool hasModifierRegion(ModuleOp module) {
    return module
        .walk([&](Operation *op) {
            return (isa<CtrlOp, AdjointOp>(op)) ? WalkResult::interrupt() : WalkResult::advance();
        })
        .wasInterrupted();
}

void parseFixedDecomps(ArrayRef<std::string> fixedDecompsOption,
                       llvm::StringMap<std::string> &opToFixedDecompName,
                       llvm::StringSet<> &userRuleNames) {
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

void parseAltDecomps(ArrayRef<std::string> altDecompsOption,
                     llvm::StringMap<llvm::StringSet<>> &opToAltDecompNames,
                     llvm::StringSet<> &userRuleNames) {
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
                opRulesList.insert(ruleNameRef.str());
                userRuleNames.insert(ruleNameRef.str());
            }
        }
    }
}

LogicalResult parseGateset(ArrayRef<std::string> targetGateSetOption,
                           WeightedGateset &targetGateSet) {
    for (const std::string &opCostPair : targetGateSetOption) {
        llvm::StringRef pairRef(opCostPair);

        auto [opNameRaw, costRaw] = pairRef.split("=");
        llvm::StringRef opName = opNameRaw.trim();
        llvm::StringRef cost = costRaw.trim();

        // Note gate_set is now a DictionaryAttr which quotes any key that is
        // not an MLIR op (e.g. "Adjoint(TemporaryAND)").
        // As the result, we need to strip the surrounding quotes so the stored name
        // matches the op's graphOpId name following parseFixedDecomps / parseAltDecomps.
        opName.consume_front("\"");
        opName.consume_back("\"");

        cost.consume_back(": f64");
        cost = cost.trim();

        bool success = to_float(cost, targetGateSet.ops[opName.str()]);

        if (!success) {
            return failure();
        }
    }
    return success();
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
    ruleNode.output = GraphDecompositionPrep::parseOperator(targetGateAttr.getValue());

    for (const auto &namedAttr : operations) {
        if (auto intAttr = dyn_cast<IntegerAttr>(namedAttr.getValue())) {
            ruleNode.inputs.push_back(
                {GraphDecompositionPrep::parseOperator(namedAttr.getName().strref()),
                 static_cast<uint32_t>(intAttr.getInt())});
        }
    }

    // 4. Add RuleNode
    ruleNodes.push_back(std::move(ruleNode));
    return success();
}

LogicalResult loadBuiltInDecompositionRules(ModuleOp module, llvm::StringRef filename) {
    ParserConfig config(module.getContext());
    OwningOpRef<ModuleOp> builtinModule = parseSourceFile<ModuleOp>(filename, config);

    SymbolTable symbolTable(module);

    if (!builtinModule) {
        llvm::errs() << "failed to load built-in decomposition rules from '" << filename
                     << "': the rules file could not be parsed\n";
        return failure();
    }

    // add to module
    for (auto rule : llvm::make_early_inc_range(builtinModule.get().getOps<func::FuncOp>())) {
        // avoid double-insertion
        if (!symbolTable.lookup<func::FuncOp>(rule.getName())) {
            rule->remove();
            module.push_back(std::move(rule));
        }
    }
    return success();
}

/**
 * @brief Load the listed user rules into the set of RuleNodes for the graph.
 */
LogicalResult loadDecompositionRules(ModuleOp module,
                                     llvm::StringMap<std::string> &opToFixedDecompName,
                                     llvm::StringMap<llvm::StringSet<>> &opToAltDecompNames,
                                     std::vector<RuleNode> &ruleNodes) {
    WalkResult walkResult =
        module.walk([&](mlir::func::FuncOp func) {
            if (func->hasAttr(DecompUtils::target_gate_attr_name)) {
                // TODO: remove GOID inspection
                llvm::StringRef targetGate =
                    func->getAttrOfType<mlir::StringAttr>(DecompUtils::target_gate_attr_name)
                        .getValue()
                        .take_until([](char c) { return c == '{'; }) // remove GOID data
                        .drop_while(llvm::isDigit);                  // remove <n>C numeric prefix

                // if this op has alt or fixed decomps then we are we only take the specified
                // rules
                if (opToFixedDecompName.contains(targetGate) ||
                    opToAltDecompNames.contains(targetGate)) {

                    // frontend name of the decomp rule for matching with fixed/alt-decomps
                    if (!func->hasAttr("frontend_name")) {
                        llvm::errs()
                            << "The " << func.getName()
                            << " decomposition rule targets a gate with fixed/alt-decomps, but "
                               "doesn't have the `frontend_name` attribute.";
                    }
                    llvm::StringRef funcName =
                        func->getAttrOfType<mlir::StringAttr>("frontend_name");

                    if (opToFixedDecompName[targetGate] == funcName) {
                        if (failed(addRuleNode(func, ruleNodes))) {
                            return WalkResult::interrupt();
                        }
                        return WalkResult::skip();
                    } else if (llvm::is_contained(opToAltDecompNames[targetGate], funcName)) {
                        if (failed(addRuleNode(func, ruleNodes))) {
                            return WalkResult::interrupt();
                        }
                        return WalkResult::skip();
                    }
                    LDBG() << "Decomposition rule " << func.getName()
                           << " was registered to an op with fixed or alt decomps, and wasn't in "
                              "the list - skipping";
                    return WalkResult::advance();
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
LogicalResult loadPythonDecomps(ModuleOp module, llvm::StringRef libQPDPath,
                                llvm::StringRef libpythonPath) {
    MLIRContext *context = module.getContext();

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

    llvm::SmallVector<DecomposableGate> decomposableOps;
    module.walk([&](DecomposableGate op) {
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

    for (DecomposableGate op : decomposableOps) {
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
            module.push_back(std::move(outOp.release()));
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

LogicalResult getRuleNodes(ModuleOp module, llvm::StringRef filename,
                           llvm::StringMap<std::string> &opToFixedDecompName,
                           llvm::StringMap<llvm::StringSet<>> &opToAltDecompNames,
                           llvm::StringRef libQPDPath, llvm::StringRef libpythonPath,
                           std::vector<RuleNode> &rules) {
    ScopedDiagnosticTimer t("decomp:rules");
    if (!filename.empty()) {
        std::ignore = loadBuiltInDecompositionRules(module, filename);
    }

    if (failed(loadPythonDecomps(module, libQPDPath, libpythonPath))) {
        return failure();
    }

    return loadDecompositionRules(module, opToFixedDecompName, opToAltDecompNames, rules);
}

void getOperators(ModuleOp module, std::vector<OperatorNode> &operators) {
    module.walk([&](DecomposableGate op) {
        if (DecompUtils::isInDecompRule(op)) {
            return;
        }
        assert(!op->getParentOfType<AdjointOp>() && !op->getParentOfType<CtrlOp>() &&
               "graph-decomposition requires op-level modifiers: ctrl/adjoint regions must be "
               "lowered before the decomposition graph is built");

        // Derive the id and modifier-wrapped name from the single parse path, so operator
        // nodes and rule nodes agree on the spelling of `C(...)`/`Adjoint(...)`.
        OperatorNode node = GraphDecompositionPrep::parseOperator(op.getGraphOpId());

        // numWires/numParams are debug-only; parseOperator leaves them at defaults for the
        // graphOpId form, so we need to fill them accurately from the op here.
        node.numWires = op.getNonCtrlQubitOperands().size();
        if (auto paramOp = llvm::dyn_cast<catalyst::quantum::ParametrizedGate>(op.getOperation())) {
            node.numParams = paramOp.getAllParams().size();
        } else {
            node.numParams = 0;
        }

        operators.push_back(node);
    });
}

} // namespace

namespace catalyst {
namespace quantum {
namespace GraphDecompositionPrep {

OperatorNode parseOperator(llvm::StringRef raw) {
    OperatorNode node;

    // Base op: either the graphOpId "Name{...}..." form or the legacy "Name(w,p)" form.
    if (raw.contains('[') || raw.contains('{')) {
        node.id = raw.str();
        node.name = raw.take_until([](char c) { return c == '[' || c == '{'; });
    } else {
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

std::unique_ptr<DecompositionGraph> prepareGraph(Operation *op, const PrepOptions &opts,
                                                 LowerModifiersFn lowerModifiers) {
    LLVM_DEBUG(llvm::dbgs() << "Running GraphDecompositionPrep with options:\n");
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

        llvm::dbgs() << "altDecompsOption\n";
        for (auto item : opts.altDecompsOption) {
            llvm::dbgs() << "\t" << item << ",\n";
        }
        llvm::dbgs() << "\n";
    });

    ModuleOp module = cast<ModuleOp>(op);
    // Strip away adjoint and control regions to match the graph, where modifiers are on
    // the individual ops
    if (hasModifierRegion(module) && failed(lowerModifiers(module))) {
        return nullptr;
    }

    std::vector<OperatorNode> setOfOps;
    std::vector<RuleNode> setOfRules;
    llvm::StringSet<> userRuleNames;
    llvm::StringMap<std::string> opToFixedDecompName;
    llvm::StringMap<llvm::StringSet<>> opToAltDecompNames;
    WeightedGateset targetGateSet;

    // get names for fixed and alt decomps
    parseFixedDecomps(opts.fixedDecompsOption, opToFixedDecompName, userRuleNames);
    parseAltDecomps(opts.altDecompsOption, opToAltDecompNames, userRuleNames);
    if (failed(parseGateset(opts.targetGateSetOption, targetGateSet))) {
        return nullptr;
    }

    if (failed(getRuleNodes(module, opts.bytecodeRulesFile, opToFixedDecompName, opToAltDecompNames,
                            opts.libQPDPath, opts.libpythonPath, setOfRules))) {
        return nullptr;
    }
    getOperators(module, setOfOps);

    // NOTE: fixed and alt-decomps are handled by filtering during rule collection.
    return std::make_unique<DecompositionGraph>(std::move(setOfOps), std::move(targetGateSet),
                                                std::move(setOfRules));
}

} // namespace GraphDecompositionPrep
} // namespace quantum
} // namespace catalyst
