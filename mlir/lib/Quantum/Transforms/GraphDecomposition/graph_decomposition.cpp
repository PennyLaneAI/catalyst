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

#include <cstddef>
#include <cstdint>
#include <numeric>
#include <string>
#include <tuple>
#include <unordered_set>
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
#include "Driver/Timer.h"
#include "QRef/Transforms/Passes.h"
#include "Quantum/IR/QuantumDialect.h"
#include "Quantum/IR/QuantumInterfaces.h"
#include "Quantum/IR/QuantumOps.h"
#include "Quantum/Transforms/Passes.h"
#include "Quantum/Transforms/QPDLoader.h"

#include "DGBuilder.hpp"
#include "DGSolver.hpp"
#include "DGTypes.hpp"
#include "DGUtils.hpp"
#include "DecompUtils.hpp"

#define DEBUG_TYPE "graph-decomposition"

using namespace mlir;
using namespace catalyst::quantum;
using namespace DecompGraph::Core;
using namespace DecompGraph::Solver;

namespace {

/**
 * @brief RAII wrapper around `catalyst::utils::Timer` that times the enclosing scope and dumps
 * the result when the scope exits (including on the early returns taken by pass failures).
 *
 * Timing is only collected when the driver is run with `ENABLE_DIAGNOSTICS=ON`; otherwise
 * `start()`/`dump()` are no-ops and the only cost is the `getenv` done by the Timer constructor.
 */
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

} // namespace

namespace catalyst {
namespace quantum {

#define GEN_PASS_DEF_GRAPHDECOMPOSITIONPASS
#include "Quantum/Transforms/Passes.h.inc"

struct GraphDecompositionPass : public impl::GraphDecompositionPassBase<GraphDecompositionPass> {
    using GraphDecompositionPassBase::GraphDecompositionPassBase;
    void runOnOperation() final {
        ScopedDiagnosticTimer totalTimer("decomp:total");

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

        // NOTE: this is unused
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

        if (failed(getRuleNodes(bytecodeRulesFile, setOfRules, opToFixedDecompName,
                                opToAltDecompNames))) {
            return signalPassFailure();
        }
        getOperators(setOfOps);

        ///////////////////////////
        // Step 2: Build and solve the decomposition graph
        GraphResult solution;
        {
            ScopedDiagnosticTimer t("decomp:solver");
            // NOTE: fixed and alt-decomps are handled by filtering during rule collection. This is
            // dead code that should be removed
            FixedDecomps fixedDecomps = buildFixedDecomps(opToFixedDecompName, rulesByName);
            AltDecomps altDecomps = buildAltDecomps(opToAltDecompNames, rulesByName);
            DecompositionGraph graph(setOfOps, targetGateSet, setOfRules, std::move(fixedDecomps),
                                     std::move(altDecomps));
            DecompositionSolver solver(graph);
            solution = solver.solve();
        }
        LLVM_DEBUG(showSolution(solution));

        ///////////////////////////
        // Step 3: Convert python-decompositions from reference to value semantics and run
        // decompose-lowering to apply the chosen decomposition rules.

        ///////////////////////////
        // CQRs:
        //  - Adjoint: A chosen rule for an adjoint operator may re-emit its base decomposition
        //             wrapped in a `quantum.adjoint` region. Such a region is only reduced to
        //             op-level modified gates by `adjoint-lowering`. We therefore iterate
        //             `(decompose-lowering -> adjoint-lowering)` to a fixpoint.
        // The solver has already chosen every rule up front; this loop only applies them.
        ModuleOp module = getOperation();

        qref::DecomposeLoweringPassOptions dlOptions;
        // Collect only the rules on the chosen decomp tree, reachable from the circuit
        // root operators by following each op's chosen rule inputs.
        //
        // Note `solution` is the solver's map to target gateset, so it also holds ops explored
        // while costing rejected candidate rules (their inputs are solved to compute costs).
        // Feeding every one of those rules to the greedy decompose-lowering rewriter would
        // let stray rules fire on ops the chosen plan never routes through,
        // emitting gates beyond the planned resource counts.
        //
        // Basis rule names are collected too. `decompose-lowering` treats an empty
        // target-rules list as "apply every rule", so a circuit already in the target gateset
        // must still produce a non-empty list, or its terminals would be
        // decomposed by whatever rules happen to be loaded.
        {
            std::unordered_set<OperatorNode, OperatorNodeHash> visited;
            llvm::StringSet<> seenRules;
            std::vector<OperatorNode> worklist = setOfOps;
            while (!worklist.empty()) {
                OperatorNode op = worklist.back();
                worklist.pop_back();
                if (!visited.insert(op).second) {
                    continue;
                }
                auto it = solution.find(op);
                if (it == solution.end()) {
                    continue;
                }
                const ChosenDecompRule &chosenRule = it->second;
                if (seenRules.insert(chosenRule.ruleName).second) {
                    dlOptions.targetRulesOption.push_back(chosenRule.ruleName);
                }
                for (const RuleTerm &input : chosenRule.inputs) {
                    worklist.push_back(input.op);
                }
            }
        }

        // Convert reference-semantics python decompositions to value semantics.
        {
            ScopedDiagnosticTimer t("decomp:ref-to-value");
            OpPassManager valueSemanticsPm("builtin.module");
            valueSemanticsPm.addPass(qref::createValueSemanticsConversionPass());
            if (failed(runPipeline(valueSemanticsPm, module))) {
                return signalPassFailure();
            }
        }

        auto countOps = [](ModuleOp m) {
            size_t count = 0;
            m->walk([&](mlir::Operation *) { count++; });
            return count;
        };

        // Fixpoint: apply the chosen rules and distribute any `quantum.adjoint` regions they emit,
        // until the module stops changing.
        constexpr unsigned maxIterations = 64;
        size_t previousOpCount = countOps(module);
        ScopedDiagnosticTimer fixpointTimer("decomp:greedy-lowering");
        unsigned iterationsRun = 0;
        for (unsigned iter = 0; iter < maxIterations; ++iter) {
            iterationsRun = iter + 1;
            {
                OpPassManager decomposePm("builtin.module");
                decomposePm.addPass(createDecomposeLoweringPass(dlOptions));
                if (failed(runPipeline(decomposePm, module))) {
                    return signalPassFailure();
                }
            }

            // Distribute any `quantum.ctrl`/`quantum.adjoint` regions the rules emitted, lazily.
            // `lower-modifiers` reduces both (including nested `ctrl(adjoint(...))`) to a fixpoint
            // in one greedy pass.
            bool hasModifierRegion = module
                                         ->walk([&](mlir::Operation *op) {
                                             return (isa<CtrlOp, AdjointOp>(op))
                                                        ? mlir::WalkResult::interrupt()
                                                        : mlir::WalkResult::advance();
                                         })
                                         .wasInterrupted();
            if (hasModifierRegion) {
                OpPassManager modifierPm("builtin.module");
                modifierPm.addPass(createLowerModifiersPass());
                if (failed(runPipeline(modifierPm, module))) {
                    return signalPassFailure();
                }
            }

            size_t currentOpCount = countOps(module);
            if (currentOpCount == previousOpCount) {
                break;
            }
            previousOpCount = currentOpCount;
        }
        LDBG() << "lowering fixpoint reached after " << iterationsRun << " iteration(s), "
               << previousOpCount << " ops";
    }

  private:
    void parseFixedDecomps(llvm::StringMap<std::string> &opToFixedDecompName,
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

    void parseAltDecomps(llvm::StringMap<llvm::SmallVector<std::string>> &opToAltDecompNames,
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
                    opRulesList.push_back(ruleNameRef.str());
                    userRuleNames.insert(ruleNameRef.str());
                }
            }
        }
    }

    LogicalResult parseGateset(WeightedGateset &targetGateSet) {
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

    LogicalResult addRuleNode(mlir::func::FuncOp rule, std::vector<RuleNode> &ruleNodes) {
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
            ResourceAnalysis analysis(rule, {}, /*collectDetailedOperations=*/true);
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

    LogicalResult loadBuiltInDecompositionRules(llvm::StringRef filename) {
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

        // add to module
        for (auto rule :
             llvm::make_early_inc_range(builtinModule.get().getOps<mlir::func::FuncOp>())) {
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
    LogicalResult
    loadDecompositionRules(llvm::StringMap<std::string> &opToFixedDecompName,
                           llvm::StringMap<llvm::SmallVector<std::string>> &opToAltDecompNames,
                           std::vector<RuleNode> &ruleNodes) {
        mlir::ModuleOp module = getOperation();

        WalkResult walkResult = module.walk([&](mlir::func::FuncOp func) {
            if (func->hasAttr(DecompUtils::target_gate_attr_name)) {
                // TODO: remove GOID inspection
                llvm::StringRef targetGate =
                    func->getAttrOfType<mlir::StringAttr>(DecompUtils::target_gate_attr_name)
                        .getValue()
                        .take_until([](char c) { return c == '{'; }) // remove GOID data
                        .drop_while(llvm::isDigit);                  // remove <n>C numeric prefix

                // if this op has alt or fixed decomps then we are we only take the specified rules
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
    mlir::LogicalResult loadPythonDecomps() {
        mlir::ModuleOp module = getOperation();
        MLIRContext *context = &getContext();

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

        llvm::SmallVector<quantum::DecomposableGate> decomposableOps;
        module.walk([&](quantum::DecomposableGate op) {
            if (!DecompUtils::isInDecompRule(op)) {
                decomposableOps.push_back(op);
            }
        });

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

    void getOperators(std::vector<OperatorNode> &operators) {
        getOperation().walk([&](DecomposableGate op) {
            if (DecompUtils::isInDecompRule(op)) {
                return;
            }
            assert(!op->getParentOfType<AdjointOp>() && !op->getParentOfType<CtrlOp>() &&
                   "graph-decomposition requires op-level modifiers: ctrl/adjoint regions must be "
                   "lowered before the decomposition graph is built");

            // Derive the id and modifier-wrapped name from the single parse path, so operator
            // nodes and rule nodes agree on the spelling of `C(...)`/`Adjoint(...)`.
            OperatorNode node = parseOperator(op.getGraphOpId());

            // numWires/numParams are debug-only; parseOperator leaves them at defaults for the
            // graphOpId form, so we need to fill them accurately from the op here.
            node.numWires = op.getNonCtrlQubitOperands().size();
            if (auto paramOp =
                    llvm::dyn_cast<catalyst::quantum::ParametrizedGate>(op.getOperation())) {
                node.numParams = paramOp.getAllParams().size();
            } else {
                node.numParams = 0;
            }

            operators.push_back(node);
        });
    }

    /**
     * @brief Parse a graphOpId string into an OperatorNode.
     *
     * The graphOpId format is "<name>{params}{wires}{static}[uid]", where <name> already carries
     * any name-wrapped op-level modifiers produced by `defaultGetGraphOpId`, e.g.
     * "C(Adjoint(RX)){0:[f64]}{wires:1}{}".
     */
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

    /**
     * @brief Create RuleNodes for each rule available to be used in graph decomposition.
     */
    LogicalResult
    getRuleNodes(llvm::StringRef filename, std::vector<RuleNode> &rules,
                 llvm::StringMap<std::string> &opToFixedDecompName,
                 llvm::StringMap<llvm::SmallVector<std::string>> &opToAltDecompNames) {
        ScopedDiagnosticTimer t("decomp:rules");
        // Load pre-compiled rules (ignore failure, we can try to solve without) into the module
        std::ignore = loadBuiltInDecompositionRules(filename);

        // Lower compile-time rules into the module
        if (failed(loadPythonDecomps())) {
            return failure();
        }

        // Load rules from the module into the set of rules used by the graph, filtering by fixed-
        // and alt-decomps
        if (failed(loadDecompositionRules(opToFixedDecompName, opToAltDecompNames, rules))) {
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
                                   const llvm::StringMap<const RuleNode *> &rulesByName) {
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
                    const llvm::StringMap<const RuleNode *> &rulesByName) {
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
