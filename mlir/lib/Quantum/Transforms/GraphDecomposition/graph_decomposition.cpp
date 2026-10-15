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

#include <memory>
#include <sstream>
#include <string>
#include <unordered_set>
#include <vector>

#include "llvm/ADT/StringSet.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LogicalResult.h"
#include "stablehlo/dialect/StablehloOps.h"

#include "Driver/Timer.h"
#include "QRef/Transforms/Passes.h"
#include "Quantum/Transforms/Passes.h"

#include "DGBuilder.hpp"
#include "DGSolver.hpp"
#include "DGTypes.hpp"
#include "DGUtils.hpp"
#include "GraphDecompositionPrep.hpp"

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

        ModuleOp module = getOperation();

        ///////////////////////////
        // Step 1–2: Prepare and solve the decomposition graph
        std::unique_ptr<DecompositionGraph> graph = GraphDecompositionPrep::prepareGraph(
            module,
            {
                .targetGateSetOption = targetGateSetOption,
                .fixedDecompsOption = fixedDecompsOption,
                .altDecompsOption = altDecompsOption,
                .bytecodeRulesFile = bytecodeRulesFile,
                .libQPDPath = libQPDPath,
                .libpythonPath = libpythonPath,
            },
            [&](Operation *op) {
                OpPassManager modifierPm("builtin.module");
                modifierPm.addPass(createModifiersLoweringPass());
                return runPipeline(modifierPm, op);
            });
        if (!graph) {
            return signalPassFailure();
        }

        GraphResult solution;
        {
            ScopedDiagnosticTimer t("decomp:solver");
            DecompositionSolver solver(*graph);
            solution = solver.solve();
        }
        // Dump the solver's choices when asked for (`graph_decomposition(..., verbose=True)`), or
        // whenever the pass runs under `-debug-only=graph-decomposition` on a debug build. The
        // debug build routes the dump through `llvm::dbgs()` like the rest of this pass's debug
        // output, so it honours `-debug-output=...` rather than going straight to stderr.
        if (verboseOption) {
            showSolution(solution);
        } else {
            LLVM_DEBUG({
                std::ostringstream dump;
                showSolution(solution, dump);
                llvm::dbgs() << dump.str();
            });
        }

        ///////////////////////////
        // Step 3: use decompose-lowering to apply the chosen decomposition rules.
        // Note that on-demand rules may have introduced mixed semantics at this point, but
        // decompose-lowering runs conversion and will ensure consistency
        {
            ScopedDiagnosticTimer fixpointTimer("decomp:decompose-lowering");

            ///////////////////////////
            // CQRs:
            //  - Adjoint: A chosen rule for an adjoint operator may re-emit its base decomposition
            //             wrapped in a `quantum.adjoint` region. Such a region is only reduced to
            //             op-level modified gates by `adjoint-lowering`. We therefore iterate
            //             `(decompose-lowering -> adjoint-lowering)` to a fixpoint.
            // The solver has already chosen every rule up front; this loop only applies them.

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
            qref::DecomposeLoweringPassOptions dlOptions;
            dlOptions.inlineRuleBody = inlineRuleBody;
            std::unordered_set<OperatorNode, OperatorNodeHash> visited;
            llvm::StringSet<> seenRules;
            std::vector<OperatorNode> worklist = graph->getRootOps();
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
            OpPassManager decomposePm("builtin.module");
            decomposePm.addPass(createDecomposeLoweringPass(dlOptions));
            if (failed(runPipeline(decomposePm, module))) {
                return signalPassFailure();
            }
        }
    }
};

} // namespace quantum
} // namespace catalyst
