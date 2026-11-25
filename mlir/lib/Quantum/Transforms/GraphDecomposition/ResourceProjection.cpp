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

#include "ResourceProjection.hpp"

#include <algorithm>
#include <memory>
#include <string>
#include <vector>

#include "mlir/IR/OwningOpRef.h"

#include "Quantum/IR/QuantumInterfaces.h"

#include "DGSolver.hpp"
#include "DGTypes.hpp"
#include "DGUtils.hpp"

using namespace mlir;
using namespace DecompGraph::Core;
using namespace DecompGraph::Solver;

namespace catalyst {
namespace quantum {

namespace {

class DecompositionProjection {
  public:
    explicit DecompositionProjection(const GraphResult &solution) : solution(solution) {}

    ResourceResult projectFunction(const ResourceResult &result) {
        ResourceResult projected;
        projected.measurements = result.measurements;
        projected.classicalInstructions = result.classicalInstructions;
        projected.functionCalls = result.functionCalls;
        projected.varFunctionCalls = result.varFunctionCalls;
        projected.numAllocQubits = result.numAllocQubits;
        projected.numArgQubits = result.numArgQubits;
        projected.liveQubits = result.liveQubits;
        projected.peakQubits = result.peakQubits;
        projected.liveQubitsAtCall = result.liveQubitsAtCall;
        projected.deviceName = result.deviceName;
        projected.isQnode = result.isQnode;
        projected.hasBranches = result.hasBranches;
        projected.hasDynLoop = result.hasDynLoop;
        projected.autoQubitManagement = result.autoQubitManagement;

        projected.operations = result.nonDecomposableOperations;
        for (const auto &entry : result.detailedOperations) {
            if (result.observedGraphOpIds.contains(entry.getKey())) {
                addOperator(projected, entry.getKey(), entry.getValue());
            }
        }
        return projected;
    }

    llvm::StringMap<ResourceResult> takeRuleFunctions() { return std::move(ruleFunctions); }

  private:
    void addOperator(ResourceResult &resources, llvm::StringRef graphOpId, double count) {
        OperatorNode node;
        node.id = graphOpId.str();
        auto chosen = solution.find(node);
        if (chosen == solution.end() || chosen->second.isBasis) {
            int numQubits = static_cast<int>(getGraphOpNumQubits(graphOpId).value_or(-1));
            // The number of parameters is not in the graph op id.
            resources.operations[GraphPrep::getGraphOpName(graphOpId)][{numQubits, -1}] += count;
            return;
        }
        addRuleCall(resources, chosen->second, count);
    }

    void addRuleCall(ResourceResult &resources, const ChosenDecompRule &rule, double count) {
        std::string name = rule.ruleName + "(" + rule.op.id + ")";
        resources.functionCalls[name] += count;
        // Where the gate is in the function is not known, so all its qubits may be live.
        double &live = resources.liveQubitsAtCall[name];
        live = std::max(live, resources.peakQubits);

        if (ruleFunctions.contains(name)) {
            return;
        }
        ruleFunctions[name] = ResourceResult();
        ResourceResult ruleResources;
        for (const RuleTerm &input : rule.inputs) {
            addOperator(ruleResources, input.op.id, input.multiplicity);
        }
        ruleFunctions[name] = std::move(ruleResources);
    }

    const GraphResult &solution;
    llvm::StringMap<ResourceResult> ruleFunctions;
};

} // namespace

FailureOr<llvm::StringMap<ResourceResult>>
projectDecomposition(ModuleOp module, const ResourceAnalysis &analysis,
                     const GraphPrep::PrepOptions &options, bool verbose) {
    std::vector<std::string> opIDs;
    if (const ResourceResult *flat = analysis.getFlattenedResource(analysis.getEntryFunc())) {
        for (const auto &opID : flat->observedGraphOpIds) {
            opIDs.push_back(opID.getKey().str());
        }
    }

    OwningOpRef<ModuleOp> ruleLibrary;
    std::unique_ptr<DecompositionGraph> graph =
        GraphPrep::prepareGraph(module, opIDs, options, ruleLibrary);
    if (!graph) {
        return failure();
    }

    DecompositionSolver solver(*graph);
    GraphResult solution = solver.solve();
    if (verbose) {
        showSolution(solution);
    }

    DecompositionProjection projection(solution);
    llvm::StringMap<ResourceResult> projected;
    for (const auto &entry : analysis.getResults()) {
        projected[entry.getKey()] = projection.projectFunction(entry.getValue());
    }
    for (auto &entry : projection.takeRuleFunctions()) {
        projected[entry.getKey()] = std::move(entry.getValue());
    }
    return projected;
}

} // namespace quantum
} // namespace catalyst
