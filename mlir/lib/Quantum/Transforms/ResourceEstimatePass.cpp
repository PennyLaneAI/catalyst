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

#include <fstream>
#include <memory>
#include <string>

#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Complex/IR/Complex.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "stablehlo/dialect/StablehloOps.h"

#include "Catalyst/Analysis/ResourceAnalysis.h"
#include "Catalyst/Analysis/ResourceResult.h"
#include "Quantum/Transforms/Passes.h"

#include "DGBuilder.hpp"
#include "DGSolver.hpp"
#include "DGTypes.hpp"
#include "DGUtils.hpp"
#include "GraphPrep.hpp"

using namespace mlir;
using namespace DecompGraph::Core;
using namespace DecompGraph::Solver;

namespace catalyst {
namespace quantum {

#define GEN_PASS_DEF_RESOURCEESTIMATEPASS
#include "Quantum/Transforms/Passes.h.inc"

struct ResourceEstimatePass : public impl::ResourceEstimatePassBase<ResourceEstimatePass> {
    using ResourceEstimatePassBase::ResourceEstimatePassBase;

    void runOnOperation() final {
        ModuleOp module = getOperation();

        ResourceAnalysis analysis(module, {}, /*collectDetailedOperations=*/true);
        llvm::StringRef entryFuncName = analysis.getEntryFunc();

        std::vector<std::string> opIDs;
        if (const catalyst::ResourceResult *flat = analysis.getFlattenedResource(entryFuncName)) {
            for (const auto &opID : flat->observedGraphOpIds) {
                opIDs.push_back(opID.getKey().str());
            }
        }

        OwningOpRef<ModuleOp> ruleLibrary;
        std::unique_ptr<DecompositionGraph> graph =
            GraphPrep::prepareGraph(module, opIDs,
                                    {
                                        .targetGateSetOption = targetGateSetOption,
                                        .fixedDecompsOption = fixedDecompsOption,
                                        .excludeDecompsOption = excludeDecompsOption,
                                        .bytecodeRulesFile = bytecodeRulesFile,
                                        .libQPDPath = libQPDPath,
                                        .libpythonPath = libpythonPath,
                                    },
                                    ruleLibrary);

        if (!graph) {
            return signalPassFailure();
        }

        DecompositionSolver solver(*graph);
        GraphResult solution = solver.solve();
        if (verboseOption) {
            showSolution(solution);
        }

        const auto &results = analysis.getResults();

        llvm::StringMap<ResourceResult> estimates;
        llvm::StringMap<ResourceResult> ruleFunctions;
        for (const auto &funcEntry : results) {
            estimates[funcEntry.getKey()] =
                estimateResources(funcEntry.getValue(), solution, ruleFunctions);
        }
        for (auto &ruleEntry : ruleFunctions) {
            estimates[ruleEntry.getKey()] = std::move(ruleEntry.getValue());
        }

        if (outputJson) {
            std::string jsonStr = buildJsonString(estimates);
            if (outputFname.empty()) {
                llvm::outs() << jsonStr;
            } else {
                writeJsonToFile(jsonStr, outputFname);
            }
        }
    }

  private:
    /// Replace each op in a function's resources `result` with a call to the function of its
    /// chosen decomposition rule, added to `ruleFunctions`. Like the decomposition rules after
    /// `graph-decomposition`, each rule function holds the basis gates of the rule and calls the
    /// rule functions of the other ops of the rule.
    static ResourceResult estimateResources(const ResourceResult &result,
                                            const GraphResult &solution,
                                            llvm::StringMap<ResourceResult> &ruleFunctions) {
        ResourceResult estimate;
        estimate.measurements = result.measurements;
        estimate.classicalInstructions = result.classicalInstructions;
        estimate.functionCalls = result.functionCalls;
        estimate.varFunctionCalls = result.varFunctionCalls;
        estimate.numAllocQubits = result.numAllocQubits;
        estimate.numArgQubits = result.numArgQubits;
        estimate.liveQubits = result.liveQubits;
        estimate.peakQubits = result.peakQubits;
        estimate.liveQubitsAtCall = result.liveQubitsAtCall;
        estimate.deviceName = result.deviceName;
        estimate.isQnode = result.isQnode;
        estimate.hasBranches = result.hasBranches;
        estimate.hasDynLoop = result.hasDynLoop;
        estimate.autoQubitManagement = result.autoQubitManagement;

        // Operations that are not decomposable gates have no graph op id, and are kept as they are.
        for (const auto &opEntry : result.nonGraphOperations) {
            for (const auto &sizeEntry : opEntry.getValue()) {
                estimate.operations[opEntry.getKey()][sizeEntry.first] += sizeEntry.second;
            }
        }

        // Decomposable gates are keyed by their graph op id (GOID).
        for (const auto &opEntry : result.detailedOperations) {
            OperatorNode operatorNode;
            std::string goid = opEntry.getKey().str();
            double scale = opEntry.getValue();
            operatorNode.id = goid;

            auto graphOp = solution.find(operatorNode);
            if (graphOp == solution.end()) {
                // Not a graph operator: counted from `nonGraphOperations` above.
                continue;
            }
            addDecomposition(estimate, graphOp->second, scale, solution, ruleFunctions);
        }
        return estimate;
    }

    /// Add `count` ops decomposed with `rule` to `estimate`: a basis gate is counted, and another
    /// op calls the function of its rule, which is added to `ruleFunctions` if it is not there.
    static void addDecomposition(ResourceResult &estimate, const ChosenDecompRule &rule,
                                 double count, const GraphResult &solution,
                                 llvm::StringMap<ResourceResult> &ruleFunctions) {
        if (rule.isBasis) {
            // The number of params cannot be recovered from the GOID.
            estimate.operations[getResourceName(rule.op.name)][{getNumWires(rule.op.id), -1}] +=
                count;
            return;
        }
        std::string name = rule.ruleName + "(" + rule.op.id + ")";
        estimate.functionCalls[name] += count;
        // Where the op is in the function is not known, so all its qubits may be live
        double &live = estimate.liveQubitsAtCall[name];
        live = std::max(live, estimate.peakQubits);
        if (ruleFunctions.contains(name)) {
            return;
        }
        ruleFunctions[name] = ResourceResult();
        ResourceResult ruleResult;
        for (const auto &input : rule.inputs) {
            if (input.multiplicity > 0) {
                addDecomposition(ruleResult, solution.at(input.op), input.multiplicity, solution,
                                 ruleFunctions);
            }
        }
        ruleFunctions[name] = std::move(ruleResult);
    }

    /// Name under which the resource analysis counts an op of the given graph operator name.
    static std::string getResourceName(StringRef graphName) {
        return llvm::StringSwitch<std::string>(graphName)
            .Case("PauliMeasure", "PPM")
            .Case("PPR_2", "PPR-pi/2")
            .Case("PPR_4", "PPR-pi/4")
            .Case("PPR_8", "PPR-pi/8")
            .Default(graphName.str());
    }

    /// Number of wires of an operator, from the wire lengths of its GOID
    /// (`name{dynamic shape}{wire lengths}{static data}[extra]`), or -1 if they cannot be found.
    static int getNumWires(StringRef goid) {
        size_t groupStart = 0;
        int depth = 0;
        int groupIndex = 0;
        for (size_t i = 0; i < goid.size(); ++i) {
            if (goid[i] == '{' && depth++ == 0) {
                groupStart = i + 1;
            } else if (goid[i] == '}' && --depth == 0 && groupIndex++ == 1) {
                int numWires = 0;
                SmallVector<StringRef> entries;
                goid.slice(groupStart, i).split(entries, ',', -1, /*KeepEmpty=*/false);
                for (StringRef entry : entries) {
                    int len;
                    if (entry.rsplit(':').second.getAsInteger(10, len)) {
                        return -1;
                    }
                    numWires += len;
                }
                return numWires;
            }
        }
        return -1;
    }

    /// Serialize the estimates into a JSON string. The JSON borrows keys from `estimates`, so it
    /// must be printed while they are alive.
    static std::string buildJsonString(const llvm::StringMap<ResourceResult> &estimates) {
        llvm::json::Object root;
        for (const auto &funcEntry : estimates) {
            root[funcEntry.getKey()] = funcEntry.getValue().toJson();
        }

        llvm::json::Value jsonValue(std::move(root));
        return llvm::formatv("{0:2}", jsonValue).str() + "\n";
    }

    /// Write JSON to a file.
    static void writeJsonToFile(const std::string &jsonStr, const std::string &fileName) {
        std::ofstream ofile(fileName);
        if (!ofile.is_open()) {
            llvm::errs() << "Error: could not open resource output file: " << fileName << "\n";
            return;
        }
        ofile << jsonStr;
    }
};

} // namespace quantum
} // namespace catalyst
