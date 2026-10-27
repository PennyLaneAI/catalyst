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
#include "GraphDecompositionPrep.hpp"

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

        std::unique_ptr<DecompositionGraph> graph = GraphDecompositionPrep::prepareGraph(
            module, {
                        .targetGateSetOption = targetGateSetOption,
                        .fixedDecompsOption = fixedDecompsOption,
                        .altDecompsOption = altDecompsOption,
                        .bytecodeRulesFile = bytecodeRulesFile,
                        .libQPDPath = libQPDPath,
                        .libpythonPath = libpythonPath,
                    });
        if (!graph) {
            return signalPassFailure();
        }

        DecompositionSolver solver(*graph);
        GraphResult solution = solver.solve();
        if (verboseOption) {
            showSolution(solution);
        }

        // Must run after prepareGraph, which lowers modifier regions in place, so that the graph
        // op ids collected here carry the same op-level modifiers as the graph's operators.
        ResourceAnalysis analysis(module, {}, /*collectDetailedOperations=*/true);
        const auto &results = analysis.getResults();

        llvm::StringMap<ResourceResult> estimates;
        StringRef entry = analysis.getEntryFunc();
        if (const ResourceResult *flat = analysis.getFlattenedResource(entry)) {
            estimates[entry] = estimateResources(*flat, solution);
        }

        // Flattening only folds in static calls, so dynamic loop bodies are estimated separately
        // and stay referenced from function_calls.dynamic.
        for (const auto &funcEntry : results) {
            for (const auto &call : funcEntry.getValue().varFunctionCalls) {
                if (const ResourceResult *body = analysis.getFlattenedResource(call.getKey())) {
                    estimates[call.getKey()] = estimateResources(*body, solution);
                }
            }
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
    /// Replace each op in the flattened resources `flat` with the basis gates of its chosen
    /// decomposition rule, scaled by the op's count.
    static ResourceResult estimateResources(const ResourceResult &flat,
                                            const GraphResult &solution) {
        ResourceResult estimate;
        estimate.measurements = flat.measurements;
        estimate.classicalInstructions = flat.classicalInstructions;
        // Static calls are already folded into `flat`, but dynamic loop bodies are not.
        estimate.varFunctionCalls = flat.varFunctionCalls;
        estimate.numAllocQubits = flat.numAllocQubits;
        estimate.numArgQubits = flat.numArgQubits;
        estimate.deviceName = flat.deviceName;
        estimate.isQnode = flat.isQnode;
        estimate.hasBranches = flat.hasBranches;
        estimate.hasDynLoop = flat.hasDynLoop;
        estimate.autoQubitManagement = flat.autoQubitManagement;

        // Decomposable gates are keyed by their graph op id (GOID).
        for (const auto &opEntry : flat.detailedOperations) {
            OperatorNode operatorNode;
            operatorNode.id = opEntry.getKey().str();

            auto graphOp = solution.find(operatorNode);
            if (graphOp == solution.end()) {
                // this case should not happen, because the graph should be complete
                // Not a graph operator (e.g. a mid-circuit measurement): count the op itself.
                estimate.operations[opEntry.getKey()][{-1, -1}] += opEntry.getValue();
                continue;
            }
            for (const auto &[basisOp, count] : graphOp->second.basisCounts) {
                // Getting wires and params from GOID is unreliable. The JSON is flattened anyways,
                // so for qp.estimate this will be enough.
                estimate.operations[basisOp.name][{-1, -1}] += count * opEntry.getValue();
            }
        }
        return estimate;
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
