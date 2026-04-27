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

#define DEBUG_TYPE "resource-analysis"

#include <chrono>
#include <fstream>
#include <string>

#include "llvm/Support/JSON.h"
#include "mlir/Pass/Pass.h"

#include "Catalyst/Analysis/ResourceAnalysis.h"
#include "Catalyst/Analysis/ResourceResult.h"

using namespace mlir;
using namespace llvm;

namespace catalyst {

#define GEN_PASS_DECL_RESOURCEANALYSISPASS
#define GEN_PASS_DEF_RESOURCEANALYSISPASS
#include "Catalyst/Transforms/Passes.h.inc"

struct ResourceAnalysisPass : public impl::ResourceAnalysisPassBase<ResourceAnalysisPass> {
    using ResourceAnalysisPassBase::ResourceAnalysisPassBase;

    // Explicit default and copy constructors required because Statistic
    // wraps std::atomic which is non-copyable. MLIR's clonePass() needs
    // the pass to be copyable.
    ResourceAnalysisPass() = default;
    ResourceAnalysisPass(const ResourceAnalysisPass & /*pass*/) : ResourceAnalysisPassBase() {}

    Statistic totalGates{this, "total-gates", "Total number of gate operations"};
    Statistic totalMeasurements{this, "total-measurements", "Total number of measurements"};
    Statistic totalQubits{this, "total-qubits", "Total number of qubits"};
    Statistic totalAllocQubits{this, "total-alloc-qubits",
                               "Total number of qubits from allocation"};
    Statistic totalArgQubits{this, "total-arg-qubits",
                             "Total number of qubits from function arguments"};
    Statistic totalClassicalOps{this, "total-classical-ops",
                                "Total number of classical instructions"};
    Statistic totalFunctionCalls{this, "total-function-calls", "Total number of function calls"};

    void runOnOperation() final
    {
        auto &analysis = getAnalysis<ResourceAnalysis>();
        const auto &results = analysis.getResults();

        // Populate statistics from the entry function only (to avoid
        // double-counting resolved callees). Fall back to summing all
        // functions when no entry function is present.
        StringRef entry = analysis.getEntryFunc();
        if (!entry.empty()) {
            if (const ResourceResult *r = analysis.getResult(entry)) {
                accumulateStats(*r);
            }
        }
        else {
            for (const auto &funcEntry : results) {
                accumulateStats(funcEntry.getValue());
            }
        }
        std::string jsonStr = "";

        if (outputJson) {
            jsonStr = buildJsonString(results);

            if (outputFname.empty()) {
                printJsonOutput(jsonStr);
            }
            else {
                writeJsonToFile(jsonStr, outputFname);
            }
        }

        markAllAnalysesPreserved();
    }

  private:
    /**
     * @brief Accumulate resource counts from a ResourceResult into the pass statistics.
     *
     * This is used to populate the pass's Statistic members with totals from the ResourceResult.
     *
     * @param result The ResourceResult to accumulate stats from.
     */
    void accumulateStats(const ResourceResult &result)
    {
        for (const auto &opEntry : result.operations) {
            for (const auto &sizeEntry : opEntry.getValue()) {
                totalGates += sizeEntry.second;
            }
        }

        for (const auto &measEntry : result.measurements) {
            totalMeasurements += measEntry.getValue();
        }

        totalQubits += result.numQubits();
        totalAllocQubits += result.numAllocQubits;
        totalArgQubits += result.numArgQubits;

        for (const auto &classEntry : result.classicalInstructions) {
            totalClassicalOps += classEntry.getValue();
        }

        for (const auto &fcEntry : result.functionCalls) {
            totalFunctionCalls += fcEntry.getValue();
        }
    }

    /// Serialize a single ResourceResult into a JSON object.
    static llvm::json::Object resultToJson(const ResourceResult &result)
    {
        llvm::json::Object funcObj;

        llvm::json::Object opsObj;
        for (const auto &opEntry : result.operations) {
            StringRef opName = opEntry.getKey();
            for (const auto &sizeEntry : opEntry.getValue()) {
                const auto &[nQubits, nParams] = sizeEntry.first;
                int64_t count = sizeEntry.second;
                std::string key = opName.str() + "(" + std::to_string(nQubits) + ")";
                opsObj[key] = count;
            }
        }
        funcObj["operations"] = std::move(opsObj);

        llvm::json::Object measObj;
        for (const auto &entry : result.measurements) {
            measObj[entry.getKey()] = entry.getValue();
        }
        funcObj["measurements"] = std::move(measObj);

        llvm::json::Object classObj;
        for (const auto &entry : result.classicalInstructions) {
            classObj[entry.getKey()] = entry.getValue();
        }
        funcObj["classical_instructions"] = std::move(classObj);

        llvm::json::Object fcObj;
        for (const auto &entry : result.functionCalls) {
            fcObj[entry.getKey()] = entry.getValue();
        }
        funcObj["function_calls"] = std::move(fcObj);

        funcObj["num_qubits"] = static_cast<int64_t>(result.numQubits());
        funcObj["num_alloc_qubits"] = static_cast<int64_t>(result.numAllocQubits);
        funcObj["num_arg_qubits"] = static_cast<int64_t>(result.numArgQubits);
        funcObj["device_name"] = result.deviceName;
        funcObj["qnode"] = result.isQnode;
        funcObj["has_branches"] = result.hasBranches;
        funcObj["has_dyn_loop"] = result.hasDynLoop;

        return funcObj;
    }

    /// Serialize all per-function ResourceResults into a JSON string.
    /// qnode functions are inserted first so that the PennyLane reader
    /// (which uses the first entry) picks the correct function.
    std::string buildJsonString(const llvm::StringMap<ResourceResult> &results) const
    {
        llvm::json::Object root;

        for (const auto &funcEntry : results) {
            if (funcEntry.getValue().isQnode) {
                root[funcEntry.getKey()] = resultToJson(funcEntry.getValue());
            }
        }
        for (const auto &funcEntry : results) {
            if (!funcEntry.getValue().isQnode) {
                root[funcEntry.getKey()] = resultToJson(funcEntry.getValue());
            }
        }

        llvm::json::Value jsonValue(std::move(root));
        return llvm::formatv("{0:2}", jsonValue).str() + "\n";
    }

    /// Print JSON to stdout.
    void printJsonOutput(const std::string &jsonStr) const { llvm::outs() << jsonStr; }

    /// Write JSON to a file.
    static void writeJsonToFile(const std::string &jsonStr, const std::string &fileName)
    {
        std::ofstream ofile(fileName);
        if (!ofile.is_open()) {
            llvm::errs() << "Error: could not open resource output file: " << fileName << "\n";
            return;
        }
        ofile << jsonStr;
        ofile.close();
    }
};

} // namespace catalyst
