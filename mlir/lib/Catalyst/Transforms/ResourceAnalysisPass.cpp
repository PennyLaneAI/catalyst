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

        // Populate statistics from the entry function. The flattened view
        // walks the structural call graph (function_calls) for us, so we
        // just sum its fields directly.
        StringRef entry = analysis.getEntryFunc();
        if (!entry.empty()) {
            if (const ResourceResult *flat = analysis.getFlattenedResource(entry)) {
                accumulateStats(*flat);
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
    /// Sum a ResourceResult's content into the pass's
    /// Statistic counters. Caller is responsible for choosing whether to
    /// pass a per-function or flattened result.
    void accumulateStats(const ResourceResult &r)
    {
        for (const auto &opEntry : r.operations) {
            for (const auto &sizeEntry : opEntry.getValue()) {
                totalGates += sizeEntry.second;
            }
        }
        for (const auto &measEntry : r.measurements) {
            totalMeasurements += measEntry.getValue();
        }
        for (const auto &classEntry : r.classicalInstructions) {
            totalClassicalOps += classEntry.getValue();
        }
        totalAllocQubits += r.numAllocQubits;
        totalArgQubits += r.numArgQubits;
        totalQubits += r.numQubits();
        for (const auto &fcEntry : r.functionCalls) {
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

        // Store hashes as hex strings so JSON readers don't break high bits
        llvm::json::Object vfcObj;
        for (const auto &entry : result.varFunctionCalls) {
            vfcObj[entry.getKey()] = formatv("{0:x16}", entry.getValue()).str();
        }
        funcObj["var_function_calls"] = std::move(vfcObj);

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
