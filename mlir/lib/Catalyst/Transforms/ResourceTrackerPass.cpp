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

#define DEBUG_TYPE "resource-tracker"

#include "llvm/Support/JSON.h"

#include "mlir/Pass/Pass.h"

#include "Catalyst/Analysis/ResourceAnalysis.h"
#include "Catalyst/Analysis/ResourceResult.h"

using namespace mlir;
using namespace llvm;

namespace catalyst {

#define GEN_PASS_DECL_RESOURCETRACKERPASS
#define GEN_PASS_DEF_RESOURCETRACKERPASS
#include "Catalyst/Transforms/Passes.h.inc"

struct ResourceTrackerPass : public impl::ResourceTrackerPassBase<ResourceTrackerPass> {
    using ResourceTrackerPassBase::ResourceTrackerPassBase;

    // Explicit default and copy constructors required because Statistic
    // wraps std::atomic which is non-copyable. MLIR's clonePass() needs
    // the pass to be copyable.
    ResourceTrackerPass() = default;
    ResourceTrackerPass(const ResourceTrackerPass & /*pass*/) : ResourceTrackerPassBase() {}

    Statistic totalGates{this, "total-gates", "Total number of gate operations"};
    Statistic totalMeasurements{this, "total-measurements", "Total number of measurements"};
    Statistic totalQubits{this, "total-qubits", "Total number of qubits allocated"};
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

        if (outputJson) {
            printJsonOutput(results);
        }

        markAllAnalysesPreserved();
    }

  private:
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

        totalQubits += result.numQubits;

        for (const auto &classEntry : result.classicalInstructions) {
            totalClassicalOps += classEntry.getValue();
        }

        for (const auto &fcEntry : result.functionCalls) {
            totalFunctionCalls += fcEntry.getValue();
        }
    }

    void printJsonOutput(const llvm::StringMap<ResourceResult> &results) const
    {
        llvm::json::Object root;

        for (const auto &funcEntry : results) {
            llvm::json::Object funcObj;
            const ResourceResult &result = funcEntry.getValue();

            // Operations
            llvm::json::Object opsObj;
            for (const auto &opEntry : result.operations) {
                StringRef opName = opEntry.getKey();
                for (const auto &sizeEntry : opEntry.getValue()) {
                    int nQubits = sizeEntry.first;
                    int64_t count = sizeEntry.second;
                    std::string key = opName.str() + "(" + std::to_string(nQubits) + ")";
                    opsObj[key] = count;
                }
            }
            funcObj["operations"] = std::move(opsObj);

            // Measurements
            llvm::json::Object measObj;
            for (const auto &entry : result.measurements) {
                measObj[entry.getKey()] = entry.getValue();
            }
            funcObj["measurements"] = std::move(measObj);

            // Classical instructions
            llvm::json::Object classObj;
            for (const auto &entry : result.classicalInstructions) {
                classObj[entry.getKey()] = entry.getValue();
            }
            funcObj["classical_instructions"] = std::move(classObj);

            // Function calls
            llvm::json::Object fcObj;
            for (const auto &entry : result.functionCalls) {
                fcObj[entry.getKey()] = entry.getValue();
            }
            funcObj["function_calls"] = std::move(fcObj);

            funcObj["num_qubits"] = static_cast<int64_t>(result.numQubits);
            funcObj["device_name"] = result.deviceName;

            root[funcEntry.getKey()] = std::move(funcObj);
        }

        // TODO: write to file, when called from frontend. Then, frontend read and delete the file.
        llvm::json::Value jsonValue(std::move(root));
        llvm::outs() << llvm::formatv("{0:2}", jsonValue) << "\n";
    }
};

} // namespace catalyst
