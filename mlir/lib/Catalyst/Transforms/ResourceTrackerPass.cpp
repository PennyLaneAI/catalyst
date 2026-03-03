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

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
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

        if (outputJson) {
            printJsonOutput(results);
        }

        if (updateAttr || decompAttr) {
            auto module = llvm::cast<mlir::ModuleOp>(getOperation());
            MLIRContext *ctx = &getContext();
            OpBuilder builder(ctx);

            for (auto func : module.getOps<mlir::func::FuncOp>()) {
                if (decompAttr && !func->hasAttr("target_gate")) {
                    // Only annotate functions that are decomposition rules
                    // i.e., marked with "target_gate" attribute.
                    continue;
                }

                StringRef funcName = func.getName();
                if (results.count(funcName)) {
                    const ResourceResult &result = results.lookup(funcName);
                    // add a test attr with value name of funcName
                    func->setAttr("resources",
                                  buildResourceDict(ctx, result, /*isDecompAttr*/ decompAttr));
                }
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

    /**
     * @brief Print the resource results as JSON to stdout.
     *
     * The structure of the JSON will mirror the DictionaryAttr built in buildResourceDict(),
     * but with JSON objects and values.
     *
     * @param results The map of function names to ResourceResults to print.
     */
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

            funcObj["num_qubits"] = static_cast<int64_t>(result.numQubits());
            funcObj["num_alloc_qubits"] = static_cast<int64_t>(result.numAllocQubits);
            funcObj["num_arg_qubits"] = static_cast<int64_t>(result.numArgQubits);
            funcObj["device_name"] = result.deviceName;

            root[funcEntry.getKey()] = std::move(funcObj);
        }

        // TODO: write to file, when called from frontend. Then, frontend read and delete the file.
        llvm::json::Value jsonValue(std::move(root));
        llvm::outs() << llvm::formatv("{0:2}", jsonValue) << "\n";
    }

    /**
     * @brief Build a DictionaryAttr from a ResourceResult for annotating functions.
     *
     * The structure of the DictionaryAttr will mirror the JSON output,
     * but with MLIR attributes.
     *
     * If isDecompAttr is true, we omit classical instructionsand function calls
     * from the attributes, as well as the total qubits, since these are not relevant
     * for decomposition rules and would add unnecessary clutter.
     * Decomposition functions are typically focused on the specific gate, mid-circuit
     * measurements and allocated auxiliary qubits.
     *
     * @param ctx MLIRContext for creating attributes
     * @param result The ResourceResult to convert into attributes
     * @param isDecompAttr Whether this is for a decomposition function
     * @return DictionaryAttr representing the resource counts
     *
     */
    DictionaryAttr buildResourceDict(MLIRContext *ctx, const ResourceResult &result,
                                     bool isDecompAttr) const
    {
        SmallVector<NamedAttribute> entries;

        // operations
        SmallVector<NamedAttribute> opsEntries;
        for (const auto &opEntry : result.operations) {
            llvm::StringRef opName = opEntry.getKey();
            for (const auto &sizeEntry : opEntry.getValue()) {
                int nQubits = sizeEntry.first;
                int64_t count = sizeEntry.second;
                std::string key = (opName + "(" + std::to_string(nQubits) + ")").str();
                opsEntries.push_back(NamedAttribute(
                    StringAttr::get(ctx, key), IntegerAttr::get(IntegerType::get(ctx, 64), count)));
            }
        }
        entries.push_back(NamedAttribute(StringAttr::get(ctx, "operations"),
                                         DictionaryAttr::get(ctx, opsEntries)));

        // measurements
        SmallVector<NamedAttribute> measEntries;
        for (const auto &entry : result.measurements) {
            measEntries.push_back(
                NamedAttribute(StringAttr::get(ctx, entry.getKey()),
                               IntegerAttr::get(IntegerType::get(ctx, 64), entry.getValue())));
        }
        entries.push_back(NamedAttribute(StringAttr::get(ctx, "measurements"),
                                         DictionaryAttr::get(ctx, measEntries)));

        if (!decompAttr) {
            // classical instructions
            SmallVector<NamedAttribute> classEntries;
            for (const auto &entry : result.classicalInstructions) {
                classEntries.push_back(
                    NamedAttribute(StringAttr::get(ctx, entry.getKey()),
                                   IntegerAttr::get(IntegerType::get(ctx, 64), entry.getValue())));
            }
            entries.push_back(NamedAttribute(StringAttr::get(ctx, "classical_instructions"),
                                             DictionaryAttr::get(ctx, classEntries)));

            // function calls
            SmallVector<NamedAttribute> fcEntries;
            for (const auto &entry : result.functionCalls) {
                fcEntries.push_back(
                    NamedAttribute(StringAttr::get(ctx, entry.getKey()),
                                   IntegerAttr::get(IntegerType::get(ctx, 64), entry.getValue())));
            }
            entries.push_back(NamedAttribute(StringAttr::get(ctx, "function_calls"),
                                             DictionaryAttr::get(ctx, fcEntries)));

            // scalars
            entries.push_back(
                NamedAttribute(StringAttr::get(ctx, "num_qubits"),
                               IntegerAttr::get(IntegerType::get(ctx, 64), result.numQubits())));
            entries.push_back(
                NamedAttribute(StringAttr::get(ctx, "num_arg_qubits"),
                               IntegerAttr::get(IntegerType::get(ctx, 64), result.numArgQubits)));
            entries.push_back(NamedAttribute(StringAttr::get(ctx, "device_name"),
                                             StringAttr::get(ctx, result.deviceName)));
        }

        entries.push_back(
            NamedAttribute(StringAttr::get(ctx, "num_alloc_qubits"),
                           IntegerAttr::get(IntegerType::get(ctx, 64), result.numAllocQubits)));

        return DictionaryAttr::get(ctx, entries);
    }
};

} // namespace catalyst
