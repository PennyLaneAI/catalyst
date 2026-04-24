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

#define DEBUG_TYPE "register-decomp-rule-resource"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Pass/Pass.h"

#include "Catalyst/Analysis/ResourceAnalysis.h"
#include "Catalyst/Analysis/ResourceResult.h"

using namespace mlir;
using namespace llvm;

namespace catalyst {

#define GEN_PASS_DECL_REGISTERDECOMPRULERESOURCEPASS
#define GEN_PASS_DEF_REGISTERDECOMPRULERESOURCEPASS
#include "Catalyst/Transforms/Passes.h.inc"

struct RegisterDecompRuleResourcePass
    : public impl::RegisterDecompRuleResourcePassBase<RegisterDecompRuleResourcePass> {
    using RegisterDecompRuleResourcePassBase::RegisterDecompRuleResourcePassBase;

    void runOnOperation() final
    {
        auto &analysis = getAnalysis<ResourceAnalysis>();
        const auto &results = analysis.getResults();
        auto module = llvm::cast<mlir::ModuleOp>(getOperation());

        MLIRContext *ctx = &getContext();
        OpBuilder builder(ctx);

        // Annotate decomposition rule functions with their resource counts, so that
        // the resources can be queried by other passes include graph-decomposition.
        // Note this pass only annotates functions that are marked with "target_gate"
        // attribute, which is how we identify decomposition rules in the main module.
        // Regular or nested functions will not be annotated.
        for (auto func : module.getOps<mlir::func::FuncOp>()) {
            if (!func->hasAttr("target_gate")) {
                continue;
            }

            StringRef funcName = func.getName();
            if (results.count(funcName)) {
                const ResourceResult &result = results.lookup(funcName);
                func->setAttr("resources", buildResourceDict(ctx, result));
            }
        }

        markAllAnalysesPreserved();
    }

  private:
    /**
     * @brief Build a DictionaryAttr from a ResourceResult for annotating functions.
     *
     * The structure of the DictionaryAttr will mirror the JSON output,
     * but with MLIR attributes.
     *
     * Note that this is a simplified version of the ResourceResult
     * only including operations, measurements, and num_qubits,
     * but it can be extended to include more fields such as
     * classical instructions and function calls as needed
     * for the decomposition framework.
     *
     * @param ctx MLIRContext for creating attributes
     * @param result The ResourceResult to convert into attributes
     * @return DictionaryAttr representing the resource counts
     *
     */
    DictionaryAttr buildResourceDict(MLIRContext *ctx, const ResourceResult &result) const
    {
        SmallVector<NamedAttribute> entries;

        // operations
        SmallVector<NamedAttribute> opsEntries;
        for (const auto &opEntry : result.operations) {
            llvm::StringRef opName = opEntry.getKey();
            for (const auto &sizeEntry : opEntry.getValue()) {
                const auto &[nQubits, nParams] = sizeEntry.first;
                int64_t count = sizeEntry.second;
                std::string key = (opName + "(" + std::to_string(nQubits) + ", " + std::to_string(nParams) + ")").str();
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

        // scalars
        entries.push_back(
            NamedAttribute(StringAttr::get(ctx, "num_qubits"),
                           IntegerAttr::get(IntegerType::get(ctx, 64), result.numQubits())));
        entries.push_back(
            NamedAttribute(StringAttr::get(ctx, "num_arg_qubits"),
                           IntegerAttr::get(IntegerType::get(ctx, 64), result.numArgQubits)));
        entries.push_back(
            NamedAttribute(StringAttr::get(ctx, "num_alloc_qubits"),
                           IntegerAttr::get(IntegerType::get(ctx, 64), result.numAllocQubits)));

        return DictionaryAttr::get(ctx, entries);
    }
};

} // namespace catalyst
