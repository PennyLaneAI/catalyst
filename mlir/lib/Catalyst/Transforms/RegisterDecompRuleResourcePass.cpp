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
        auto &analysis = getAnalysis<ResourceAnalysis, ModuleOp>();
        auto module = llvm::cast<mlir::ModuleOp>(getOperation());

        MLIRContext *ctx = &getContext();

        // Annotate decomposition-rule functions (those tagged with
        // "target_gate") with their flattened resource counts. The analysis
        // owns the memoization cache, so callees shared across multiple
        // rules are flattened only once.
        for (auto func : module.getOps<mlir::func::FuncOp>()) {
            if (!func->hasAttr("target_gate")) {
                continue;
            }
            if (const ResourceResult *flat = analysis.getFlattenedResource(func.getName())) {
                func->setAttr("resources", buildResourceDict(ctx, *flat));
            }
        }

        markAllAnalysesPreserved();
    }
};

} // namespace catalyst
