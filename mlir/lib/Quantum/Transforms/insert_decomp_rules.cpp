// Copyright 2023 Xanadu Quantum Technologies Inc.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#define DEBUG_TYPE "insert-decomp-rules"

#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"

namespace catalyst {
namespace quantum {

#define GEN_PASS_DEF_INSERTDECOMPRULESPASS
#define GEN_PASS_DECL_INSERTDECOMPRULESPASS
#include "Quantum/Transforms/Passes.h.inc"

static constexpr std::string_view AOT_RULES_FILE = "./aot_rule_cache.mlirbc";

mlir::OwningOpRef<mlir::func::FuncOp> readMLIRBCFunc(mlir::MLIRContext *context)
{
    llvm::StringRef fileRef(AOT_RULES_FILE.data(), AOT_RULES_FILE.size());

    mlir::ParserConfig config(context);
    mlir::OwningOpRef<mlir::ModuleOp> moduleOp =
        mlir::parseSourceFile<mlir::ModuleOp>(fileRef, config);

    if (!moduleOp) {
        return nullptr;
    }

    auto funcOps = moduleOp->getOps<mlir::func::FuncOp>();

    if (funcOps.empty()) {
        return nullptr;
    }

    mlir::func::FuncOp funcOp = *funcOps.begin();

    if (!funcOp) {
        return nullptr;
    }

    funcOp->remove();

    return mlir::OwningOpRef<mlir::func::FuncOp>(funcOp);
}

struct InsertDecompRulesPass : public impl::InsertDecompRulesPassBase<InsertDecompRulesPass> {
    using InsertDecompRulesPassBase::InsertDecompRulesPassBase;

    void runOnOperation() final
    {
        LLVM_DEBUG(llvm::dbgs() << "insert decomposition rules"
                                << "\n");

        mlir::ModuleOp module = getOperation();

        mlir::OwningOpRef<mlir::func::FuncOp> funcOpRef = readMLIRBCFunc(module.getContext());
        if (!funcOpRef) {
            return signalPassFailure();
        }

        module.getBody()->push_back(funcOpRef.release());
    }
};

} // namespace quantum
} // namespace catalyst
