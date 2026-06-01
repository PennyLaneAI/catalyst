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

#include "llvm/ADT/SmallSet.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace catalyst {

#define GEN_PASS_DEF_MARKENTRYPOINTSPASS
#include "Catalyst/Transforms/Passes.h.inc"

namespace {

struct MarkEntryPointsPass : impl::MarkEntryPointsPassBase<MarkEntryPointsPass> {
    using MarkEntryPointsPassBase::MarkEntryPointsPassBase;

    void runOnOperation() final
    {
        ModuleOp root = getOperation();
        UnitAttr entryAttr = UnitAttr::get(&getContext());

        // Collect leaf names referenced from outside any nested module, then
        // stamp matching functions inside each nested module. The host's own
        // entry is annotated by the frontend, not this pass.
        llvm::SmallSet<StringRef, 8> exposed;
        for (Operation &topOp : root.getBody()->getOperations()) {
            if (isa<ModuleOp>(&topOp)) {
                continue;
            }
            if (auto uses = SymbolTable::getSymbolUses(&topOp)) {
                for (SymbolTable::SymbolUse use : *uses) {
                    exposed.insert(use.getSymbolRef().getLeafReference().getValue());
                }
            }
        }
        for (Operation &topOp : root.getBody()->getOperations()) {
            auto mod = dyn_cast<ModuleOp>(&topOp);
            if (!mod) {
                continue;
            }
            for (auto func : mod.getOps<func::FuncOp>()) {
                if (exposed.contains(func.getName())) {
                    func->setAttr("catalyst.entry_point", entryAttr);
                }
            }
        }
    }
};

} // namespace

} // namespace catalyst
