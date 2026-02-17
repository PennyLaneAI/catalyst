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

#pragma once

#include "llvm/ADT/StringMap.h"

#include "PBC/IR/PBCOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Operation.h"

#include "Catalyst/Analysis/ResourceResult.h"

using namespace mlir;

namespace catalyst {

class ResourceAnalysis {
  public:
    // walk all func::FuncOps within the operation.
    explicit ResourceAnalysis(mlir::Operation *op);

    const llvm::StringMap<ResourceResult> &getResults() const { return funcResults; }

    const ResourceResult *getResult(llvm::StringRef funcName) const
    {
        auto it = funcResults.find(funcName);
        if (it == funcResults.end()) {
            return nullptr;
        }
        return &it->second;
    }

    // get the entry function name (the function marked with "qnode")
    // returns empty string if no entry function was found
    llvm::StringRef getEntryFunc() const { return entryFuncName; }

  private:
    // per-function resource counts
    llvm::StringMap<ResourceResult> funcResults;

    // name of the entry function (marked with "qnode"), empty if none
    std::string entryFuncName;

    // analyze a region and accumulate results
    void analyzeRegion(Region &region, ResourceResult &result, bool isAdjoint);

    void analyzeForLoop(scf::ForOp forOp, ResourceResult &result, bool isAdjoint);
    void analyzeWhileLoop(scf::WhileOp whileOp, ResourceResult &result, bool isAdjoint);
    void analyzeIfOp(scf::IfOp ifOp, ResourceResult &result, bool isAdjoint);
    void analyzeIndexSwitchOp(scf::IndexSwitchOp switchOp, ResourceResult &result, bool isAdjoint);
    void analyzePBCLayer(pbc::LayerOp layerOp, ResourceResult &result, bool isAdjoint);

    // categorize and count a single operation
    void collectOperation(Operation *op, ResourceResult &result, bool isAdjoint);

    // resolve function calls: inline callee resources into caller
    void resolveFunctionCalls(llvm::StringRef funcName);
};

} // namespace catalyst
