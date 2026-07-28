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

#include "PBC/Analysis/PBCDepthAnalysis.h"

#include "mlir/IR/Diagnostics.h"

#include "Catalyst/Analysis/ResourceAnalysisRegistry.h"
#include "Catalyst/Analysis/ResourceResult.h"
#include "PBC/Utils/PBCLayer.h"

using namespace mlir;

namespace catalyst {
namespace pbc {

llvm::json::Value PBCDepthExtension::toJson() const {
    llvm::json::Object depthObj;
    if (depth) {
        depthObj["any_commuting_depth"] = depth->first;
        depthObj["qubit_disjoint_depth"] = depth->second;
        return depthObj;
    }
    else {
        return nullptr;
    }
}

void PBCDepthExtension::mergeWith(const ResourceResultExtension &other, MergeMethod mergeMethod) {
    const auto &o = static_cast<const PBCDepthExtension &>(other);
    if (!o.depth) {
        return;
    }
    if (!depth) {
        depth = o.depth;
        return;
    }
    switch (mergeMethod) {
    case MergeMethod::Max:
        depth = {
            {std::max(depth->first, o.depth->first), std::max(depth->second, o.depth->second)}};
        break;
    case MergeMethod::Min:
        depth = {
            {std::min(depth->first, o.depth->first), std::min(depth->second, o.depth->second)}};
        break;
    case MergeMethod::Sum:
        depth = {{depth->first + o.depth->first, depth->second + o.depth->second}};
        break;
    }
}

void PBCDepthExtension::multiplyBy(double factor) {
    if (depth) {
        depth->first *= factor;
        depth->second *= factor;
    }
}

void PBCDepthAnalysis::analyze(Region &region, PBCDepthExtension &ext, bool /*isAdjoint*/) {
    Block *block = &region.front();

    // Swallow expected errors from dynamic-loop depth computation.
    ScopedDiagnosticHandler depthDiagHandler(
        block->getParentOp()->getContext(), [](Diagnostic &diag) {
            if (diag.getSeverity() == DiagnosticSeverity::Error &&
                diag.str().find("worst-case depth") != std::string::npos) {
                return success();
            }
            return failure();
        });

    PBCLayerContext layerContext;
    ext.setDepth(layerContext.computePBCDepth(block));
}

void registerPBCResourceAnalysisExtensions() {
    [[maybe_unused]] static const bool once = [] {
        ResourceAnalysisRegistry::get().add([] { return std::make_unique<PBCDepthAnalysis>(); });
        return true;
    }();
}

} // namespace pbc
} // namespace catalyst
