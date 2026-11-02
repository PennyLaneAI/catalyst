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

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/CallInterfaces.h"

#include "Catalyst/Analysis/ResourceAnalysisRegistry.h"
#include "Catalyst/Analysis/ResourceResult.h"
#include "PBC/IR/PBCOpInterfaces.h"
#include "PBC/IR/PBCOps.h"
#include "PBC/Utils/PBCDepth.h"
#include "PBC/Utils/PBCLayer.h"
#include "PBC/Utils/PBCWireIndex.h"

using namespace mlir;

namespace catalyst {
namespace pbc {

namespace {

// Captured QNode parameters live in the calling wrapper. Retain constants shared
// by every visible call site rather than analyzing those captures as unknowns.
llvm::SmallVector<Attribute> capturedArguments(Block *block) {
    auto function = dyn_cast<func::FuncOp>(block->getParentOp());
    if (!function || !function->hasAttr("quantum.node")) {
        return {};
    }
    Operation *root = function;
    while (root->getParentOp()) {
        root = root->getParentOp();
    }
    llvm::SmallVector<Attribute> result;
    bool invalid = false;
    root->walk([&](CallOpInterface call) {
        if (call.resolveCallable() != function) {
            return;
        }
        auto operands = call.getArgOperands();
        if (operands.size() != block->getNumArguments()) {
            invalid = true;
            return;
        }
        WireIndexResolver resolver;
        llvm::SmallVector<Attribute> arguments;
        for (Value operand : operands) {
            arguments.push_back(resolver.resolveConstant(operand));
        }
        if (result.empty()) {
            result = std::move(arguments);
        } else {
            for (auto [known, argument] : llvm::zip(result, arguments)) {
                if (known != argument) {
                    known = {};
                }
            }
        }
    });
    return invalid ? llvm::SmallVector<Attribute>{} : result;
}

} // namespace

llvm::json::Value PBCDepthExtension::toJson() const {
    llvm::json::Object depthObj;
    if (depth || hasQubitDisjointDepthAnalysis) {
        // Preserve the exclusive fields consumed by existing clients. A region containing
        // only calls/loops has no local depth, but can still have an inclusive total.
        depthObj["any_commuting_depth"] = depth ? depth->first : 0;
        depthObj["qubit_disjoint_depth"] = depth ? depth->second : 0;
        depthObj["qubit_disjoint_depth_status"] = qubitDisjointDepthTotal ? "exact" : "unavailable";
        if (qubitDisjointDepthTotal) {
            depthObj["qubit_disjoint_depth_total"] = *qubitDisjointDepthTotal;
        }
        return depthObj;
    }
    return nullptr;
}

void PBCDepthExtension::mergeWith(const ResourceResultExtension &other, MergeMethod mergeMethod) {
    qubitDisjointDepthTotal.reset();
    const auto &o = static_cast<const PBCDepthExtension &>(other);
    hasQubitDisjointDepthAnalysis |= o.hasQubitDisjointDepthAnalysis;
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
    qubitDisjointDepthTotal.reset();
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
    bool hasLocalPBC = false;
    block->walk(
        [&](Operation *op) { hasLocalPBC |= isa<PBCOpInterface, SelectPPMeasurementOp>(op); });
    ext.setQubitDisjointDepthTotal(computeQubitDisjointDepthTotal(block, capturedArguments(block)),
                                   hasLocalPBC);
}

void registerPBCResourceAnalysisExtensions() {
    [[maybe_unused]] static const bool once = [] {
        ResourceAnalysisRegistry::get().add([] { return std::make_unique<PBCDepthAnalysis>(); });
        return true;
    }();
}

} // namespace pbc
} // namespace catalyst
