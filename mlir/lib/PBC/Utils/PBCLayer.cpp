// Copyright 2025 Xanadu Quantum Technologies Inc.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "PBC/Utils/PBCLayer.h"

#include <algorithm>

#include "llvm/ADT/STLExtras.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinAttributes.h"

#include "Catalyst/Utils/ConstantResolve.h"
#include "PBC/IR/PBCOpInterfaces.h"
#include "PBC/IR/PBCOps.h"
#include "PBC/Utils/PauliStringWrapper.h"
#include "Quantum/IR/QuantumOps.h" // for quantum.extract op

using namespace mlir;

namespace catalyst {
namespace pbc {

FailureOr<int64_t> PBCLayerContext::ifWorstCaseDepth(scf::IfOp ifOp, bool onlyOnDisjointQubit,
                                                     bool skipDynamic)
{
    FailureOr<int64_t> thenDepth =
        computeBlockWorstCaseDepth(&ifOp.getThenRegion().front(), onlyOnDisjointQubit, skipDynamic);
    if (failed(thenDepth)) {
        return failure();
    }

    int64_t elseDepth = 0;
    if (!ifOp.getElseRegion().empty()) {
        FailureOr<int64_t> elseD = computeBlockWorstCaseDepth(&ifOp.getElseRegion().front(),
                                                              onlyOnDisjointQubit, skipDynamic);
        if (failed(elseD)) {
            return failure();
        }
        elseDepth = *elseD;
    }
    return std::max(*thenDepth, elseDepth);
}

FailureOr<int64_t> PBCLayerContext::switchWorstCaseDepth(scf::IndexSwitchOp switchOp,
                                                         bool onlyOnDisjointQubit, bool skipDynamic)
{
    FailureOr<int64_t> defaultDepth =
        computeBlockWorstCaseDepth(&switchOp.getDefaultBlock(), onlyOnDisjointQubit, skipDynamic);
    if (failed(defaultDepth)) {
        return failure();
    }

    int64_t maxDepth = *defaultDepth;

    for (unsigned i = 0, n = switchOp.getNumCases(); i < n; ++i) {
        FailureOr<int64_t> caseDepth =
            computeBlockWorstCaseDepth(&switchOp.getCaseBlock(i), onlyOnDisjointQubit, skipDynamic);
        if (failed(caseDepth)) {
            return failure();
        }
        maxDepth = std::max(maxDepth, *caseDepth);
    }
    return maxDepth;
}

FailureOr<int64_t> PBCLayerContext::forWorstCaseDepth(scf::ForOp forOp, bool onlyOnDisjointQubit,
                                                      bool skipDynamic)
{
    // Same trip-count rules as resource counting (`estimated_iterations`, static bounds, …).
    std::optional<int64_t> tripCount;
    if (auto estAttr = forOp->getAttrOfType<IntegerAttr>("estimated_iterations")) {
        tripCount = estAttr.getValue().getSExtValue();
    }
    else if (auto sTrip = forOp.getStaticTripCount()) {
        tripCount = sTrip->getSExtValue();
    }
    else {
        auto lb = resolveConstantInt(forOp.getLowerBound());
        auto ub = resolveConstantInt(forOp.getUpperBound());
        auto step = resolveConstantInt(forOp.getStep());
        if (lb && ub && step && *step != 0 && *ub > *lb) {
            tripCount = (*ub - *lb + *step - 1) / *step;
        }
    }

    if (!tripCount) {
        if (skipDynamic) {
            return 0;
        }
        return forOp.emitOpError(
            "worst-case depth is not available when there are dynamically sized for loops");
    }

    FailureOr<int64_t> bodyDepth =
        computeBlockWorstCaseDepth(forOp.getBody(), onlyOnDisjointQubit, skipDynamic);
    if (failed(bodyDepth)) {
        return failure();
    }
    return *tripCount * (*bodyDepth);
}

FailureOr<int64_t> PBCLayerContext::computeBlockWorstCaseDepth(Block *block,
                                                               bool onlyOnDisjointQubit,
                                                               bool skipDynamic)
{
    int64_t depth = 0;
    PBCLayer layer(this);

    auto flushLayer = [&]() {
        if (!layer.empty()) {
            depth += 1;
            layer = PBCLayer(this);
        }
    };

    for (Operation &op : *block) {
        if (isa<scf::WhileOp>(&op)) {
            return op.emitOpError(
                "worst-case depth is not available when PBC ops are inside scf.while");
        }
        if (auto ifOp = dyn_cast<scf::IfOp>(&op)) {
            flushLayer();
            FailureOr<int64_t> branchDepth =
                ifWorstCaseDepth(ifOp, onlyOnDisjointQubit, skipDynamic);
            if (failed(branchDepth)) {
                return failure();
            }
            depth += *branchDepth;
            continue;
        }

        if (auto switchOp = dyn_cast<scf::IndexSwitchOp>(&op)) {
            flushLayer();
            FailureOr<int64_t> branchDepth =
                switchWorstCaseDepth(switchOp, onlyOnDisjointQubit, skipDynamic);
            if (failed(branchDepth)) {
                return failure();
            }
            depth += *branchDepth;
            continue;
        }

        if (auto forOp = dyn_cast<scf::ForOp>(&op)) {
            flushLayer();
            FailureOr<int64_t> loopDepth =
                forWorstCaseDepth(forOp, onlyOnDisjointQubit, skipDynamic);
            if (failed(loopDepth)) {
                return failure();
            }
            depth += *loopDepth;
            continue;
        }

        if (auto pbcOp = dyn_cast<PBCOpInterface>(&op)) {
            if (!layer.insert(pbcOp, onlyOnDisjointQubit)) {
                flushLayer();
                bool inserted = layer.insert(pbcOp, onlyOnDisjointQubit);
                assert(inserted && "PBCLayer::insert must accept any op into an empty layer");
            }
            continue;
        }

        // PPM variants without PBCOpInterface still contribute one layer each.
        if (isa<SelectPPMeasurementOp>(&op)) {
            flushLayer();
            depth += 1;
            continue;
        }

        // plain ops
        if (op.getNumRegions() == 0) {
            continue;
        }

        // multi-region ops
        if (op.getNumRegions() != 1) {
            return op.emitOpError(
                "worst-case depth cannot analyze operations with multiple regions");
        }

        // Recurse into the op's single-block region body (e.g. quantum.adjoint)
        Region &region = op.getRegion(0);
        if (region.empty()) {
            continue;
        }
        if (!region.hasOneBlock()) {
            return op.emitOpError("worst-case depth cannot analyze regions with multiple blocks");
        }

        flushLayer();
        FailureOr<int64_t> innerDepth =
            computeBlockWorstCaseDepth(&region.front(), onlyOnDisjointQubit, skipDynamic);
        if (failed(innerDepth)) {
            return failure();
        }

        depth += *innerDepth;
    }

    flushLayer();
    return depth;
}

PBCDepths PBCLayerContext::computePBCDepth(Block *block)
{
    // Try to calculate the depth with static first, then fallback to skip-dynamic.
    auto d0 = computeBlockWorstCaseDepth(block, /*onlyOnDisjointQubit=*/false);
    if (failed(d0)) {
        auto p0 = computeBlockWorstCaseDepth(block, /*onlyOnDisjointQubit=*/false,
                                             /*skipDynamic=*/true);
        if (failed(p0) || *p0 == 0) {
            return std::nullopt;
        }
        auto p1 = computeBlockWorstCaseDepth(block, /*onlyOnDisjointQubit=*/true,
                                             /*skipDynamic=*/true);
        if (failed(p1) || *p1 == 0) {
            return std::nullopt;
        }
        return {{*p0, *p1}};
    }
    if (*d0 == 0) {
        return std::nullopt;
    }
    auto d1 = computeBlockWorstCaseDepth(block, /*onlyOnDisjointQubit=*/true);
    if (failed(d1) || *d1 == 0) {
        return std::nullopt;
    }
    return {{*d0, *d1}};
}

// Partition PBC ops into layer groups using commutation/disjoint-qubit rules.
// Only op membership is recorded; operand/result bookkeeping is deferred until
// construction so that SSA values reflect any layers already materialized in IR.
llvm::SmallVector<std::vector<PBCOpInterface>>
PBCLayerContext::groupLayers(mlir::Operation *root, bool onlyOnDisjointQubit)
{
    bool hasExistingLayers = false;
    root->walk([&](LayerOp) {
        hasExistingLayers = true;
        return WalkResult::interrupt();
    });
    assert(!hasExistingLayers &&
           "groupLayers expects flat PBC ops; pbc.layer must not exist in IR yet");

    llvm::SmallVector<std::vector<PBCOpInterface>> groups;
    PBCLayer layer(this);
    root->walk([&](PBCOpInterface op) {
        if (layer.insert(op, onlyOnDisjointQubit)) {
            return WalkResult::skip();
        }

        groups.emplace_back(layer.getOps());
        layer = PBCLayer(this);
        layer.insert(op, onlyOnDisjointQubit);

        return WalkResult::advance();
    });

    if (!layer.empty()) {
        groups.emplace_back(layer.getOps());
    }
    return groups;
}

void PBCLayer::insertToLayer(PBCOpInterface op)
{
    ops.emplace_back(op);
    updateResultAndOperand(op);

    // Update the cached entry qubit set when inserting
    auto entryQubits = getEntryQubitsFrom(op);
    layerEntryQubits.insert(entryQubits.begin(), entryQubits.end());

    // Track the op's results for dependency lookups
    for (Value r : op->getResults()) {
        layerOpResults.insert(r);
    }
}

void PBCLayer::eraseOp(PBCOpInterface op)
{
    llvm::erase(ops, op);
    for (Value r : op->getResults()) {
        layerOpResults.erase(r);
    }
}

void PBCLayer::updateResultAndOperand(PBCOpInterface op)
{
    // Ensure layer operand set contains canonical origins for any input qubits
    ValueRange inQubits = op.getInQubits();
    ValueRange outQubits = op.getOutQubits();

    // Map each input qubit to its canonical origin (entry) for this layer
    llvm::SmallVector<Value> inputOrigins;
    inputOrigins.reserve(inQubits.size());

    for (Value in : inQubits) {
        // If the operand was a previously exposed result, remove it from layer results
        if (results.contains(in)) {
            results.remove(in);
        }

        if (resultToOperand.contains(in)) {
            in = resultToOperand[in];
        }
        inputOrigins.push_back(in);
        operands.insert(in);
    }

    // Insert non-qubit results first (e.g., classical measurements)
    for (Value r : op->getResults()) {
        if (!llvm::isa<catalyst::quantum::QubitType>(r.getType())) {
            results.insert(r);
        }
    }

    // Pair qubit outputs with corresponding input origins and update mappings
    for (auto [origin, out] : llvm::zip(inputOrigins, outQubits)) {
        resultToOperand[out] = origin;
        results.insert(out);
    }
}

std::vector<Value> PBCLayer::getEntryQubitsFrom(PBCOpInterface op)
{
    std::vector<Value> entryQubits;
    entryQubits.reserve(op.getInQubits().size());

    for (auto [inQubit, outQubit] : llvm::zip(op.getInQubits(), op.getOutQubits())) {
        // Resolve entry for inQubit within this layer
        // If inQubit is a region argument of this layer, it is the entry;
        // otherwise, if it is produced by a previous op in this layer,
        // we should have mapped its defining value to an entry already.
        // Fallback: use inQubit itself.
        Value entry = localQubitToEntry.contains(inQubit) ? localQubitToEntry[inQubit] : inQubit;

        entryQubits.push_back(entry);
        // Propagate the entry to the result value inside this layer
        localQubitToEntry[outQubit] = entry;
    }

    return entryQubits;
}

std::vector<Value> PBCLayer::getEntryQubitsFrom(YieldOp yieldOp)
{
    std::vector<Value> entries;
    entries.reserve(yieldOp->getNumOperands());
    for (Value yOperand : yieldOp->getOperands()) {
        if (!llvm::isa<catalyst::quantum::QubitType>(yOperand.getType())) {
            continue;
        }
        Value entry = yOperand;
        if (localQubitToEntry.contains(yOperand)) {
            entry = localQubitToEntry[yOperand];
        }
        else if (resultToOperand.contains(yOperand)) {
            entry = resultToOperand[yOperand];
        }
        entries.push_back(entry);
    }
    return entries;
}

bool PBCLayer::actOnDisjointQubits(PBCOpInterface op)
{
    // Check for overlap with cached index set
    return llvm::none_of(getEntryQubitsFrom(op),
                         [&](const auto &q) { return layerEntryQubits.contains(q); });
}

// Commute two ops if they act on the same qubits based on qubit indexes on that layer
bool PBCLayer::commute(PBCOpInterface src, PBCOpInterface dst)
{
    auto srcEntryQubits = getEntryQubitsFrom(src);
    auto dstEntryQubits = getEntryQubitsFrom(dst);

    auto normalizedOps = normalizePPROps(src, dst, srcEntryQubits, dstEntryQubits);

    return normalizedOps.first.commutes(normalizedOps.second);
}

// Commute an op to all the ops in the layer
bool PBCLayer::commuteToLayer(PBCOpInterface op)
{
    return llvm::all_of(ops, [&](auto existingOp) { return commute(op, existingOp); });
}

bool PBCLayer::isSameBlock(PBCOpInterface op) const
{
    if (ops.empty()) {
        return true;
    }
    return op->getBlock() == ops.back()->getBlock();
}

bool PBCLayer::dependsOnLayerOps(mlir::Value value) const
{
    if (layerOpResults.empty()) {
        return false;
    }

    llvm::SmallVector<Value> worklist = {value};
    llvm::DenseSet<Value> visited;

    while (!worklist.empty()) {
        Value current = worklist.pop_back_val();
        if (!visited.insert(current).second) {
            continue;
        }
        if (layerOpResults.contains(current)) {
            return true;
        }
        Operation *defOp = current.getDefiningOp();
        if (!defOp) {
            continue; // block argument
        }
        for (Value operand : defOp->getOperands()) {
            worklist.push_back(operand);
        }
    }
    return false;
}

bool PBCLayer::extractOperandsDependOnLayerOps(PBCOpInterface op) const
{
    for (mlir::Value operand : op->getOperands()) {
        mlir::Operation *defOp = operand.getDefiningOp();
        if (llvm::isa_and_nonnull<quantum::ExtractOp>(defOp)) {
            for (mlir::Value extractOperand : defOp->getOperands()) {
                if (dependsOnLayerOps(extractOperand)) {
                    return true;
                }
            }
        }
    }
    return false;
}

bool PBCLayer::insert(PBCOpInterface op, bool onlyDisjointQubit)
{
    if (empty()) {
        insertToLayer(op);
        return true;
    }

    // 1. It is in the same block
    // 2. No operand depends on a layer op result through an insert→extract chain
    if (!isSameBlock(op) || extractOperandsDependOnLayerOps(op)) {
        return false;
    }

    // 3. It acts on disjoint qubits
    // 4. Or it commutes with all the ops in the layer
    if (actOnDisjointQubits(op)) {
        insertToLayer(op);
        return true;
    }

    // If onlyOnDisjointQubit is true, we only check the disjoint qubit condition
    if (!onlyDisjointQubit && commuteToLayer(op)) {
        insertToLayer(op);
        return true;
    }

    return false;
}

llvm::SmallVector<mlir::Value> PBCLayer::getResultsOrderedByTypeThenOperand() const
{
    llvm::SmallVector<mlir::Value> ordered;

    // 1. Collect classical first in program order
    for (const auto &v : results) {
        if (!llvm::isa<catalyst::quantum::QubitType>(v.getType())) {
            ordered.push_back(v);
        }
    }

    // 2. Group qubit results by their originating operand (entry qubit) order
    // Build buckets: origin -> list of results (preserve result order)
    llvm::DenseMap<mlir::Value, llvm::SmallVector<mlir::Value>> originToQubitResults;
    originToQubitResults.reserve(results.size());
    for (const auto &v : results) {
        if (!llvm::isa<catalyst::quantum::QubitType>(v.getType())) {
            continue;
        }
        if (auto it = resultToOperand.find(v); it != resultToOperand.end()) {
            originToQubitResults[it->second].push_back(v);
        }
    }

    for (const auto &origin : operands) {
        if (auto it = originToQubitResults.find(origin); it != originToQubitResults.end()) {
            ordered.append(it->second.begin(), it->second.end());
        }
    }

    return ordered;
}

} // namespace pbc
} // namespace catalyst
