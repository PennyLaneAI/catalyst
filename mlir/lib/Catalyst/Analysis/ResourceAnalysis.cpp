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

#define DEBUG_TYPE "resource-analysis"

#include "Catalyst/Analysis/ResourceAnalysis.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Operation.h"

#include "Catalyst/Analysis/ResourceResult.h"
#include "Catalyst/Utils/ConstantResolve.h"
#include "MBQC/IR/MBQCOps.h"
#include "PBC/IR/PBCOps.h"
#include "Quantum/IR/QuantumInterfaces.h"
#include "Quantum/IR/QuantumOps.h"

using namespace mlir;
using namespace llvm;

namespace catalyst {

//===----------------------------------------------------------------------===//
// Skipped operations set (mirrors Python _SKIPPED_OPS)
//===----------------------------------------------------------------------===//

static bool isSkippedOp(Operation *op)
{
    return isa<quantum::ComputationalBasisOp, quantum::DeallocOp, quantum::DeallocQubitOp,
               quantum::DeviceReleaseOp, quantum::ExtractOp, quantum::FinalizeOp,
               quantum::HamiltonianOp, quantum::HermitianOp, quantum::InitializeOp,
               quantum::InsertOp, quantum::NamedObsOp, quantum::NumQubitsOp, quantum::TensorOp,
               quantum::YieldOp, pbc::YieldOp>(op);
}

/// Check if the operation belongs to one of the tracked quantum dialects.
static bool isCustomDialectOp(Operation *op)
{
    mlir::Dialect *dialect = op->getDialect();
    if (!dialect) {
        return false;
    }
    return isa<quantum::QuantumDialect, pbc::PBCDialect, mbqc::MBQCDialect>(dialect);
}

//===----------------------------------------------------------------------===//
// Operation categorization helpers
//===----------------------------------------------------------------------===//

/// Get the name to use for a quantum gate op.
static std::string getGateOpName(Operation *op, bool isAdjoint)
{
    std::string name =
        llvm::TypeSwitch<Operation *, std::string>(op)
            .Case<quantum::CustomOp>([](auto customOp) { return customOp.getGateName().str(); })
            .Case<quantum::PauliRotOp>([](auto) { return "PauliRot"; })
            .Case<quantum::GlobalPhaseOp>([](auto) { return "GlobalPhase"; })
            .Case<quantum::MultiRZOp>([](auto) { return "MultiRZ"; })
            .Case<quantum::PCPhaseOp>([](auto) { return "PCPhase"; })
            .Case<quantum::QubitUnitaryOp>([](auto) { return "QubitUnitary"; })
            .Case<quantum::SetStateOp>([](auto) { return "SetState"; })
            .Case<quantum::SetBasisStateOp>([](auto) { return "SetBasisState"; })
            .Default([](Operation *o) { return o->getName().getStringRef().str(); });

    // Combine region-level adjoint (from quantum.adjoint) with
    // op-level adjoint flag (from the adj attribute on the gate).
    if (auto gate = dyn_cast<quantum::QuantumGate>(op)) {
        isAdjoint ^= gate.getAdjointFlag();
    }

    if (isAdjoint) {
        name = "Adjoint(" + name + ")";
    }
    return name;
}

/// Get the number of qubits for a gate operation.
static int getGateQubitCount(Operation *op)
{
    if (auto qOp = dyn_cast<quantum::QuantumOperation>(op)) {
        return static_cast<int>(qOp.getQubitOperands().size());
    }
    return 0;
}

/// Get the number of parameters for a gate operation.
static int getGateParamCount(Operation *op)
{
    if (auto gate = dyn_cast<quantum::ParametrizedGate>(op)) {
        return static_cast<int>(gate.getAllParams().size());
    }
    return 0;
}

/// Get the name for a PBC operation.
static std::string getPBCOpName(Operation *op)
{
    return llvm::TypeSwitch<Operation *, std::string>(op)
        .Case<pbc::PPRotationOp>([](auto pprOp) -> std::string {
            int8_t rk = pprOp.getRotationKind();
            if (rk == 0) {
                return "PPR-identity";
            }
            return "PPR-pi/" + std::to_string(std::abs(rk));
        })
        .Case<pbc::PPRotationArbitraryOp>([](auto) -> std::string { return "PPR-Phi"; })
        .Case<pbc::PPMeasurementOp, pbc::SelectPPMeasurementOp>(
            [](auto) -> std::string { return "PPM"; })
        .Default([](Operation *o) { return o->getName().getStringRef().str(); });
}

/// Get the qubit count for a PBC operation.
static int getPBCQubitCount(Operation *op)
{
    // if the operation is one of these operations, it will return the number of qubits in the input
    return llvm::TypeSwitch<Operation *, int>(op)
        .Case<pbc::PPRotationOp, pbc::PPRotationArbitraryOp, pbc::PPMeasurementOp,
              pbc::SelectPPMeasurementOp>(
            [](auto typedOp) { return static_cast<int>(typedOp.getInQubits().size()); })
        .Default(0);
}

/// Resolve the observable name from its defining operation.
/// Mirrors the Python `xdsl_to_qml_measurement_name` in xdsl_conversion.py.
static std::string getObservableName(Operation *obsOp)
{
    if (!obsOp) {
        return "all wires";
    }

    return llvm::TypeSwitch<Operation *, std::string>(obsOp)
        .Case<quantum::ComputationalBasisOp>([](auto cbOp) {
            unsigned n = cbOp.getQubits().size();
            return n == 0 ? std::string("all wires") : std::to_string(n) + " wires";
        })
        .Case<quantum::NamedObsOp>(
            [](auto op) { return stringifyNamedObservable(op.getType()).str(); })
        .Case<quantum::TensorOp>(
            [](auto op) { return "Prod(num_terms=" + std::to_string(op.getTerms().size()) + ")"; })
        .Case<quantum::HamiltonianOp>([](auto op) {
            return "Hamiltonian(num_terms=" + std::to_string(op.getTerms().size()) + ")";
        })
        .Default([](Operation *) { return std::string("all wires"); });
}

/// Get the full measurement name including observable info.
/// e.g. "MidCircuitMeasure", "expval(PauliZ)", "sample(all wires)", "probs(2 wires)".
static std::string getMeasurementName(Operation *op)
{
    if (isa<quantum::MeasureOp>(op)) {
        return "MidCircuitMeasure";
    }

    std::string baseName =
        llvm::TypeSwitch<Operation *, std::string>(op)
            .Case<quantum::SampleOp>([](auto) { return "sample"; })
            .Case<quantum::CountsOp>([](auto) { return "counts"; })
            .Case<quantum::ExpvalOp>([](auto) { return "expval"; })
            .Case<quantum::VarianceOp>([](auto) { return "var"; })
            .Case<quantum::ProbsOp>([](auto) { return "probs"; })
            .Case<quantum::StateOp>([](auto) { return "state"; })
            .Default([](Operation *o) { return o->getName().getStringRef().str(); });

    if (auto measProc = dyn_cast<quantum::MeasurementProcess>(op)) {
        return baseName + "(" + getObservableName(measProc.getObs().getDefiningOp()) + ")";
    }

    return baseName;
}

//===----------------------------------------------------------------------===//
// ResourceAnalysis implementation
//===----------------------------------------------------------------------===//

ResourceAnalysis::ResourceAnalysis(ModuleOp moduleOp)
{
    LLVM_DEBUG(dbgs() << "ResourceAnalysis: analyzing operation " << moduleOp->getName() << "\n");

    // Reserve every user function's name in `funcResults`. This
    // ensures `makeUniqueSyntheticName` will skip past names like
    // `for_loop_3` that the user already chose, regardless of walk order.
    for (auto funcOp : moduleOp.getOps<func::FuncOp>()) {
        if (funcOp.isDeclaration()) {
            return;
        }
        funcResults.try_emplace(funcOp.getName());
    };

    StringRef entryFunc;
    for (auto funcOp : moduleOp.getOps<func::FuncOp>()) {
        if (funcOp.isDeclaration()) {
            return;
        }

        ResourceResult result;
        for (auto &region : funcOp->getRegions()) {
            analyzeRegion(region, result, /*isAdjoint=*/false);
        }

        // count qubit arguments only for the entry function (first non-declaration)
        // to avoid double-counting when callees are folded into a flattened view.
        if (entryFunc.empty()) {
            for (auto argType : funcOp.getArgumentTypes()) {
                if (isa<quantum::QubitType>(argType)) {
                    result.numArgQubits += 1;
                }
            }
        }

        result.isQnode = funcOp->hasAttrOfType<UnitAttr>("quantum.node");
        funcResults[funcOp.getName()] = std::move(result);

        // main/entry function is the first function with no declaration
        if (entryFunc.empty()) {
            entryFunc = funcOp.getName();
        }
    };

    assert(!entryFunc.empty() && "expected at least one non-declaration function");

    entryFuncName = entryFunc.str();
}

std::string ResourceAnalysis::makeUniqueSyntheticName(StringRef prefix, int64_t &counter)
{
    // Bump `counter` until the resulting name does not collide with an
    // existing entry. This protects against user functions named e.g.
    // `for_loop_3` shadowing or being shadowed by a lifted body.
    std::string candidate;
    do {
        candidate = prefix.str() + std::to_string(++counter);
    } while (funcResults.contains(candidate));

    return candidate;
}

void ResourceAnalysis::analyzeForLoop(scf::ForOp forOp, ResourceResult &result, bool isAdjoint)
{
    ResourceResult bodyResult;
    analyzeRegion(forOp.getBodyRegion(), bodyResult, isAdjoint);

    // Try to resolve a static trip count.
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

    if (tripCount.has_value()) {
        // Record the loop body under a new name (for_loop_1, …).
        // The parent stores how many times the loop runs.
        // Later, classical ops from that body are added into the parent, multiplied by that count.
        // The name is always new, so we don't overwrite an old entry.
        std::string name = makeUniqueSyntheticName("for_loop_", forLoopCounter);
        funcResults[name] = std::move(bodyResult);
        result.functionCalls[name] = tripCount.value();
        return;
    }

    // Loop trip count comes from the user's inputs. Record the body under dyn_for_loop_<N>
    // and store a fixed number (hash) so each such loop has its own id in the output.
    std::string name = makeUniqueSyntheticName("dyn_for_loop_", dynForLoopCounter);
    funcResults[name] = std::move(bodyResult);
    result.varFunctionCalls[name] = static_cast<uint64_t>(llvm::hash_value(forOp.getOperation()));
    result.hasDynLoop = true;
}

void ResourceAnalysis::analyzeWhileLoop(scf::WhileOp whileOp, ResourceResult &result,
                                        bool isAdjoint)
{
    ResourceResult bodyResult;
    analyzeRegion(whileOp.getAfter(), bodyResult, isAdjoint);

    if (auto estAttr = whileOp->getAttrOfType<IntegerAttr>("estimated_iterations")) {
        int64_t iters = estAttr.getValue().getSExtValue();
        bodyResult.multiplyByScalar(iters);
    }
    else {
        result.hasDynLoop = true;
    }

    result.mergeWith(bodyResult);
}

void ResourceAnalysis::analyzeIfOp(scf::IfOp ifOp, ResourceResult &result, bool isAdjoint)
{
    result.hasBranches = true;

    ResourceResult thenResult;
    analyzeRegion(ifOp.getThenRegion(), thenResult, isAdjoint);

    if (!ifOp.getElseRegion().empty()) {
        ResourceResult elseResult;
        analyzeRegion(ifOp.getElseRegion(), elseResult, isAdjoint);
        thenResult.mergeWith(elseResult, ResourceResult::MergeMethod::Max);
    }
    result.mergeWith(thenResult);
}

void ResourceAnalysis::analyzeIndexSwitchOp(scf::IndexSwitchOp switchOp, ResourceResult &result,
                                            bool isAdjoint)
{
    result.hasBranches = true;

    ResourceResult maxResult;
    bool first = true;

    for (auto &caseRegion : switchOp.getCaseRegions()) {
        ResourceResult caseResult;
        analyzeRegion(caseRegion, caseResult, isAdjoint);
        if (first) {
            maxResult = std::move(caseResult);
            first = false;
        }
        else {
            maxResult.mergeWith(caseResult, ResourceResult::MergeMethod::Max);
        }
    }

    // default region
    ResourceResult defaultResult;
    analyzeRegion(switchOp.getDefaultRegion(), defaultResult, isAdjoint);
    maxResult.mergeWith(defaultResult, ResourceResult::MergeMethod::Max);

    result.mergeWith(maxResult);
}

void ResourceAnalysis::analyzePBCLayer(pbc::LayerOp layerOp, ResourceResult &result, bool isAdjoint)
{
    for (auto &layerRegion : layerOp->getRegions()) {
        analyzeRegion(layerRegion, result, isAdjoint);
    }
}

/**
 * @brief Analyze a region and accumulate resource counts into the given ResourceResult.
 *
 * This implementation is based on the Python implementation in `specs_collector.py`.
 *
 * @param region The MLIR region to analyze.
 * @param result The ResourceResult to accumulate counts into.
 * @param isAdjoint Whether the current region is under an adjoint (quantum.adjoint) operation.
 */
void ResourceAnalysis::analyzeRegion(Region &region, ResourceResult &result, bool isAdjoint)
{
    for (Block &block : region) {
        for (Operation &op : block) {
            llvm::TypeSwitch<Operation &, void>(op)
                .Case([&](quantum::AdjointOp adjOp) {
                    analyzeRegion(adjOp.getRegion(), result, !isAdjoint);
                })
                .Case([&](mlir::scf::ForOp forLoopOp) {
                    analyzeForLoop(forLoopOp, result, isAdjoint);
                })
                .Case([&](mlir::scf::WhileOp whileOp) {
                    analyzeWhileLoop(whileOp, result, isAdjoint);
                })
                .Case([&](mlir::scf::IfOp ifConditionalOp) {
                    analyzeIfOp(ifConditionalOp, result, isAdjoint);
                })
                .Case([&](mlir::scf::IndexSwitchOp switchOp) {
                    analyzeIndexSwitchOp(switchOp, result, isAdjoint);
                })
                .Case([&](pbc::LayerOp regionLayerOp) {
                    analyzePBCLayer(regionLayerOp, result, isAdjoint);
                })
                .Default([&](Operation &op) {
                    // other operations - do nothing
                });

            collectOperation(&op, result, isAdjoint);
        }
    }
}

/**
 * @brief Collect a single operation into the ResourceResult.
 *
 * This categorizes the operation into gates, measurements, classical instructions,
 * or function calls, and updates the corresponding counts in the ResourceResult.
 *
 * @param op The operation to collect.
 * @param result The ResourceResult to update with the operation's resource usage.
 * @param isAdjoint Whether the current region is under an adjoint (quantum.adjoint) operation.
 */
void ResourceAnalysis::collectOperation(Operation *op, ResourceResult &result, bool isAdjoint)
{
    // Quantum gates
    if (isa<quantum::CustomOp, quantum::PauliRotOp, quantum::GlobalPhaseOp, quantum::MultiRZOp,
            quantum::PCPhaseOp, quantum::QubitUnitaryOp, quantum::SetStateOp,
            quantum::SetBasisStateOp>(op)) {
        std::string name = getGateOpName(op, isAdjoint);
        int nQubits = getGateQubitCount(op);
        int nParams = getGateParamCount(op);
        result.operations[name][{nQubits, nParams}] += 1;
        return;
    }

    // Measurements
    if (isa<quantum::MeasureOp, quantum::SampleOp, quantum::CountsOp, quantum::ExpvalOp,
            quantum::VarianceOp, quantum::ProbsOp, quantum::StateOp>(op)) {
        result.measurements[getMeasurementName(op)] += 1;
        return;
    }

    // PBC operations
    if (isa<pbc::PPRotationOp, pbc::PPRotationArbitraryOp, pbc::PPMeasurementOp,
            pbc::SelectPPMeasurementOp, pbc::PrepareStateOp, pbc::FabricateOp>(op)) {
        std::string name = getPBCOpName(op);
        int nQubits = getPBCQubitCount(op);
        result.operations[name][{nQubits, 0}] += 1;
        return;
    }

    // MBQC operations
    if (isa<mbqc::MeasureInBasisOp, mbqc::GraphStatePrepOp>(op)) {
        std::string name = op->getName().getStringRef().str();
        result.operations[name][{0, 0}] += 1;
        return;
    }

    // Metadata: device init
    if (auto deviceOp = dyn_cast<quantum::DeviceInitOp>(op)) {
        // Extract device name from the op
        result.deviceName = deviceOp.getDeviceName().str();
        return;
    }

    // Metadata: qubit allocations
    if (auto allocOp = dyn_cast<quantum::AllocOp>(op)) {
        uint64_t nqubits = allocOp.getNqubitsAttr().value_or(0);
        result.numAllocQubits += nqubits;
        return;
    }

    // Metadata: qubit allocation
    if (isa<quantum::AllocQubitOp>(op)) {
        result.numAllocQubits += 1;
        return;
    }

    // Function calls
    if (auto callOp = dyn_cast<func::CallOp>(op)) {
        result.functionCalls[callOp.getCallee()] += 1;
        return;
    }

    // Skipped ops
    if (isSkippedOp(op)) {
        return;
    }

    // Other ops from custom dialects: emit a warning so users are aware
    if (isCustomDialectOp(op)) {
        op->emitWarning() << "ResourceAnalysis encountered an unknown operation '" << op->getName()
                          << "' from a tracked dialect. Some resource data may be missing.";
        return;
    }

    result.classicalInstructions[op->getName().getStringRef()] += 1;
}

/**
 * @brief Merge `child`'s quantum content, classical content, and transitive
 * call counts into `flat`, scaled by `count`. Used to fold callee/loop-body
 * contributions into a flattened view.
 *
 * @param flat The ResourceResult to accumulate counts into.
 * @param child The ResourceResult to merge.
 * @param count A scalar to multiply the child's counts by (defaults to 1).
 */
static void accumulateScaled(ResourceResult &flat, const ResourceResult &child, int64_t count = 1)
{
    for (const auto &opEntry : child.operations) {
        auto &innerDst = flat.operations[opEntry.getKey()];
        for (const auto &sizeEntry : opEntry.getValue()) {
            innerDst[sizeEntry.first] += sizeEntry.second * count;
        }
    }
    for (const auto &m : child.measurements) {
        flat.measurements[m.getKey()] += m.getValue() * count;
    }
    for (const auto &ci : child.classicalInstructions) {
        flat.classicalInstructions[ci.getKey()] += ci.getValue() * count;
    }
    for (const auto &fc : child.functionCalls) {
        flat.functionCalls[fc.getKey()] += fc.getValue() * count;
    }
    flat.numAllocQubits += child.numAllocQubits * count;
    flat.hasBranches = flat.hasBranches || child.hasBranches;
    flat.hasDynLoop = flat.hasDynLoop || child.hasDynLoop;
}

/**
 * @brief Recursively flatten `funcName`'s resources through the call graph.
 *
 * Per one invocation of `funcName`, quantum content (operations,
 * measurements, qubit counts) is accumulated from this function and each
 * callee (including `for_loop_<N>` bodies), scaled by its call count.
 * `functionCalls` is also flattened: each entry holds the total number of
 * times that function is invoked during one run of `funcName`, including
 * indirect calls. Summing the values gives the total dynamic invocation
 * count.
 *
 * @param funcName Function whose flattened view to compute.
 * @return Pointer to the cached flattened result, or nullptr if `funcName`
 *         is external or unknown.
 */
const ResourceResult *ResourceAnalysis::getFlattenedResource(StringRef funcName) const
{
    if (auto it = flattenedCache.find(funcName); it != flattenedCache.end()) {
        return &it->second;
    }

    auto srcIt = funcResults.find(funcName);
    if (srcIt == funcResults.end()) {
        return nullptr; // external / unknown function
    }
    const ResourceResult &r = srcIt->second;

    // Cache this name early so circular calls see one shared partial result
    // and don't recurse forever.
    ResourceResult &flat = flattenedCache[funcName];

    // Self-contributions (no scaling).
    accumulateScaled(flat, r);
    flat.numArgQubits = r.numArgQubits; // own arg qubits only
    flat.deviceName = r.deviceName;
    flat.isQnode = r.isQnode;

    // Any var_function_calls implies a dynamic loop with no static count.
    flat.hasDynLoop = flat.hasDynLoop || !r.varFunctionCalls.empty();

    // Pull in each direct callee's flattened counts, × how often we call it.
    for (const auto &fc : r.functionCalls) {
        const ResourceResult *child = getFlattenedResource(fc.getKey());
        if (!child || child == &flat) {
            continue;
        }
        accumulateScaled(flat, *child, fc.getValue());
    }
    return &flat;
}

} // namespace catalyst
