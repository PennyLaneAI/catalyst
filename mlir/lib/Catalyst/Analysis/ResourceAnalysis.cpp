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

#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

#include "Catalyst/Analysis/ResourceAnalysis.h"

#include "MBQC/IR/MBQCOps.h"
#include "QEC/IR/QECOps.h"
#include "Quantum/IR/QuantumOps.h"

#define DEBUG_TYPE "resource-analysis"

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
               quantum::YieldOp, qec::YieldOp>(op);
}

/// Check if the operation belongs to one of the tracked quantum dialects.
static bool isCustomDialectOp(Operation *op)
{
    mlir::Dialect *dialect = op->getDialect();
    if (!dialect)
        return false;
    return isa<quantum::QuantumDialect, qec::QECDialect, mbqc::MBQCDialect>(dialect);
}

//===----------------------------------------------------------------------===//
// Static loop iteration counting (from SCFUtils)
//===----------------------------------------------------------------------===//

static int64_t getIntFromConstant(arith::ConstantOp op)
{
    assert(isa<IntegerAttr>(op.getValue()));
    return cast<IntegerAttr>(op.getValue()).getValue().getSExtValue();
}

/// Count iterations of a scf.for loop. Returns -1 if dynamic bounds.
static int64_t countForLoopIterations(scf::ForOp forOp)
{
    Operation *lbOp = forOp.getLowerBound().getDefiningOp();
    if (!lbOp || !isa<arith::ConstantOp>(lbOp))
        return -1;
    int64_t lb = getIntFromConstant(cast<arith::ConstantOp>(lbOp));

    Operation *ubOp = forOp.getUpperBound().getDefiningOp();
    if (!ubOp || !isa<arith::ConstantOp>(ubOp))
        return -1;
    int64_t ub = getIntFromConstant(cast<arith::ConstantOp>(ubOp));

    Operation *stepOp = forOp.getStep().getDefiningOp();
    if (!stepOp || !isa<arith::ConstantOp>(stepOp))
        return -1;
    int64_t step = getIntFromConstant(cast<arith::ConstantOp>(stepOp));

    if (step <= 0 || ub <= lb)
        return 0;
    return static_cast<int64_t>(std::ceil(static_cast<double>(ub - lb) / step));
}

//===----------------------------------------------------------------------===//
// Operation categorization helpers
//===----------------------------------------------------------------------===//

/// Get the name to use for a quantum gate op.
static std::string getGateOpName(Operation *op, bool adjointMode)
{
    std::string name =
        llvm::TypeSwitch<Operation *, std::string>(op)
            .Case<quantum::CustomOp>([](auto customOp) { return customOp.getGateName().str(); })
            .Case<quantum::PauliRotOp>([](auto) { return "PauliRot"; })
            .Case<quantum::GlobalPhaseOp>([](auto) { return "GlobalPhase"; })
            .Case<quantum::MultiRZOp>([](auto) { return "MultiRZ"; })
            .Case<quantum::QubitUnitaryOp>([](auto) { return "QubitUnitary"; })
            .Case<quantum::SetStateOp>([](auto) { return "SetState"; })
            .Case<quantum::SetBasisStateOp>([](auto) { return "SetBasisState"; })
            .Default([](Operation *o) { return o->getName().getStringRef().str(); });

    // Combine region-level adjoint (from quantum.adjoint) with
    // op-level adjoint flag (from the adj attribute on the gate).
    if (auto gate = dyn_cast<quantum::QuantumGate>(op)) {
        adjointMode ^= gate.getAdjointFlag();
    }

    if (adjointMode) {
        name = "Adjoint(" + name + ")";
    }
    return name;
}

/// Get the number of qubits for a gate operation.
static int getGateQubitCount(Operation *op)
{
    return llvm::TypeSwitch<Operation *, int>(op)
        .Case<quantum::CustomOp>([](auto customOp) {
            return static_cast<int>(customOp.getInQubits().size() +
                                    customOp.getInCtrlQubits().size());
        })
        .Case<quantum::PauliRotOp>(
            [](auto pauliRotOp) { return static_cast<int>(pauliRotOp.getInQubits().size()); })
        .Case<quantum::MultiRZOp>(
            [](auto multiRZOp) { return static_cast<int>(multiRZOp.getInQubits().size()); })
        .Default(0);
}

/// Get the name for a QEC operation.
static std::string getQECOpName(Operation *op)
{
    return llvm::TypeSwitch<Operation *, std::string>(op)
        .Case<qec::PPRotationOp>([](auto pprOp) -> std::string {
            int16_t rk = pprOp.getRotationKindAttr().getValue().getSExtValue();
            if (rk == 0)
                return "PPR-identity";
            return "PPR-pi/" + std::to_string(std::abs(rk));
        })
        .Case<qec::PPRotationArbitraryOp>([](auto) -> std::string { return "PPR-Phi"; })
        .Case<qec::PPMeasurementOp, qec::SelectPPMeasurementOp>(
            [](auto) -> std::string { return "PPM"; })
        .Default([](Operation *o) { return o->getName().getStringRef().str(); });
}

/// Get the qubit count for a QEC operation.
static int getQECQubitCount(Operation *op)
{
    // if the operation is one of these operations, it will return the number of qubits in the input
    return llvm::TypeSwitch<Operation *, int>(op)
        .Case<qec::PPRotationOp, qec::PPRotationArbitraryOp, qec::PPMeasurementOp,
              qec::SelectPPMeasurementOp, qec::PrepareStateOp>(
            [](auto typedOp) { return static_cast<int>(typedOp.getInQubits().size()); })
        .Default(0);
}

/// Get the measurement name for a quantum measurement op.
static std::string getMeasurementName(Operation *op)
{
    return llvm::TypeSwitch<Operation *, std::string>(op)
        .Case<quantum::MeasureOp>([](auto) { return "MidCircuitMeasure"; })
        .Case<quantum::SampleOp>([](auto) { return "sample"; })
        .Case<quantum::CountsOp>([](auto) { return "counts"; })
        .Case<quantum::ExpvalOp>([](auto) { return "expval"; })
        .Case<quantum::VarianceOp>([](auto) { return "var"; })
        .Case<quantum::ProbsOp>([](auto) { return "probs"; })
        .Case<quantum::StateOp>([](auto) { return "state"; })
        .Default([](Operation *o) { return o->getName().getStringRef().str(); });
}

//===----------------------------------------------------------------------===//
// ResourceAnalysis implementation
//===----------------------------------------------------------------------===//

ResourceAnalysis::ResourceAnalysis(Operation *op)
{
    LLVM_DEBUG(dbgs() << "ResourceAnalysis: analyzing operation " << op->getName() << "\n");

    StringRef entryFunc;
    op->walk([&](func::FuncOp funcOp) {
        if (funcOp.isDeclaration())
            return;

        ResourceResult result;
        for (auto &region : funcOp->getRegions()) {
            analyzeRegion(region, result, /*adjointMode=*/false);
        }
        funcResults[funcOp.getName()] = std::move(result);

        if (funcOp->hasAttr("qnode")) {
            entryFunc = funcOp.getName();
        }
    });

    entryFuncName = entryFunc.str();

    if (!entryFunc.empty()) {
        resolveFunctionCalls(entryFunc);
    }
    else {
        for (auto &entry : funcResults) {
            resolveFunctionCalls(entry.getKey());
        }
    }
}

void ResourceAnalysis::analyzeRegion(Region &region, ResourceResult &result, bool adjointMode)
{
    // This implementation is based on the Python implementation in specs_collector.py
    for (Block &block : region) {
        for (Operation &op : block) {
            // quantum.adjoint regions
            if (auto adjOp = dyn_cast<quantum::AdjointOp>(&op)) {
                analyzeRegion(adjOp.getRegion(), result, !adjointMode);
                continue;
            }

            // scf.for loops
            if (auto forOp = dyn_cast<scf::ForOp>(&op)) {
                ResourceResult bodyResult;
                analyzeRegion(forOp.getBodyRegion(), bodyResult, adjointMode);

                // estimated_iterations attribute
                if (auto estAttr = forOp->getAttrOfType<IntegerAttr>("estimated_iterations")) {
                    int64_t iters = estAttr.getValue().getSExtValue();
                    bodyResult.multiplyByScalar(iters);
                }
                else {
                    int64_t iters = countForLoopIterations(forOp);
                    // iters <= 0 means the loop is dynamic
                    if (iters > 0) {
                        bodyResult.multiplyByScalar(iters);
                    }
                }
                result.mergeWith(bodyResult);
                continue;
            }

            // scf.while loops
            if (auto whileOp = dyn_cast<scf::WhileOp>(&op)) {
                ResourceResult bodyResult;
                analyzeRegion(whileOp.getAfter(), bodyResult, adjointMode);

                if (auto estAttr = whileOp->getAttrOfType<IntegerAttr>("estimated_iterations")) {
                    int64_t iters = estAttr.getValue().getSExtValue();
                    bodyResult.multiplyByScalar(iters);
                }

                result.mergeWith(bodyResult);
                continue;
            }

            // scf.if: take max across branches
            if (auto ifOp = dyn_cast<scf::IfOp>(&op)) {
                ResourceResult thenResult;
                analyzeRegion(ifOp.getThenRegion(), thenResult, adjointMode);

                if (!ifOp.getElseRegion().empty()) {
                    ResourceResult elseResult;
                    analyzeRegion(ifOp.getElseRegion(), elseResult, adjointMode);
                    thenResult.mergeWith(elseResult, MergeMethod::Max);
                }
                result.mergeWith(thenResult);
                continue;
            }

            // scf.index_switch: take max across all cases
            if (auto switchOp = dyn_cast<scf::IndexSwitchOp>(&op)) {
                ResourceResult maxResult;
                bool first = true;

                for (auto &caseRegion : switchOp.getCaseRegions()) {
                    ResourceResult caseResult;
                    analyzeRegion(caseRegion, caseResult, adjointMode);
                    if (first) {
                        maxResult = std::move(caseResult);
                        first = false;
                    }
                    else {
                        maxResult.mergeWith(caseResult, MergeMethod::Max);
                    }
                }

                // default region
                ResourceResult defaultResult;
                analyzeRegion(switchOp.getDefaultRegion(), defaultResult, adjointMode);
                if (first) {
                    maxResult = std::move(defaultResult);
                }
                else {
                    maxResult.mergeWith(defaultResult, MergeMethod::Max);
                }

                result.mergeWith(maxResult);
                continue;
            }

            // qec.layer regions
            if (auto layerOp = dyn_cast<qec::LayerOp>(&op)) {
                for (auto &layerRegion : layerOp->getRegions()) {
                    analyzeRegion(layerRegion, result, adjointMode);
                }
                continue;
            }

            collectOperation(&op, result, adjointMode);
        }
    }
}

void ResourceAnalysis::collectOperation(Operation *op, ResourceResult &result, bool adjointMode)
{
    // Quantum gates
    if (isa<quantum::CustomOp, quantum::PauliRotOp, quantum::GlobalPhaseOp, quantum::MultiRZOp,
            quantum::QubitUnitaryOp, quantum::SetStateOp, quantum::SetBasisStateOp>(op)) {
        std::string name = getGateOpName(op, adjointMode);
        int nQubits = getGateQubitCount(op);
        result.operations[name][nQubits] += 1;
        return;
    }

    // Measurements
    if (isa<quantum::MeasureOp, quantum::SampleOp, quantum::CountsOp, quantum::ExpvalOp,
            quantum::VarianceOp, quantum::ProbsOp, quantum::StateOp>(op)) {
        std::string name = getMeasurementName(op);
        result.measurements[name] += 1;
        return;
    }

    // PBC operations
    if (isa<qec::PPRotationOp, qec::PPRotationArbitraryOp, qec::PPMeasurementOp,
            qec::SelectPPMeasurementOp, qec::PrepareStateOp, qec::FabricateOp>(op)) {
        std::string name = getQECOpName(op);
        int nQubits = getQECQubitCount(op);
        result.operations[name][nQubits] += 1;
        return;
    }

    // MBQC operations
    if (isa<mbqc::MeasureInBasisOp, mbqc::GraphStatePrepOp>(op)) {
        std::string name = op->getName().getStringRef().str();
        result.operations[name][0] += 1;
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
        result.numQubits += nqubits;
        return;
    }

    // Metadata: qubit allocation
    if (isa<quantum::AllocQubitOp>(op)) {
        result.numQubits += 1;
        return;
    }

    // Function calls
    if (auto callOp = dyn_cast<func::CallOp>(op)) {
        StringRef callee = callOp.getCallee();
        result.functionCalls[callee] += 1;
        result.unresolvedFunctionCalls[callee] += 1;
        return;
    }

    // Skipped ops
    if (isSkippedOp(op))
        return;

    // Other ops from custom dialects: emit a warning so users are aware
    if (isCustomDialectOp(op)) {
        op->emitWarning() << "ResourceAnalysis encountered an unknown operation '" << op->getName()
                          << "' from a tracked dialect. Some resource data may be missing.";
        return;
    }

    result.classicalInstructions[op->getName().getStringRef()] += 1;
}

void ResourceAnalysis::resolveFunctionCalls(StringRef funcName)
{
    auto it = funcResults.find(funcName);
    if (it == funcResults.end())
        return;

    ResourceResult &resources = it->second;

    // Process all unresolved function calls
    llvm::StringMap<int64_t> toResolve = std::move(resources.unresolvedFunctionCalls);
    resources.unresolvedFunctionCalls.clear();

    for (auto &callEntry : toResolve) {
        StringRef calledFunc = callEntry.getKey();
        int64_t callCount = callEntry.getValue();

        auto calleeIt = funcResults.find(calledFunc);
        if (calleeIt == funcResults.end())
            continue; // External function, cannot resolve

        resolveFunctionCalls(calledFunc);

        // Scale and merge callee resources
        ResourceResult calleeResources = calleeIt->second;
        calleeResources.multiplyByScalar(callCount);
        resources.mergeWith(calleeResources);
    }
}

} // namespace catalyst
