// Copyright 2026 Xanadu Quantum Technologies Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Split non-commuting pass: splits quantum functions that measure non-commuting
// observables into multiple executions, with one group per observable.
//
// For a function like:
//   func.func @circuit() -> (f64, f64, f64) {
//     %ev0 = quantum.expval %obs_q0_X : f64   // group 0
//     %ev1 = quantum.expval %obs_q1_Y : f64   // group 1
//     %ev2 = quantum.expval %obs_q0_Z : f64   // group 2
//     return %ev0, %ev1, %ev2
//   }
//
// With grouping_strategy="" (default), each observable gets its own group:
//   func.func @circuit() -> (f64, f64, f64) {
//     %r0 = call @circuit.group.0()  // ev0
//     %r1 = call @circuit.group.1()  // ev1
//     %r2 = call @circuit.group.2()  // ev2
//     return %r0, %r1, %r2
//   }
//
// With grouping_strategy="wires", observables on non-overlapping wires share a group:
//   func.func @circuit() -> (f64, f64, f64) {
//     %r0:2 = call @circuit.group.0()  // ev0, ev1 (different wires)
//     %r1   = call @circuit.group.1()  // ev2 (overlaps with ev0)
//     return %r0#0, %r0#1, %r1
//   }

#define DEBUG_TYPE "split-non-commuting"

#include <deque>

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

#include "Quantum/IR/QuantumOps.h"
#include "Quantum/IR/QuantumTypes.h"
#include "Quantum/Transforms/Passes.h"

using namespace mlir;
using namespace catalyst;

namespace catalyst {
namespace quantum {

#define GEN_PASS_DEF_SPLITNONCOMMUTINGPASS
#include "Quantum/Transforms/Passes.h.inc"

struct SplitNonCommutingPass : public impl::SplitNonCommutingPassBase<SplitNonCommutingPass> {
    using impl::SplitNonCommutingPassBase<SplitNonCommutingPass>::SplitNonCommutingPassBase;

    /// Check if an operation is a supported measurement operation
    static bool isSupportedMeasOp(Operation *op) { return isa<ExpvalOp>(op); }

    /// Collect the qubit SSA values an observable acts on.
    /// For NamedObsOp: the single qubit operand.
    /// For TensorOp: the union of qubits from all sub-observables.
    static llvm::DenseSet<Value> getObservableQubits(Value obs)
    {
        llvm::DenseSet<Value> qubits;
        Operation *defOp = obs.getDefiningOp();
        if (!defOp) {
            return qubits;
        }

        if (auto namedObs = dyn_cast<NamedObsOp>(defOp)) {
            qubits.insert(namedObs.getQubit());
            return qubits;
        }

        if (auto tensorOp = dyn_cast<TensorOp>(defOp)) {
            for (Value term : tensorOp.getTerms()) {
                auto termQubits = getObservableQubits(term);
                qubits.insert(termQubits.begin(), termQubits.end());
            }
            return qubits;
        }

        return qubits;
    }

    /// Two observables are equivalent if they have the same exact structure.
    static bool observablesEqual(Value lhs, Value rhs)
    {
        if (lhs == rhs) {
            return true;
        }
        Operation *lhsDef = lhs.getDefiningOp();
        Operation *rhsDef = rhs.getDefiningOp();
        if (!lhsDef || !rhsDef) {
            return false;
        }

        return OperationEquivalence::isEquivalentTo(
            lhsDef, rhsDef,
            [](Value l, Value r) -> LogicalResult { return success(observablesEqual(l, r)); },
            nullptr, OperationEquivalence::Flags::IgnoreLocations);
    }

    struct MeasInfo {
        int idx;
        MeasurementProcess measurementOp;
        Value obs;
        llvm::DenseSet<Value> qubits;
    };

    /// Assign each measurement to a group. With the default strategy, each measurement gets its own
    /// group. And with "wires", measurements on non-overlapping wires are packed into the same
    /// group and also handles deduplication to canonicalize identical observables.
    /// This function updates the following maps:
    /// - measInfos:     a list of all measurements and their information.
    /// - measToGroup:   maps a measurement index to a group index.
    /// - canonicalMeas: maps a measurement index to a canonical measurement index.
    static int assignGroups(func::FuncOp funcOp, const std::string &strategy,
                            SmallVector<MeasInfo> &measInfos, llvm::DenseMap<int, int> &measToGroup,
                            llvm::DenseMap<int, int> &canonicalMeas)
    {
        // Collect all measurements and assign them to groups.
        int measIdx = 0;
        funcOp.walk([&](Operation *op) {
            if (!isSupportedMeasOp(op)) {
                return;
            }
            auto measurementOp = cast<MeasurementProcess>(op);
            Value obs = measurementOp.getObs();
            MeasInfo info{.idx = measIdx,
                          .measurementOp = measurementOp,
                          .obs = obs,
                          .qubits = getObservableQubits(obs)};
            measInfos.push_back(info);

            Builder builder(op->getContext());
            op->setAttr("group", builder.getI64IntegerAttr(static_cast<int64_t>(measIdx)));
            measIdx++;
        });

        // Deduplicate and assign groups for measurements.
        SmallVector<int> uniqueIndices;
        SmallVector<llvm::DenseSet<Value>> groupQubits;
        for (int i = 0; i < static_cast<int>(measInfos.size()); ++i) {
            auto currentObs = measInfos[i].obs;
            auto &currentQubits = measInfos[i].qubits;

            // Check if this observable already appeared.
            auto matchIdx = llvm::find_if(uniqueIndices, [&](int j) {
                return observablesEqual(measInfos[j].obs, currentObs);
            });
            if (matchIdx != uniqueIndices.end()) {
                canonicalMeas[i] = *matchIdx;
                measToGroup[i] = measToGroup[*matchIdx];
                continue;
            }

            // For non-duplicate measurements, add this to uniqueIndices.
            uniqueIndices.push_back(i);

            // Assign a group for this measurement.
            if (strategy == "wires") {
                // find a group that doesn't overlap with the current qubits.
                auto it = llvm::find_if(groupQubits, [&](const llvm::DenseSet<Value> &group) {
                    return llvm::none_of(currentQubits, [&](Value q) { return group.count(q); });
                });

                // create a new group if no overlap is found.
                if (it == groupQubits.end()) {
                    it = &groupQubits.emplace_back();
                }
                it->insert(currentQubits.begin(), currentQubits.end());

                // assign the new group index to the measurement.
                measToGroup[i] = static_cast<int>(it - groupQubits.begin());
            }
            else {
                measToGroup[i] = static_cast<int>(uniqueIndices.size()) - 1;
            }
        }

        int numGroups = (strategy == "wires") ? static_cast<int>(groupQubits.size())
                                              : static_cast<int>(uniqueIndices.size());
        return numGroups;
    }

    /// Find the measurement index that a return value traces back to.
    /// Returns -1 if not found.
    static int findMeasIdxForReturnValue(Value returnValue)
    {
        SmallVector<Operation *, 4> worklist;
        llvm::SmallPtrSet<Operation *, 4> visited;

        if (auto *defOp = returnValue.getDefiningOp()) {
            worklist.push_back(defOp);
        }

        while (!worklist.empty()) {
            Operation *op = worklist.pop_back_val();
            if (!op || !visited.insert(op).second) {
                continue;
            }

            if (isSupportedMeasOp(op)) {
                assert(op->hasAttrOfType<IntegerAttr>("group") &&
                       "measurement op must have group attribute");
                return static_cast<int>(op->getAttrOfType<IntegerAttr>("group").getInt());
            }

            for (Value operand : op->getOperands()) {
                if (auto *defOp = operand.getDefiningOp()) {
                    worklist.push_back(defOp);
                }
            }
        }

        return -1;
    }

    /// Analyze which return-value positions belong to each group.
    /// Returns a pair of (groupPositions, returnValueGroupIds):
    /// - groupPositions: group_id -> list of positions in the return operand list
    /// - returnValueGroupIds: list of group_ids for each return value
    ///
    /// Also tracks, for each return value, which measurement index it traces to.
    /// Return values that don't trace back to any measurement are assigned to group 0.
    static std::pair<llvm::DenseMap<int, SmallVector<int>>, SmallVector<int>>
    analyzeGroupReturnPositions(func::FuncOp funcOp, int numGroups,
                                const llvm::DenseMap<int, int> &measToGroup,
                                SmallVector<int> &returnValueMeasIds)
    {
        Operation *returnOp = funcOp.front().getTerminator();

        llvm::DenseMap<int, SmallVector<int>> groupPositions;
        for (int i = 0; i < numGroups; ++i) {
            groupPositions[i] = {};
        }

        SmallVector<int> returnValueGroupIds;
        returnValueGroupIds.reserve(returnOp->getNumOperands());
        returnValueMeasIds.reserve(returnOp->getNumOperands());

        for (auto [position, returnValue] : llvm::enumerate(returnOp->getOperands())) {
            int measIdx = findMeasIdxForReturnValue(returnValue);
            returnValueMeasIds.push_back(measIdx);
            // lookup the group id for the measurement index. Return 0 if not found.
            int groupId = measToGroup.lookup_or(measIdx, 0);
            groupPositions[groupId].push_back(static_cast<int>(position));
            returnValueGroupIds.push_back(groupId);
        }

        return std::make_pair(groupPositions, returnValueGroupIds);
    }

    /// Update the return statement of a function to remove specified values,
    /// and update the function signature accordingly.
    static void updateReturnStatement(func::FuncOp funcOp,
                                      const llvm::DenseSet<Value> &valuesToRemove)
    {
        Operation *returnOp = funcOp.front().getTerminator();
        if (!returnOp) {
            return;
        }

        // Filter out values to remove
        SmallVector<Value> newReturnValues;
        for (Value val : returnOp->getOperands()) {
            if (!valuesToRemove.contains(val)) {
                newReturnValues.push_back(val);
            }
        }

        // Replace return operation with a new one
        IRRewriter rewriter(funcOp.getContext());
        rewriter.setInsertionPoint(returnOp);
        rewriter.replaceOpWithNewOp<func::ReturnOp>(returnOp, newReturnValues);

        // Update function signature
        SmallVector<Type> newOutputTypes;
        for (Value val : newReturnValues) {
            newOutputTypes.push_back(val.getType());
        }

        auto inputTypes = funcOp.getArgumentTypes();
        auto newFuncType = FunctionType::get(funcOp.getContext(), inputTypes, newOutputTypes);
        funcOp.setFunctionType(newFuncType);

        // update res attributes
        SmallVector<DictionaryAttr> newResAttrs(newReturnValues.size(),
                                                DictionaryAttr::get(funcOp.getContext()));
        function_interface_impl::setAllResultAttrDicts(funcOp, newResAttrs);
    }

    /// Walk the def chain upward from the given seed operations and erase dead
    /// operations (those with no remaining uses), stopping at qubit/qreg boundaries.
    static void eraseDeadOps(ArrayRef<Operation *> seedOps)
    {
        std::deque<Operation *> removeOps(seedOps.begin(), seedOps.end());
        llvm::SmallPtrSet<Operation *, 4> visited;
        while (!removeOps.empty()) {
            Operation *op = removeOps.front();
            removeOps.pop_front();

            if (!visited.insert(op).second) {
                continue;
            }

            // Still has live users
            if (llvm::any_of(op->getResults(), [](OpResult r) { return !r.use_empty(); })) {
                continue;
            }

            for (Value operand : op->getOperands()) {
                if (isa<QubitType, QuregType>(operand.getType())) {
                    continue;
                }
                if (auto *defOp = operand.getDefiningOp()) {
                    removeOps.push_back(defOp);
                }
            }

            op->erase();
        }
    }

    /// Remove the given return values from the function, then erase dead def chains.
    static void removeReturnValues(func::FuncOp funcOp, const llvm::DenseSet<Value> &toRemove)
    {
        if (toRemove.empty()) {
            return;
        }
        SmallVector<Operation *> seedOps;
        for (Value val : toRemove) {
            if (Operation *defOp = val.getDefiningOp()) {
                seedOps.push_back(defOp);
            }
        }
        updateReturnStatement(funcOp, toRemove);
        eraseDeadOps(seedOps);
    }

    /// Remove return values that belong to groups other than targetGroup.
    static void removeOtherGroups(func::FuncOp groupFunc, int targetGroup,
                                  ArrayRef<int> returnValueGroupIds)
    {
        Operation *returnOp = groupFunc.front().getTerminator();
        llvm::DenseSet<Value> toRemove;
        for (auto [pos, operand] : llvm::enumerate(returnOp->getOperands())) {
            if (returnValueGroupIds[pos] != targetGroup)
                toRemove.insert(operand);
        }
        removeReturnValues(groupFunc, toRemove);
    }

    /// Remove duplicate measurements within a group function, keeping only the first occurrence of
    /// each canonical observable.
    static void deduplicateMeasurements(func::FuncOp groupFunc,
                                        const llvm::DenseMap<int, int> &canonicalMeas)
    {
        Operation *returnOp = groupFunc.front().getTerminator();
        llvm::DenseSet<Value> toRemove;
        llvm::DenseSet<int> seen;

        for (Value operand : returnOp->getOperands()) {
            int measIdx = findMeasIdxForReturnValue(operand);
            if (measIdx < 0) {
                continue;
            }
            auto canonIt = canonicalMeas.find(measIdx);
            int effectiveIdx = (canonIt != canonicalMeas.end()) ? canonIt->second : measIdx;
            if (!seen.insert(effectiveIdx).second) {
                toRemove.insert(operand);
            }
        }
        removeReturnValues(groupFunc, toRemove);
    }

    /// Distribute device shots among group functions by dividing the original shots by the number
    /// of groups.
    static void distributeShots(func::FuncOp groupFunc, int numGroups)
    {
        // Find the DeviceInitOp in the group function
        DeviceInitOp deviceOp = nullptr;
        for (auto op : groupFunc.getOps<DeviceInitOp>()) {
            deviceOp = op;
            break;
        }

        if (!deviceOp) {
            return;
        }

        Value shots = deviceOp.getShots();
        if (!shots) {
            return;
        }

        OpBuilder builder(deviceOp);
        Location loc = deviceOp.getLoc();
        Value dividedShots;

        // Simplify the shots to a constant if possible
        IntegerAttr intAttr;
        if (matchPattern(shots, m_Constant(&intAttr))) {
            int64_t dividedVal = intAttr.getValue().getSExtValue() / numGroups;
            dividedShots =
                arith::ConstantOp::create(builder, loc, builder.getI64IntegerAttr(dividedVal));
        }
        else {
            Value numGroupsVal = arith::ConstantOp::create(
                builder, loc, builder.getI64IntegerAttr(static_cast<int64_t>(numGroups)));
            dividedShots = arith::DivSIOp::create(builder, loc, shots, numGroupsVal);
        }

        deviceOp.getShotsMutable().assign(dividedShots);
    }

    /// Create a duplicate function for the given group index.
    /// Clones the original function, removes measurements from other groups,
    /// deduplicates measurements within the group, and inserts the new function into the module.
    func::FuncOp createGroupFunction(func::FuncOp funcOp, int groupIdx, int numGroups,
                                     ArrayRef<int> returnValueGroupIds,
                                     const llvm::DenseMap<int, int> &canonicalMeas,
                                     SymbolTable &modSymTable)
    {
        // clone the entire function
        func::FuncOp groupFunc = funcOp.clone();
        std::string groupName = funcOp.getSymName().str() + ".group." + std::to_string(groupIdx);
        groupFunc.setSymName(groupName);
        groupFunc.setPrivate();

        modSymTable.insert(groupFunc);

        // Remove measurements from groups other than groupIdx
        removeOtherGroups(groupFunc, groupIdx, returnValueGroupIds);

        // Remove duplicate measurements within the group
        deduplicateMeasurements(groupFunc, canonicalMeas);

        // Distribute shots among groups
        distributeShots(groupFunc, numGroups);

        return groupFunc;
    }

    /// Replace the original function body with calls to the group functions.
    /// The return values are reassembled in the original order.
    void replaceOriginalWithCalls(func::FuncOp funcOp, ArrayRef<func::FuncOp> groupFunctions,
                                  const llvm::DenseMap<int, SmallVector<int>> &groupReturnPositions,
                                  const llvm::DenseMap<int, int> &canonicalMeas,
                                  ArrayRef<int> returnValueMeasIds)
    {
        Block &originalBlock = funcOp.front();

        // Erase all operations in reverse order
        while (!originalBlock.empty()) {
            Operation &op = originalBlock.back();
            op.dropAllUses();
            op.erase();
        }

        // Build new body: call each group function, then assemble return
        OpBuilder builder(funcOp.getContext());
        builder.setInsertionPointToStart(&originalBlock);
        Location loc = funcOp.getLoc();

        SmallVector<Value> callArgs(originalBlock.getArguments().begin(),
                                    originalBlock.getArguments().end());

        // Call each group function.
        SmallVector<SmallVector<Value>> groupResults;
        for (func::FuncOp groupFunc : groupFunctions) {
            auto callOp = func::CallOp::create(builder, loc, groupFunc, callArgs);
            groupResults.emplace_back(callOp.getResults().begin(), callOp.getResults().end());
        }

        // Map each return position to the correct group call result.
        // Duplicates share a result slot through their canonical measIdx.
        SmallVector<Value> finalReturnValues(returnValueMeasIds.size());
        for (auto &[groupId, positions] : groupReturnPositions) {
            auto &groupVals = groupResults[groupId];
            llvm::DenseMap<int, int> canonToResultIdx;
            int nextResult = 0;

            for (int pos : positions) {
                int measIdx = returnValueMeasIds[pos];
                auto canonIt = canonicalMeas.find(measIdx);
                int effectiveIdx = (canonIt != canonicalMeas.end()) ? canonIt->second : measIdx;

                auto [it, inserted] = canonToResultIdx.try_emplace(effectiveIdx, nextResult);
                if (inserted) {
                    nextResult++;
                }
                finalReturnValues[pos] = groupVals[it->second];
            }
        }

        func::ReturnOp::create(builder, loc, finalReturnValues);
    }

    /// Simplify the identity-expval to a constant 1.0.
    void simplifyIdentityExpval(func::FuncOp funcOp)
    {
        SmallVector<std::pair<ExpvalOp, NamedObsOp>> toSimplify;
        funcOp.walk([&](ExpvalOp expvalOp) {
            auto namedObsOp = expvalOp.getObs().getDefiningOp<NamedObsOp>();
            if (namedObsOp && namedObsOp.getType() == NamedObservable::Identity) {
                toSimplify.emplace_back(expvalOp, namedObsOp);
            }
        });

        for (auto [expvalOp, namedObsOp] : toSimplify) {
            // Identity expval is always 1.0
            OpBuilder builder(expvalOp);
            Value one =
                arith::ConstantOp::create(builder, expvalOp.getLoc(), builder.getF64FloatAttr(1.0));
            expvalOp.replaceAllUsesWith(one);
            expvalOp.erase();
            namedObsOp.erase();
        }
    }

    void runOnOperation() override
    {
        ModuleOp moduleOp = getOperation();

        // Run split-to-single-terms pass first to decompose Hamiltonian expvals
        {
            MLIRContext *ctx = &getContext();
            auto pm = PassManager::on<ModuleOp>(ctx);
            pm.addPass(createSplitToSingleTermsPass());
            pm.addPass(createCanonicalizerPass());
            if (failed(pm.run(moduleOp))) {
                emitError(moduleOp.getLoc()) << "split-to-single-terms pass failed";
                return signalPassFailure();
            }
        }

        // Collect qnode functions to process
        SmallVector<func::FuncOp> funcsToProcess;
        for (auto funcOp : moduleOp.getOps<func::FuncOp>()) {
            if (funcOp->hasAttrOfType<UnitAttr>("quantum.node")) {
                funcsToProcess.push_back(funcOp);
            }
        }

        for (func::FuncOp funcOp : funcsToProcess) {
            // Simplify the identity-expval to a constant 1.0.
            simplifyIdentityExpval(funcOp);

            // Calculate the number of groups and assign groups for measurements
            SmallVector<MeasInfo> measInfos;
            llvm::DenseMap<int, int> measToGroup;
            llvm::DenseMap<int, int> canonicalMeas;

            int numGroups =
                assignGroups(funcOp, groupingStrategy, measInfos, measToGroup, canonicalMeas);

            // Skip if nothing to do. Still proceed with 1 group when duplicates exist so
            // deduplicateMeasurements can remove redundant expvals from the group function.
            if (numGroups == 0 || (numGroups <= 1 && canonicalMeas.empty())) {
                continue;
            }

            // Analyze return value positions for each group
            SmallVector<int> returnValueMeasIds;
            auto [groupReturnPositions, returnValueGroupIds] =
                analyzeGroupReturnPositions(funcOp, numGroups, measToGroup, returnValueMeasIds);

            // Create a duplicate function for each group
            SymbolTable modSymTable(moduleOp);
            SmallVector<func::FuncOp> groupFunctions;
            for (int i = 0; i < numGroups; ++i) {
                groupFunctions.push_back(createGroupFunction(
                    funcOp, i, numGroups, returnValueGroupIds, canonicalMeas, modSymTable));
            }

            // Replace original function body with calls to group functions
            replaceOriginalWithCalls(funcOp, groupFunctions, groupReturnPositions, canonicalMeas,
                                     returnValueMeasIds);
            funcOp->removeAttr("quantum.node");
        }
    }
};

} // namespace quantum
} // namespace catalyst
