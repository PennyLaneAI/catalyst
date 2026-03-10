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
// The pass produces:
//   func.func @circuit() -> (f64, f64, f64) {
//     %r0 = call @circuit.group.0()  // ev0
//     %r1 = call @circuit.group.1()  // ev1
//     %r2 = call @circuit.group.2()  // ev2
//     return %r0, %r1, %r2
//   }

#define DEBUG_TYPE "split-non-commuting"

#include <deque>

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
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

    /// Check if an operation is an observable operation.
    static bool isObservableOp(Operation *op)
    {
        return isa<NamedObsOp, ComputationalBasisOp, HamiltonianOp, TensorOp, HermitianOp>(op);
    }

    /// Calculate the number of groups: one group per observable
    static int calculateNumGroups(func::FuncOp funcOp)
    {
        int groupId = 0;
        funcOp.walk([&](Operation *op) {
            if (isSupportedMeasOp(op)) {
                Builder builder(op->getContext());
                op->setAttr("group", builder.getI64IntegerAttr(static_cast<int64_t>(groupId)));
                groupId++;
            }
        });

        return groupId;
    }

    /// Trace back from a return value through the SSA graph to find which
    /// group's measurement produced it. Returns the group id, or -1 if not found.
    static int findGroupForReturnValue(Value returnValue)
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
    /// Return values that don't trace back to any measurement (e.g. constants produced by
    /// `split-to-single-terms` for Identity observables) are assigned to group 0 so that group 0's
    /// function computes and returns them.
    static std::pair<llvm::DenseMap<int, SmallVector<int>>, SmallVector<int>>
    analyzeGroupReturnPositions(func::FuncOp funcOp, int numGroups)
    {
        Operation *returnOp = funcOp.front().getTerminator();

        llvm::DenseMap<int, SmallVector<int>> groupPositions;
        for (int i = 0; i < numGroups; ++i) {
            groupPositions[i] = {};
        }

        SmallVector<int> returnValueGroupIds;
        returnValueGroupIds.reserve(returnOp->getNumOperands());

        for (auto [position, returnValue] : llvm::enumerate(returnOp->getOperands())) {
            int groupId = findGroupForReturnValue(returnValue);

            // For those return values that don't trace back to any supported measurement, assign
            // them to group 0. For example, constants produced by `split-to-single-terms` for
            // Identity observables are assigned to group 0.
            if (groupId == -1) {
                groupId = 0;
            }
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

    /// Remove measurement operations (and their observable / intermediate chains) that do not
    /// belong to the target group.
    static void removeGroup(func::FuncOp groupFunc, int targetGroup,
                            ArrayRef<int> returnValueGroupIds)
    {
        Operation *returnOp = groupFunc.front().getTerminator();

        // Identify return values that belong to other groups.
        llvm::DenseSet<Value> returnValuesToRemove;
        for (auto [position, operand] : llvm::enumerate(returnOp->getOperands())) {
            int groupId = returnValueGroupIds[position];
            bool keep = (groupId == targetGroup);
            if (!keep) {
                returnValuesToRemove.insert(operand);
            }
        }

        std::deque<Operation *> removeOps;
        for (Value val : returnValuesToRemove) {
            if (Operation *defOp = val.getDefiningOp()) {
                removeOps.push_back(defOp);
            }
        }

        // Update return statement first (drops uses of the removed values)
        updateReturnStatement(groupFunc, returnValuesToRemove);

        // Walk the def chain upward and erase dead operations.
        // But skip observable ops as they are leaf nodes, we erase them but don't recurse into
        // their operands
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

            // Queue operand producers before erasing. And skip for observable ops
            if (!isObservableOp(op)) {
                for (Value operand : op->getOperands()) {
                    if (auto *defOp = operand.getDefiningOp()) {
                        removeOps.push_back(defOp);
                    }
                }
            }

            op->erase();
        }
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
    /// and inserts the new function into the module.
    func::FuncOp createGroupFunction(func::FuncOp funcOp, int groupIdx, int numGroups,
                                     ArrayRef<int> returnValueGroupIds, SymbolTable &modSymTable)
    {
        // clone the entire function
        func::FuncOp groupFunc = funcOp.clone();
        std::string groupName = funcOp.getSymName().str() + ".group." + std::to_string(groupIdx);
        groupFunc.setSymName(groupName);
        groupFunc.setPrivate();

        modSymTable.insert(groupFunc);

        // Remove measurements from groups other than groupIdx
        removeGroup(groupFunc, groupIdx, returnValueGroupIds);

        // Distribute shots among groups
        distributeShots(groupFunc, numGroups);

        return groupFunc;
    }

    /// Replace the original function body with calls to the group functions.
    /// The return values are reassembled in the original order.
    void replaceOriginalWithCalls(func::FuncOp funcOp, ArrayRef<func::FuncOp> groupFunctions,
                                  const llvm::DenseMap<int, SmallVector<int>> &groupReturnPositions)
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

        llvm::DenseMap<int, SmallVector<Value>> groupResults;

        for (int groupId = 0; groupId < static_cast<int>(groupFunctions.size()); ++groupId) {
            func::FuncOp groupFunc = groupFunctions[groupId];
            auto callOp = func::CallOp::create(builder, loc, groupFunc, callArgs);
            groupResults[groupId] =
                SmallVector<Value>(callOp.getResults().begin(), callOp.getResults().end());
        }

        // Calculate total number of return values
        int totalReturns = 0;
        for (auto &entry : groupReturnPositions) {
            totalReturns += static_cast<int>(entry.second.size());
        }

        SmallVector<Value> finalReturnValues(totalReturns);

        for (auto &[groupId, positions] : groupReturnPositions) {
            auto &groupVals = groupResults[groupId];
            assert(static_cast<int>(groupVals.size()) == static_cast<int>(positions.size()) &&
                   "number of group values and positions must match");

            for (int i = 0; i < static_cast<int>(positions.size()); ++i) {
                finalReturnValues[positions[i]] = groupVals[i];
            }
        }

        assert(llvm::all_of(finalReturnValues, [](Value v) { return v != nullptr; }) &&
               "all return values must be filled");

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
        moduleOp.walk([&](func::FuncOp funcOp) {
            if (funcOp->hasAttrOfType<UnitAttr>("quantum.node")) {
                funcsToProcess.push_back(funcOp);
            }
        });

        for (func::FuncOp funcOp : funcsToProcess) {
            // Simplify the identity-expval to a constant 1.0.
            simplifyIdentityExpval(funcOp);

            // Calculate the number of groups (no grouping strategy for now)
            int numGroups = calculateNumGroups(funcOp);

            // If there is only one group, no need to split the execution.
            if (numGroups <= 1) {
                continue;
            }

            // Analyze return value positions for each group
            auto [groupReturnPositions, returnValueGroupIds] =
                analyzeGroupReturnPositions(funcOp, numGroups);

            // Create a duplicate function for each group
            SymbolTable modSymTable(moduleOp);
            SmallVector<func::FuncOp> groupFunctions;
            for (int i = 0; i < numGroups; ++i) {
                groupFunctions.push_back(
                    createGroupFunction(funcOp, i, numGroups, returnValueGroupIds, modSymTable));
            }

            // Replace original function body with calls to group functions
            replaceOriginalWithCalls(funcOp, groupFunctions, groupReturnPositions);
            funcOp->removeAttr("quantum.node");
        }
    }
};

} // namespace quantum
} // namespace catalyst
