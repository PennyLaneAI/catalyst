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

#define DEBUG_TYPE "split-to-single-terms"

#include "llvm/ADT/DenseMap.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Pass/Pass.h"

#include "Quantum/IR/QuantumDialect.h"
#include "Quantum/IR/QuantumOps.h"
#include "stablehlo/dialect/StablehloOps.h"

using namespace mlir;
using namespace catalyst;

namespace catalyst {
namespace quantum {

#define GEN_PASS_DEF_SPLITTOSINGLETERMSPASS
#include "Quantum/Transforms/Passes.h.inc"

struct SplitToSingleTermsPass : public impl::SplitToSingleTermsPassBase<SplitToSingleTermsPass> {
    using impl::SplitToSingleTermsPassBase<SplitToSingleTermsPass>::SplitToSingleTermsPassBase;

    /// Information collected for each Hamiltonian expval that needs to be split
    struct HamiltonianExpvalInfo {
        ExpvalOp expvalOp;
        Value hamiltonianObs;
        SmallVector<Value> leafObservables;
        tensor::FromElementsOp fromElementsWrapper;
    };

    /// Check if an observable is Identity
    bool isIdentityObservable(Value obs)
    {
        if (auto namedObsOp = obs.getDefiningOp<NamedObsOp>()) {
            return namedObsOp.getType() == NamedObservable::Identity;
        }
        return false;
    }

    /// Recursively collect all leaf observables from a potentially nested Hamiltonian
    void collectLeafObservables(Value obs, SmallVectorImpl<Value> &leafObs)
    {
        Operation *defOp = obs.getDefiningOp();

        if (auto hamOp = dyn_cast_or_null<HamiltonianOp>(defOp)) {
            for (Value term : hamOp.getTerms()) {
                collectLeafObservables(term, leafObs);
            }
        }
        else {
            leafObs.push_back(obs);
        }
    }

    /// Recursively collect coefficients from Hamiltonian.
    /// This creates coefficient computation operations in the current insertion point
    void buildCoefficientsExpr(Value obs, Value coeffMultiplier, OpBuilder &builder, Location loc,
                               SmallVectorImpl<Value> &coefficients)
    {
        Operation *defOp = obs.getDefiningOp();

        if (auto hamOp = dyn_cast_or_null<HamiltonianOp>(defOp)) {
            Value coeffs = hamOp.getCoeffs();
            auto termOperands = hamOp.getTerms();

            // coeffs can be either tensor<Nxf64> or memref<Nxf64>
            auto coeffsType = cast<ShapedType>(coeffs.getType());
            int64_t numTerms = coeffsType.getDimSize(0);
            bool isTensor = isa<RankedTensorType>(coeffsType);

            for (int64_t i = 0; i < numTerms; i++) {
                Value idx = arith::ConstantIndexOp::create(builder, loc, i);

                Value coeff;
                if (isTensor) {
                    coeff = tensor::ExtractOp::create(builder, loc, coeffs, ValueRange{idx});
                }
                else {
                    coeff = memref::LoadOp::create(builder, loc, coeffs, ValueRange{idx});
                }

                Value coeffTensor = tensor::FromElementsOp::create(
                    builder, loc, RankedTensorType::get({}, builder.getF64Type()),
                    ValueRange{coeff});

                Value finalCoeff;
                if (coeffMultiplier) {
                    finalCoeff =
                        stablehlo::MulOp::create(builder, loc, coeffMultiplier, coeffTensor);
                }
                else {
                    finalCoeff = coeffTensor;
                }

                buildCoefficientsExpr(termOperands[i], finalCoeff, builder, loc, coefficients);
            }
        }
        else {
            if (!coeffMultiplier) {
                Value one = arith::ConstantOp::create(builder, loc, builder.getF64FloatAttr(1.0));
                coeffMultiplier = tensor::FromElementsOp::create(
                    builder, loc, RankedTensorType::get({}, builder.getF64Type()), ValueRange{one});
            }
            coefficients.push_back(coeffMultiplier);
        }
    }

    /// Create the post-processing computation: weighted sum of expval results
    ///
    /// Given:
    ///   - expvalResults = [<Z x X>, <Y>, 1.0]  (individual expectation values, Identity = 1.0)
    ///   - coefficients  = [c0, c1, c_id]      (extracted from Hamiltonian structure)
    ///
    /// Computes:
    ///   result = c0 * <Z x X> + c1 * <Y> + c_id * 1.0
    ///
    /// Generated MLIR (example with 3 terms including Identity):
    ///   %w0 = stablehlo.multiply %c0, %expval0 : tensor<f64>   // c0 * <Z x X>
    ///   %w1 = stablehlo.multiply %c1, %expval1 : tensor<f64>   // c1 * <Y>
    ///   %w2 = stablehlo.multiply %c_id, %one : tensor<f64>    // c_id * 1.0
    ///   %b0 = stablehlo.broadcast_in_dim %w0 : tensor<f64> -> tensor<1xf64>
    ///   %b1 = stablehlo.broadcast_in_dim %w1 : tensor<f64> -> tensor<1xf64>
    ///   %b2 = stablehlo.broadcast_in_dim %w2 : tensor<f64> -> tensor<1xf64>
    ///   %concat = stablehlo.concatenate %b0, %b1, %b2 : tensor<3xf64>
    ///   %zero = stablehlo.constant 0.0 : tensor<f64>
    ///   %result = stablehlo.reduce(%concat init: %zero) applies stablehlo.add
    ///
    Value createPostProcessing(OpBuilder &builder, Location loc, ValueRange expvalResults,
                               ValueRange coefficients)
    {
        assert(expvalResults.size() == coefficients.size());
        assert(expvalResults.size() > 0 && "Hamiltonian must have at least one observable");

        SmallVector<Value> weightedExpvals;
        for (size_t i = 0; i < expvalResults.size(); i++) {
            Value weighted =
                stablehlo::MulOp::create(builder, loc, coefficients[i], expvalResults[i]);
            weightedExpvals.push_back(weighted);
        }

        // Single value: return the weighted expval
        if (weightedExpvals.size() == 1) {
            return weightedExpvals[0];
        }

        // Broadcast, concatenate, reduce
        auto tensor1xf64Type = RankedTensorType::get({1}, builder.getF64Type());
        auto tensorNxf64Type = RankedTensorType::get({static_cast<int64_t>(weightedExpvals.size())},
                                                     builder.getF64Type());

        SmallVector<Value> broadcastedValues;
        for (Value v : weightedExpvals) {
            Value broadcasted = stablehlo::BroadcastInDimOp::create(
                builder, loc, tensor1xf64Type, v, builder.getDenseI64ArrayAttr({}));
            broadcastedValues.push_back(broadcasted);
        }

        Value concatenated = stablehlo::ConcatenateOp::create(builder, loc, tensorNxf64Type,
                                                              broadcastedValues, /*dimension=*/0);

        auto zeroAttr = DenseElementsAttr::get(RankedTensorType::get({}, builder.getF64Type()),
                                               builder.getF64FloatAttr(0.0).getValue());
        Value zero = stablehlo::ConstantOp::create(builder, loc, zeroAttr);

        auto reduceOp = stablehlo::ReduceOp::create(
            builder, loc, TypeRange{RankedTensorType::get({}, builder.getF64Type())},
            ValueRange{concatenated}, ValueRange{zero}, builder.getDenseI64ArrayAttr({0}));

        {
            OpBuilder::InsertionGuard guard(builder);
            Block *reduceBody = builder.createBlock(&reduceOp.getBody());
            auto scalarF64Type = RankedTensorType::get({}, builder.getF64Type());
            reduceBody->addArgument(scalarF64Type, loc);
            reduceBody->addArgument(scalarF64Type, loc);

            builder.setInsertionPointToStart(reduceBody);
            Value addResult = stablehlo::AddOp::create(builder, loc, reduceBody->getArgument(0),
                                                       reduceBody->getArgument(1));
            stablehlo::ReturnOp::create(builder, loc, ValueRange{addResult});
        }

        return reduceOp.getResult(0);
    }

    /// Remove dead operations before a given operation in a block
    /// Iteratively removes ops with no users until no more can be removed
    void removeDeadOpsBeforeOp(func::FuncOp func, Operation *boundaryOp,
                               bool reserveDeviceOps = false)
    {
        bool changed = true;
        while (changed) {
            changed = false;
            SmallVector<Operation *> deadOps;

            Block &block = func.getBody().front();
            for (auto it = block.rbegin(); it != block.rend(); ++it) {
                Operation *op = &(*it);

                if (reserveDeviceOps && isa<DeviceInitOp, DeviceReleaseOp, DeallocOp>(op)) {
                    continue;
                }

                if (op->hasTrait<OpTrait::IsTerminator>() || op == boundaryOp ||
                    !op->isBeforeInBlock(boundaryOp)) {
                    continue;
                }

                // Check if all results are unused
                bool allResultsUnused =
                    llvm::all_of(op->getResults(), [](Value result) { return result.use_empty(); });

                if (allResultsUnused) {
                    deadOps.push_back(op);
                }
            }

            for (Operation *op : deadOps) {
                op->erase();
                changed = true;
            }
        }
    }

    /// Information about how return values are mapped after transformation
    struct ReturnValueMappingInfo {
        SmallVector<std::pair<bool, size_t>> mapping; // (isHamiltonian, numValues)
        SmallVector<Type> newReturnTypes;
    };

    /// Information about which arguments were removed from the quantum function
    struct ArgumentRemovalInfo {
        SmallVector<unsigned> removedArgIndices; // Sorted indices of removed arguments
    };

    /// Modify the quantum function to return individual expvals instead of Hamiltonian expvals
    ///
    /// Before: quantumFunc returns (<H>, <Z>) where <H> is Hamiltonian expval
    ///         where <H> = c0 * <Z x X> + c1 * <Y>
    /// After:  quantumFunc returns (<Z x X>, <Y>, <Z>) individual leaf expvals
    ///
    /// Returns the mapping info needed for the entry function transformation
    LogicalResult rewriteQuantumFunc(func::FuncOp quantumFunc, Location loc,
                                     ReturnValueMappingInfo &mappingInfo)
    {
        // Collect hamiltonian-expval pairs
        SmallVector<std::pair<ExpvalOp, Value>> hamiltonianExpvalPairs;
        quantumFunc.walk([&](ExpvalOp expvalOp) {
            Value obs = expvalOp.getObs();
            if (isa_and_nonnull<HamiltonianOp>(obs.getDefiningOp())) {
                hamiltonianExpvalPairs.push_back(std::make_pair(expvalOp, obs));
            }
        });

        // Track which return values need to be replaced
        DenseMap<Value, SmallVector<Value>> replacementMap;

        for (auto &[expvalOp, obs] : hamiltonianExpvalPairs) {
            OpBuilder builder(expvalOp);

            // Collect leaf observables
            SmallVector<Value> leafObs;
            collectLeafObservables(obs, leafObs);

            // Create individual expvals with from_elements wrappers
            // For Identity observables, use constant 1.0 instead of computing expval
            SmallVector<Value> newExpvalTensors;
            for (Value leaf : leafObs) {
                Value tensor;
                if (isIdentityObservable(leaf)) {
                    // Identity expval is always 1.0
                    Value one =
                        arith::ConstantOp::create(builder, loc, builder.getF64FloatAttr(1.0));
                    tensor = tensor::FromElementsOp::create(
                        builder, loc, RankedTensorType::get({}, builder.getF64Type()),
                        ValueRange{one});
                }
                else {
                    Value expval = ExpvalOp::create(builder, loc, builder.getF64Type(), leaf);
                    tensor = tensor::FromElementsOp::create(
                        builder, loc, RankedTensorType::get({}, builder.getF64Type()),
                        ValueRange{expval});
                }
                newExpvalTensors.push_back(tensor);
            }

            // expval should have exactly one user
            if (!expvalOp.getResult().hasOneUse()) {
                expvalOp.emitError() << "expval result is expected to have exactly one user";
                return failure();
            }

            // Replace the expval result with the new expval tensors
            Operation *user = *expvalOp.getResult().getUsers().begin();
            if (auto fromElementsOp = dyn_cast<tensor::FromElementsOp>(user)) {
                replacementMap[fromElementsOp.getResult()] = newExpvalTensors;
            }
            else {
                replacementMap[expvalOp.getResult()] = newExpvalTensors;
            }
        }

        // Update quantum function's return
        auto quantumReturnOp =
            dyn_cast<func::ReturnOp>(quantumFunc.getBody().front().getTerminator());
        if (!quantumReturnOp) {
            emitError(loc) << "quantum function does not have a return op";
            return failure();
        }

        SmallVector<Value> newReturnValues;

        // Build return value mapping
        for (Value oldRetVal : quantumReturnOp.getOperands()) {
            if (replacementMap.count(oldRetVal)) {
                size_t numNewVals = replacementMap[oldRetVal].size();
                mappingInfo.mapping.push_back({true, numNewVals});
                for (Value newVal : replacementMap[oldRetVal]) {
                    newReturnValues.push_back(newVal);
                    mappingInfo.newReturnTypes.push_back(newVal.getType());
                }
            }
            else {
                mappingInfo.mapping.push_back({false, 1});
                newReturnValues.push_back(oldRetVal);
                mappingInfo.newReturnTypes.push_back(oldRetVal.getType());
            }
        }

        OpBuilder returnBuilder(quantumReturnOp);
        func::ReturnOp::create(returnBuilder, quantumReturnOp.getLoc(), newReturnValues);
        quantumReturnOp.erase();

        auto quantumFuncType =
            FunctionType::get(quantumFunc.getContext(), quantumFunc.getFunctionType().getInputs(),
                              mappingInfo.newReturnTypes);
        quantumFunc.setFunctionType(quantumFuncType);

        // Update res_attrs to match the new number of return values
        unsigned numResults = quantumFuncType.getNumResults();
        SmallVector<DictionaryAttr> newResAttrs(numResults,
                                                DictionaryAttr::get(quantumFunc.getContext()));
        function_interface_impl::setAllResultAttrDicts(quantumFunc, newResAttrs);

        // Find the last quantum dealloc
        Operation *lastDeallocOp = nullptr;
        quantumFunc.walk(
            [&](quantum::DeallocOp deallocOp) { lastDeallocOp = deallocOp.getOperation(); });
        if (lastDeallocOp) {
            removeDeadOpsBeforeOp(quantumFunc, lastDeallocOp, /*reserveDeviceOps=*/true);
        }

        return success();
    }

    /// Remove unused arguments from the quantum function
    /// And update information in the removalInfo to indicate which arguments were removed
    void removeUnusedArguments(func::FuncOp quantumFunc, ArgumentRemovalInfo &removalInfo)
    {
        Block &entryBlock = quantumFunc.getBody().front();
        SmallVector<unsigned> argsToRemove;

        // Find unused arguments
        for (auto [idx, arg] : llvm::enumerate(entryBlock.getArguments())) {
            if (arg.use_empty()) {
                argsToRemove.push_back(idx);
            }
        }

        // Remove arguments, we do it in the reverse order to maintain correct indices
        for (unsigned idx : llvm::reverse(argsToRemove)) {
            entryBlock.eraseArgument(idx);
            removalInfo.removedArgIndices.push_back(idx);
        }

        llvm::sort(removalInfo.removedArgIndices);

        // Update function signature
        if (!argsToRemove.empty()) {
            SmallVector<Type> newInputTypes;
            for (auto [idx, arg] : llvm::enumerate(entryBlock.getArguments())) {
                newInputTypes.push_back(arg.getType());
            }
            auto newFuncType = FunctionType::get(quantumFunc.getContext(), newInputTypes,
                                                 quantumFunc.getFunctionType().getResults());
            quantumFunc.setFunctionType(newFuncType);
        }
    }

    /// Rewrite the entry function to call quantumFunc and do post-processing
    ///
    /// Before: origFunc contains quantum ops and returns (<H>, <Z>)
    /// After:  origFunc calls quantumFunc, computes weighted sum, returns (<H>, <Z>)
    LogicalResult rewriteEntryFunc(func::FuncOp origFunc, func::FuncOp quantumFunc, Location loc,
                                   const ReturnValueMappingInfo &mappingInfo,
                                   const ArgumentRemovalInfo &removalInfo)
    {
        // Find Hamiltonian expvals in original function (for coefficient extraction)
        SmallVector<std::pair<ExpvalOp, Value>> hamiltonianExpvalPairs;
        origFunc.walk([&](ExpvalOp expvalOp) {
            Value obs = expvalOp.getObs();
            if (isa_and_nonnull<HamiltonianOp>(obs.getDefiningOp())) {
                hamiltonianExpvalPairs.push_back(std::make_pair(expvalOp, obs));
            }
        });

        auto origReturnOp = dyn_cast<func::ReturnOp>(origFunc.getBody().front().getTerminator());
        if (!origReturnOp) {
            emitError(loc) << "original function does not have a return op";
            return failure();
        }

        // Insert call to quantumFunc before the return
        OpBuilder builder(origReturnOp);
        SmallVector<Value> callArgs;
        Block &origBlock = origFunc.getBody().front();

        // Only pass arguments that weren't removed from quantumFunc
        for (auto [idx, arg] : llvm::enumerate(origBlock.getArguments())) {
            if (!llvm::is_contained(removalInfo.removedArgIndices, idx)) {
                callArgs.push_back(arg);
            }
        }

        auto callOp = func::CallOp::create(builder, loc, quantumFunc.getName(),
                                           mappingInfo.newReturnTypes, callArgs);

        // Post-processing:
        // For Hamiltonian results: collect coefficients and compute weighted sum
        // For non-Hamiltonian results: pass through from call results
        SmallVector<Value> finalResults;
        size_t callResultIdx = 0;
        size_t hamIdx = 0;

        for (auto [isHamiltonian, numValues] : mappingInfo.mapping) {
            if (isHamiltonian) {
                // Get expval results from call for this Hamiltonian
                SmallVector<Value> expvalResults;
                for (size_t i = 0; i < numValues; i++) {
                    expvalResults.push_back(callOp.getResult(callResultIdx + i));
                }
                callResultIdx += numValues;

                // Build coefficients expression from original function's Hamiltonian structure
                Value obs = hamiltonianExpvalPairs[hamIdx].second;
                SmallVector<Value> coefficients;
                buildCoefficientsExpr(obs, /*coeffMultiplier=*/nullptr, builder, loc, coefficients);

                Value result = createPostProcessing(builder, loc, expvalResults, coefficients);
                finalResults.push_back(result);
                hamIdx++;
            }
            else {
                // Non-Hamiltonian result: pass through from call
                finalResults.push_back(callOp.getResult(callResultIdx));
                callResultIdx++;
            }
        }

        func::ReturnOp::create(builder, origReturnOp.getLoc(), finalResults);
        origReturnOp.erase();

        // Clean up dead ops before the call
        removeDeadOpsBeforeOp(origFunc, callOp.getOperation(), /*reserveDeviceOps=*/false);

        // Remove qnode attribute
        origFunc->removeAttr("qnode");

        return success();
    }

    void runOnOperation() override
    {
        Operation *moduleOp = getOperation();

        // Find all qnode functions with Hamiltonian expvals
        SmallVector<func::FuncOp> funcsToProcess;
        moduleOp->walk([&](func::FuncOp funcOp) {
            if (!funcOp->hasAttr("qnode")) {
                return;
            }

            bool hasHamiltonian = false;
            funcOp.walk([&](ExpvalOp expvalOp) {
                if (isa_and_nonnull<HamiltonianOp>(expvalOp.getObs().getDefiningOp())) {
                    hasHamiltonian = true;
                    return WalkResult::interrupt();
                }
                return WalkResult::advance();
            });

            if (hasHamiltonian) {
                funcsToProcess.push_back(funcOp);
            }
        });

        for (func::FuncOp origFunc : funcsToProcess) {
            Location loc = origFunc.getLoc();
            OpBuilder moduleBuilder(origFunc);

            // Clone origFunc -> origFunc.quantum (<circuit_name>.quantum)
            // origFunc.quantum will contain the quantum operations and return individual expvals
            // origFunc will be the entry point that calls quantumFunc and does post-processing
            IRMapping cloneMapping;
            std::string quantumFuncName = origFunc.getName().str() + ".quantum";
            auto quantumFunc = cast<func::FuncOp>(moduleBuilder.clone(*origFunc, cloneMapping));
            quantumFunc.setName(quantumFuncName);

            // Modify quantumFunc to return individual expvals
            ReturnValueMappingInfo mappingInfo;
            if (failed(rewriteQuantumFunc(quantumFunc, loc, mappingInfo))) {
                return signalPassFailure();
            }

            // Remove unused arguments from quantumFunc
            ArgumentRemovalInfo removalInfo;
            removeUnusedArguments(quantumFunc, removalInfo);

            // Modify origFunc to call quantumFunc and do post-processing
            if (failed(rewriteEntryFunc(origFunc, quantumFunc, loc, mappingInfo, removalInfo))) {
                return signalPassFailure();
            }
        }
    }
};

} // namespace quantum
} // namespace catalyst
