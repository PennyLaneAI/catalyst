// Copyright 2023 Xanadu Quantum Technologies Inc.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#define DEBUG_TYPE "adjoint"

#include <algorithm>
#include <queue>
#include <string>
#include <vector>

#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Complex/IR/Complex.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Index/IR/IndexOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Transforms/DialectConversion.h"

#include "Catalyst/IR/CatalystOps.h"
#include "PBC/IR/PBCOps.h"
#include "Quantum/IR/QuantumInterfaces.h"
#include "Quantum/IR/QuantumOps.h"
#include "Quantum/IR/QuantumTypes.h"

#include "AdjointLowering.hpp"
#include "QuantumCache.hpp"

using namespace mlir;
using namespace catalyst;
using namespace catalyst::quantum;

namespace {

/// Trace the value-semantic register use-def chain backwards to the `quantum.alloc` operation that
/// originally produced it, following straight-line `quantum.insert` operations. Returns a null op
/// if the allocation cannot be determined statically, i.e. the register crosses the block boundary.
quantum::AllocOp findSourceAllocOp(Value qreg)
{
    while (Operation *defOp = qreg.getDefiningOp()) {
        if (auto allocOp = dyn_cast<quantum::AllocOp>(defOp)) {
            return allocOp;
        }
        if (auto insertOp = dyn_cast<quantum::InsertOp>(defOp)) {
            qreg = insertOp.getInQreg();
            continue;
        }
        break;
    }
    return quantum::AllocOp();
}

/// Clone the region of the adjoint operation `op` to the insertion point specified by the
/// `builder`. Build and return the value mapping `mapping`.
void cloneAdjointRegion(AdjointOp op, OpBuilder &builder, IRMapping &mapping,
                        SmallVector<Value> &reversedResults)
{
    Block &block = op.getRegion().front();
    for (Operation &op : block.without_terminator()) {
        builder.clone(op, mapping);
    }
    auto yieldOp = cast<quantum::YieldOp>(block.getTerminator());
    for (Value yieldVal : yieldOp->getOperands()) {
        reversedResults.push_back(mapping.lookupOrDefault(yieldVal));
    }
}

/// A class that generates the quantum "backwards pass" of the adjoint operation using the stored
/// gate parameters and cached control flow.
class AdjointGenerator {
  public:
    AdjointGenerator(IRMapping &remappedValues, QuantumCache &cache)
        : remappedValues(remappedValues), cache(cache)
    {
    }

    /// Recursively generate the adjoint version of `region` with reversed control flow and adjoint
    /// quantum gates.
    LogicalResult generate(Region &region, OpBuilder &builder)
    {
        generateImpl(region, builder);
        return failure(generationFailed);
    }

  private:
    void generateImpl(Region &region, OpBuilder &builder)
    {
        assert(region.hasOneBlock() &&
               "Expected only structured control flow (each region should have a single block)");

        for (Operation &op : llvm::reverse(region.front().without_terminator())) {
            if (auto callOp = dyn_cast<func::CallOp>(op)) {
                visitOperation(callOp, builder);
            }
            else if (auto forOp = dyn_cast<scf::ForOp>(op)) {
                visitOperation(forOp, builder);
            }
            else if (auto ifOp = dyn_cast<scf::IfOp>(op)) {
                visitOperation(ifOp, builder);
            }
            else if (auto whileOp = dyn_cast<scf::WhileOp>(op)) {
                visitOperation(whileOp, builder);
            }
            else if (auto switchOp = dyn_cast<scf::IndexSwitchOp>(op)) {
                visitOperation(switchOp, builder);
            }
            else if (auto insertOp = dyn_cast<quantum::InsertOp>(op)) {
                Value dynamicWire = getDynamicWire(insertOp, builder);
                auto extractOp = quantum::ExtractOp::create(
                    builder, insertOp.getLoc(), insertOp.getQubit().getType(),
                    remappedValues.lookup(insertOp.getOutQreg()), dynamicWire,
                    insertOp.getIdxAttrAttr());
                remappedValues.map(insertOp.getQubit(), extractOp.getResult());
                remappedValues.map(insertOp.getInQreg(),
                                   remappedValues.lookup(insertOp.getOutQreg()));
            }
            else if (auto extractOp = dyn_cast<quantum::ExtractOp>(op)) {
                Value dynamicWire = getDynamicWire(extractOp, builder);
                auto insertOp = quantum::InsertOp::create(
                    builder, extractOp.getLoc(), extractOp.getQreg().getType(),
                    remappedValues.lookup(extractOp.getQreg()), dynamicWire,
                    extractOp.getIdxAttrAttr(), remappedValues.lookup(extractOp.getQubit()));
                remappedValues.map(extractOp.getQreg(), insertOp.getResult());
            }
            else if (auto allocOp = dyn_cast<quantum::AllocOp>(op)) {
                quantum::DeallocOp::create(builder, allocOp.getLoc(),
                                           remappedValues.lookup(allocOp.getQreg()));
            }
            else if (auto deallocOp = dyn_cast<quantum::DeallocOp>(op)) {
                quantum::AllocOp sourceAlloc = findSourceAllocOp(deallocOp.getQreg());
                if (!sourceAlloc) {
                    deallocOp.emitError("Unable to reverse dynamic register deallocation in the "
                                        "adjoint region: allocation size could not be determined");
                    generationFailed = true;
                    return;
                }
                Value nqubits = sourceAlloc.getNqubits();
                if (nqubits) {
                    nqubits = remappedValues.lookupOrDefault(nqubits);
                }
                auto newAlloc = quantum::AllocOp::create(builder, deallocOp.getLoc(),
                                                         deallocOp.getQreg().getType(), nqubits,
                                                         sourceAlloc.getNqubitsAttrAttr());
                remappedValues.map(deallocOp.getQreg(), newAlloc.getQreg());
            }
            else if (auto allocQubitOp = dyn_cast<quantum::AllocQubitOp>(op)) {
                quantum::DeallocQubitOp::create(builder, allocQubitOp.getLoc(),
                                                remappedValues.lookup(allocQubitOp.getQubit()));
            }
            else if (auto deallocQubitOp = dyn_cast<quantum::DeallocQubitOp>(op)) {
                auto newAlloc = quantum::AllocQubitOp::create(builder, deallocQubitOp.getLoc());
                remappedValues.map(deallocQubitOp.getQubit(), newAlloc.getQubit());
            }
            else if (auto gate = dyn_cast<quantum::QuantumGate>(op)) {
                visitOperation(gate, builder);
            }
            else if (auto ppr = dyn_cast<pbc::PPRotationOp>(op)) {
                visitOperation(ppr, builder);
            }
            else if (auto adjointOp = dyn_cast<quantum::AdjointOp>(&op)) {
                for (auto [regionArg, result] : llvm::zip_equal(
                         adjointOp.getRegion().getArguments(), adjointOp.getResults())) {
                    remappedValues.map(regionArg, remappedValues.lookup(result));
                }
                SmallVector<Value> reversedResults;
                cloneAdjointRegion(adjointOp, builder, remappedValues, reversedResults);
                for (auto [operand, reversedResult] :
                     llvm::zip_equal(adjointOp.getArgs(), reversedResults)) {
                    remappedValues.map(operand, reversedResult);
                }
            }
            else if (isa<QuantumDialect>(op.getDialect())) {
                op.emitError("Unhandled operation in adjoint region");
                generationFailed = true;
                return;
            }
        }
    }

    template <typename IndexingOp> Value getDynamicWire(IndexingOp op, OpBuilder &builder)
    {
        Value dynamicWire;
        if (!op.getIdxAttr().has_value()) {
            dynamicWire = ListPopOp::create(builder, op.getLoc(), cache.wireVector);
        }
        return dynamicWire;
    }

    SmallVector<Value> getQuantumValues(ValueRange values)
    {
        SmallVector<Value> qvalues;
        for (Value value : values) {
            if (isa<quantum::QuregType, quantum::QubitType>(value.getType())) {
                qvalues.push_back(value);
            }
        }
        return qvalues;
    }

    void visitOperation(quantum::QuantumGate gate, OpBuilder &builder)
    {
        for (const auto &[qubitResult, qubitOperand] :
             llvm::zip(gate.getQubitResults(), gate.getQubitOperands())) {
            remappedValues.map(qubitOperand, remappedValues.lookup(qubitResult));
        }

        auto clone = cast<QuantumGate>(builder.clone(*gate, remappedValues));
        clone.setAdjointFlag(!gate.getAdjointFlag());

        // Read cached parameters from the recorded parameter vector.
        Operation *operation = gate;
        if (auto parametrizedGate = dyn_cast<quantum::ParametrizedGate>(operation)) {
            OpBuilder::InsertionGuard insertionGuard(builder);
            builder.setInsertionPoint(clone);
            ValueRange params = parametrizedGate.getAllParams();
            size_t numParams = params.size();
            SmallVector<Value> cachedParams(numParams);
            size_t idx = 0;
            // popping gives the parameters in reverse
            for (Value param : llvm::reverse(params)) {
                Type paramType = param.getType();
                verifyTypeIsCacheable(paramType, operation);
                if (paramType.isF64()) {
                    cachedParams[numParams - 1 - idx] =
                        ListPopOp::create(builder, parametrizedGate.getLoc(), cache.paramVector);
                    idx++;
                    continue;
                }

                // Guaranteed by verifyTypeIsPoppable above.
                auto aTensorType = cast<RankedTensorType>(paramType);
                ArrayRef<int64_t> shape = aTensorType.getShape();
                Type elementType = aTensorType.getElementType();
                // Constants
                auto loc = parametrizedGate.getLoc();
                Value c0 = index::ConstantOp::create(builder, loc, 0);
                Value c1 = index::ConstantOp::create(builder, loc, 1);
                // TODO: Generalize to all possible dimensions
                bool isDim0Static = ShapedType::kDynamic != shape[0];
                bool isDim1Static = ShapedType::kDynamic != shape[1];
                Value dim0Length = isDim0Static
                                       ? (Value)index::ConstantOp::create(builder, loc, shape[0])
                                       : (Value)tensor::DimOp::create(builder, loc, param, c0);
                Value dim1Length = isDim1Static
                                       ? (Value)index::ConstantOp::create(builder, loc, shape[1])
                                       : (Value)tensor::DimOp::create(builder, loc, param, c1);

                // Renaming for legibility
                // Note: Since this is a square matrix, upperBound for both loops is the
                // same value.
                Value lowerBoundDim0 = c0;
                Value upperBoundDim0 = dim0Length;
                Value stepDim0 = c1;
                Value lowerBoundDim1 = c0;
                Value upperBoundDim1 = dim1Length;
                Value stepDim1 = c1;
                Value beginningTensor = tensor::EmptyOp::create(builder, loc, shape, elementType);
                // This time, we are in reverse, so we need to start
                // with N-1 since MLIR does not allow for loops with negative step sizes.
                SmallVector<Value> initialValues = {beginningTensor};

                scf::ForOp iForLoop = scf::ForOp::create(builder, loc, lowerBoundDim0,
                                                         upperBoundDim0, stepDim0, initialValues);
                {
                    OpBuilder::InsertionGuard afterIForLoop(builder);
                    builder.setInsertionPointToStart(iForLoop.getBody());
                    auto iIterArgs = iForLoop.getRegionIterArgs();
                    Value currIthTensor = iIterArgs.front();

                    Value i = iForLoop.getInductionVar();
                    Value iPlusOne = index::AddOp::create(builder, loc, i, c1);
                    Value nMinusIMinusOne =
                        index::SubOp::create(builder, loc, dim0Length, iPlusOne);
                    // Just for legibility
                    Value iTensorIndex = nMinusIMinusOne;

                    scf::ForOp jForLoop = scf::ForOp::create(
                        builder, loc, lowerBoundDim1, upperBoundDim1, stepDim1, currIthTensor);
                    {
                        OpBuilder::InsertionGuard afterJForLoop(builder);
                        builder.setInsertionPointToStart(jForLoop.getBody());
                        auto jIterArgs = jForLoop.getRegionIterArgs();
                        assert(jIterArgs.size() == 1 &&
                               "jForLoop has more induction variables than necessary.");
                        Value currIthJthTensor = jIterArgs.front();

                        Value imag = ListPopOp::create(builder, loc, cache.paramVector);
                        Value real = ListPopOp::create(builder, loc, cache.paramVector);
                        Value element =
                            complex::CreateOp::create(builder, loc, elementType, real, imag);

                        // TODO: Generalize to types which are not complex
                        Value j = jForLoop.getInductionVar();
                        Value jPlusOne = index::AddOp::create(builder, loc, j, c1);
                        Value nMinusJMinusOne =
                            index::SubOp::create(builder, loc, dim1Length, jPlusOne);
                        // Just for legibility
                        Value jTensorIndex = nMinusJMinusOne;
                        SmallVector<Value> indices = {iTensorIndex, jTensorIndex};

                        Value updatedIthJthTensor = tensor::InsertOp::create(
                            builder, loc, element, currIthJthTensor, indices);
                        scf::YieldOp::create(builder, loc, updatedIthJthTensor);
                    }

                    Value ithTensor = jForLoop.getResult(0);
                    scf::YieldOp::create(builder, loc, ithTensor);
                }

                Value recreatedTensor = iForLoop.getResult(0);
                cachedParams[numParams - 1 - idx] = recreatedTensor;
                idx++;
            }
            MutableOperandRange(clone, parametrizedGate.getParamOperandIdx(), params.size())
                .assign(cachedParams);
        }

        for (const auto &[qubitResult, qubitOperand] :
             llvm::zip(clone.getQubitResults(), gate.getQubitOperands())) {
            remappedValues.map(qubitOperand, qubitResult);
        }
    }

    void visitOperation(pbc::PPRotationOp ppr, OpBuilder &builder)
    {
        for (const auto &[qubitResult, qubitOperand] :
             llvm::zip(ppr.getOutQubits(), ppr.getInQubits())) {
            remappedValues.map(qubitOperand, remappedValues.lookup(qubitResult));
        }

        auto clone = cast<pbc::PPRotationOp>(builder.clone(*ppr, remappedValues));
        clone.setRotationKind(-ppr.getRotationKind());

        for (const auto &[qubitResult, qubitOperand] :
             llvm::zip(clone.getOutQubits(), ppr.getInQubits())) {
            remappedValues.map(qubitOperand, qubitResult);
        }
    }

    void getAdjointCallOpArgs(func::CallOp callOp, std::vector<Value> &args)
    {
        // Get the reversed results
        args = {callOp.getArgOperands().begin(), callOp.getArgOperands().end()};
        std::queue<Value> reversedResults;
        for (Value callResult : callOp.getResults()) {
            if (!isa<QuregType, QubitType>(callResult.getType())) {
                continue;
            }
            reversedResults.push(remappedValues.lookup(callResult));
        }
        for (size_t i = 0; i < args.size(); i++) {
            if (!isa<QuregType, QubitType>(args[i].getType())) {
                if (!isa<BlockArgument>(args[i]) &&
                    args[i].getDefiningOp()->getParentRegion() == callOp->getParentRegion()) {
                    // Encountered a classical call operand from within the adjoint region
                    // Must use the clone outside
                    args[i] = remappedValues.lookup(args[i]);
                }
            }
            else {
                args[i] = reversedResults.front();
                reversedResults.pop();
            }
        }
    }
    void visitOperation(func::CallOp callOp, OpBuilder &builder)
    {
        // Get the the original function
        SymbolRefAttr symbol = dyn_cast_if_present<SymbolRefAttr>(callOp.getCallableForCallee());
        func::FuncOp funcOp =
            dyn_cast_or_null<func::FuncOp>(SymbolTable::lookupNearestSymbolFrom(callOp, symbol));
        assert(funcOp != nullptr && "The funcOp is null and therefore not supported.");

        auto resultTypes = funcOp.getResultTypes();

        bool quantum = std::any_of(resultTypes.begin(), resultTypes.end(), [](const auto &value) {
            return isa<QuregType, QubitType>(value);
        });
        if (!quantum) {
            // This operation is purely classical
            return;
        }

        // Save the insertion point to come back to it after creating the adjoint of the function
        auto insertionSaved = builder.saveInsertionPoint();
        MLIRContext *ctx = builder.getContext();

        // Create the adjoint of the original function at the module level
        ModuleOp moduleOp = funcOp->getParentOfType<ModuleOp>();
        builder.setInsertionPointToStart(moduleOp.getBody());

        Block *originalBlock = &funcOp.front();
        Operation *originalTerminator = originalBlock->getTerminator();
        ValueRange originalArguments = originalBlock->getArguments();

        // Get the arguments and outputs types from the original function (same signature)
        FunctionType adjointFnType =
            FunctionType::get(ctx, /*inputs=*/
                              originalArguments.getTypes(),
                              /*outputs=*/originalTerminator->getOperandTypes());
        std::string adjointName = funcOp.getName().str() + ".adjoint";
        Location loc = funcOp.getLoc();
        func::FuncOp adjointFnOp = func::FuncOp::create(builder, loc, adjointName, adjointFnType);
        adjointFnOp.setPrivate();

        // Create the block of the adjoint function
        Block *adjointFnBlock = adjointFnOp.addEntryBlock();
        builder.setInsertionPointToStart(adjointFnBlock);

        SmallVector<Value> quantumArgs;
        for (auto adjointFnArg : adjointFnBlock->getArguments()) {
            if (isa<QuregType, QubitType>(adjointFnArg.getType())) {
                quantumArgs.push_back(adjointFnArg);
            }
        }
        quantum::AdjointOp adjointOp =
            quantum::AdjointOp::create(builder, loc, TypeRange(quantumArgs), quantumArgs);

        Region *adjointRegion = &adjointOp.getRegion();
        Region &originalRegion = funcOp.getRegion();

        // Map the arguments with the original function
        IRMapping map;
        for (size_t i = 0; i < funcOp.getArguments().size(); i++) {
            Value arg = adjointFnOp.front().getArgument(i);
            Value argBlock = originalRegion.front().getArgument(i);
            if (isa<QuregType, QubitType>(argBlock.getType())) {
                continue;
            }
            map.map(argBlock, arg);
        }

        // Copy the original region in the adjoint region
        originalRegion.cloneInto(adjointRegion, map);

        // Replace the return of the quantum register with a quantum yield
        auto terminator = adjointRegion->front().getTerminator();
        ValueRange res = terminator->getOperands();
        TypeRange resTypes = terminator->getResultTypes();
        builder.setInsertionPointAfter(terminator);
        quantum::YieldOp::create(builder, loc, resTypes, res);

        // Return the adjoint operation in the adjoint function
        IRRewriter rewriter(builder);
        rewriter.eraseOp(terminator);
        builder.setInsertionPointAfter(adjointOp);
        func::ReturnOp::create(builder, loc, adjointOp.getResults());

        // Leave the adjoint func op to go back at the saved insertion
        builder.restoreInsertionPoint(insertionSaved);

        std::vector<Value> args;
        getAdjointCallOpArgs(callOp, args);

        // Call the adjoint func op
        auto adjointCallOp = func::CallOp::create(builder, loc, adjointFnOp, args);
        std::queue<Value> adjointCallOpResults;
        for (Value adjointCallResult : adjointCallOp->getResults()) {
            adjointCallOpResults.push(adjointCallResult);
        }
        for (auto callOperand : callOp.getArgOperands()) {
            if (!isa<QuregType, QubitType>(callOperand.getType())) {
                continue;
            }
            remappedValues.map(callOperand, adjointCallOpResults.front());
            adjointCallOpResults.pop();
        }
    }

    void visitOperation(scf::ForOp forOp, OpBuilder &builder)
    {
        SmallVector<Value> yieldedQValues =
            getQuantumValues(forOp.getBody()->getTerminator()->getOperands());
        if (yieldedQValues.empty()) {
            // This operation is purely classical
            return;
        }

        Value tape = cache.controlFlowTapes.at(forOp);
        // Popping the start, stop, and step implies that these are backwards relative to
        // the order they were pushed.
        Value step = ListPopOp::create(builder, forOp.getLoc(), tape);
        Value stop = ListPopOp::create(builder, forOp.getLoc(), tape);
        Value start = ListPopOp::create(builder, forOp.getLoc(), tape);

        SmallVector<Value> reversedResults;
        for (auto v : getQuantumValues(forOp.getResults())) {
            reversedResults.push_back(remappedValues.lookup(v));
        }

        auto replacedFor = scf::ForOp::create(
            builder, forOp.getLoc(), start, stop, step, /*iterArgsInit=*/reversedResults,
            [&](OpBuilder &bodyBuilder, Location loc, Value iv, ValueRange iterArgs) {
                OpBuilder::InsertionGuard insertionGuard(builder);
                builder.restoreInsertionPoint(bodyBuilder.saveInsertionPoint());

                for (auto [qvalue, iterArg] : llvm::zip_equal(yieldedQValues, iterArgs)) {
                    remappedValues.map(qvalue, iterArg);
                }

                generateImpl(forOp.getBodyRegion(), builder);

                SmallVector<Value> yields;
                for (auto v : getQuantumValues(forOp.getRegionIterArgs())) {
                    yields.push_back(remappedValues.lookup(v));
                }

                scf::YieldOp::create(builder, loc, yields);
            });

        for (auto [newForResult, initArg] :
             llvm::zip_equal(replacedFor.getResults(), getQuantumValues(forOp.getInitArgs()))) {
            remappedValues.map(initArg, newForResult);
        }
    }

    void visitOperation(scf::IfOp ifOp, OpBuilder &builder)
    {
        SmallVector<Value> yieldedQValues = getQuantumValues(ifOp.getResults());
        if (yieldedQValues.empty()) {
            // This operation is purely classical
            return;
        }

        Value tape = cache.controlFlowTapes.at(ifOp);
        Value condition = ListPopOp::create(builder, ifOp.getLoc(), tape);
        condition = index::CastSOp::create(builder, ifOp.getLoc(), builder.getI1Type(), condition);

        SmallVector<Value> reversedResults;
        for (auto v : getQuantumValues(ifOp.getResults())) {
            reversedResults.push_back(remappedValues.lookup(v));
        }

        // The quantum values are captured from outside rather than passed in through a
        // basic block argument. We thus need to traverse the region to look for it.
        auto findOldestQvaluesInRegion = [&](Region &region) -> SetVector<Value> {
            SetVector<Value> qvalues;
            for (Operation &innerOp : region.getOps()) {
                for (Value operand : innerOp.getOperands()) {
                    bool isDefinedFromOutsideRegion =
                        operand.getParentRegion()->isProperAncestor(&region);
                    if (isa<quantum::QuregType, quantum::QubitType>(operand.getType()) &&
                        isDefinedFromOutsideRegion) {
                        qvalues.insert(operand);
                    }
                }
            }
            assert(!qvalues.empty() && "failed to find quantum values in scf.if region");
            return qvalues;
        };

        auto getRegionBuilder = [&](Region &oldRegion) {
            return [&](OpBuilder &bodyBuilder, Location loc) {
                OpBuilder::InsertionGuard insertionGuard(builder);
                builder.restoreInsertionPoint(bodyBuilder.saveInsertionPoint());

                for (auto [qvalue, reversedResult] : llvm::zip_equal(
                         getQuantumValues(oldRegion.front().getTerminator()->getOperands()),
                         reversedResults)) {
                    remappedValues.map(qvalue, reversedResult);
                }

                generateImpl(oldRegion, builder);

                SmallVector<Value> yields;
                for (auto v : findOldestQvaluesInRegion(oldRegion)) {
                    yields.push_back(remappedValues.lookup(v));
                }
                scf::YieldOp::create(builder, loc, yields);
            };
        };
        auto reversedIf = scf::IfOp::create(builder, ifOp.getLoc(), condition,
                                            getRegionBuilder(ifOp.getThenRegion()),
                                            getRegionBuilder(ifOp.getElseRegion()));

        SetVector<Value> startingThenQvalues = findOldestQvaluesInRegion(ifOp.getThenRegion());
        SetVector<Value> startingElseQvalues = findOldestQvaluesInRegion(ifOp.getElseRegion());

        for (auto [idx, pair] :
             llvm::enumerate(llvm::zip_equal(startingThenQvalues, startingElseQvalues))) {
            auto [t, e] = pair;
            assert(t == e && "Expected the same input quantum values for both scf.if branches");
            remappedValues.map(t, reversedIf.getResult(idx));
        }
    }

    void visitOperation(scf::WhileOp whileOp, OpBuilder &builder)
    {
        SmallVector<Value> yieldedQValues =
            getQuantumValues(whileOp.getAfter().front().getTerminator()->getOperands());
        if (yieldedQValues.empty()) {
            // This operation is purely classical
            return;
        }

        Value tape = cache.controlFlowTapes.at(whileOp);
        Value numIterations = ListPopOp::create(builder, whileOp.getLoc(), tape);
        Value c0 = index::ConstantOp::create(builder, whileOp.getLoc(), 0);
        Value c1 = index::ConstantOp::create(builder, whileOp.getLoc(), 1);

        SmallVector<Value> iterArgsInit;
        for (auto v : getQuantumValues(whileOp.getResults())) {
            iterArgsInit.push_back(remappedValues.lookup(v));
        }

        auto replacedWhile = scf::ForOp::create(
            builder, whileOp.getLoc(), /*start=*/c0, /*stop=*/numIterations, /*step=*/c1,
            iterArgsInit,
            /*bodyBuilder=*/
            [&](OpBuilder &bodyBuilder, Location loc, Value iv, ValueRange iterArgs) {
                OpBuilder::InsertionGuard insertionGuard(builder);
                builder.restoreInsertionPoint(bodyBuilder.saveInsertionPoint());

                for (auto [qvalue, iterArg] : llvm::zip_equal(yieldedQValues, iterArgs)) {
                    remappedValues.map(qvalue, iterArg);
                }

                generateImpl(whileOp.getAfter(), builder);

                SmallVector<Value> yields;
                for (auto v : getQuantumValues(whileOp.getAfter().front().getArguments())) {
                    yields.push_back(remappedValues.lookup(v));
                }

                scf::YieldOp::create(builder, loc, yields);
            });

        for (auto [newWhileResult, initArg] :
             llvm::zip_equal(replacedWhile.getResults(), getQuantumValues(whileOp.getInits()))) {
            remappedValues.map(initArg, newWhileResult);
        }
    }

    void visitOperation(scf::IndexSwitchOp switchOp, OpBuilder &builder)
    {
        SmallVector<Value> yieldedQValues = getQuantumValues(switchOp.getResults());
        if (yieldedQValues.empty()) {
            // This operation is purely classical
            return;
        }

        Value tape = cache.controlFlowTapes.at(switchOp.getOperation());
        Value index = ListPopOp::create(builder, switchOp.getLoc(), tape);

        SmallVector<Value> reversedResults;
        for (auto v : getQuantumValues(switchOp.getResults())) {
            reversedResults.push_back(remappedValues.lookup(v));
        }

        auto findRootQvaluesInRegion = [&](Region &region) -> SetVector<Value> {
            SetVector<Value> qvalues;
            for (Operation &innerOp : region.getOps()) {
                for (Value operand : innerOp.getOperands()) {
                    bool isDefinedFromOutsideRegion =
                        operand.getParentRegion()->isProperAncestor(&region);
                    if (isa<quantum::QuregType, quantum::QubitType>(operand.getType()) &&
                        isDefinedFromOutsideRegion) {
                        qvalues.insert(operand);
                    }
                }
            }
            assert(!qvalues.empty() && "failed to find quantum values in scf.index_switch region");
            return qvalues;
        };

        auto newSwitchOp =
            scf::IndexSwitchOp::create(builder, switchOp.getLoc(), TypeRange{reversedResults},
                                       index, switchOp.getCases(), switchOp.getNumCases());

        auto fillRegion = [&](Region &oldRegion, Region &newRegion) {
            OpBuilder::InsertionGuard guard(builder);
            newRegion.push_back(new Block());
            builder.setInsertionPointToStart(&newRegion.front());

            for (auto [qvalue, reversedResult] :
                 llvm::zip_equal(getQuantumValues(oldRegion.front().getTerminator()->getOperands()),
                                 reversedResults)) {
                remappedValues.map(qvalue, reversedResult);
            }

            generateImpl(oldRegion, builder);

            SmallVector<Value> yields;
            for (auto v : findRootQvaluesInRegion(oldRegion)) {
                yields.push_back(remappedValues.lookup(v));
            }
            scf::YieldOp::create(builder, switchOp.getLoc(), yields);
        };

        // Case regions:
        for (auto [oldCaseRegion, newCaseRegion] :
             llvm::zip_equal(switchOp.getCaseRegions(), newSwitchOp.getCaseRegions())) {
            fillRegion(oldCaseRegion, newCaseRegion);
        }

        // Default region:
        fillRegion(switchOp.getDefaultRegion(), newSwitchOp.getDefaultRegion());

        for (auto [idx, v] :
             llvm::enumerate(findRootQvaluesInRegion(switchOp.getDefaultRegion()))) {
            remappedValues.map(v, newSwitchOp.getResult(idx));
        }
    }

  private:
    IRMapping &remappedValues;
    QuantumCache &cache;
    bool generationFailed = false;
};

} // namespace

namespace catalyst {
namespace quantum {

LogicalResult generateAdjointReversePass(Region &region, OpBuilder &builder,
                                         IRMapping &remappedValues, QuantumCache &cache)
{
    AdjointGenerator generator{remappedValues, cache};
    return generator.generate(region, builder);
}

} // namespace quantum
} // namespace catalyst
