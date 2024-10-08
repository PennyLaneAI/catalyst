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

#include "Zne.hpp"

#include <algorithm>
#include <deque>
#include <iostream>
#include <sstream>
#include <vector>

#include "llvm/ADT/SmallPtrSet.h"

#include "Catalyst/Utils/CallGraph.h"
#include "Mitigation/IR/MitigationOps.h"
#include "Quantum/IR/QuantumOps.h"
#include "Quantum/Utils/RemoveQuantum.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Index/IR/IndexOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/IRMapping.h"

namespace catalyst {
namespace mitigation {

LogicalResult ZneLowering::match(mitigation::ZneOp op) const { return success(); }

// TODO: Optimize the traversal of call graphs (currently used twice)
// Also all functions exploree in the call graph get their ZNE version.
void ZneLowering::rewrite(mitigation::ZneOp op, PatternRewriter &rewriter) const
{
    auto point = rewriter.saveInsertionPoint();
    Location loc = op.getLoc();
    auto moduleOp = op->getParentOfType<ModuleOp>();
    // Number of folds
    auto numFolds = op.getNumFolds();
    RankedTensorType numFoldType = cast<RankedTensorType>(numFolds.getType());
    const auto sizeInt = numFoldType.getDimSize(0);

    // Folding type
    auto foldingAlgorithm = op.getFolding();
    auto calleeOp = SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(op, op.getCalleeAttr());

    func::FuncOp fnFoldedOp;

    // Traverse the callgraph and replace qnodes with folded qnodes.
    if (!calleeOp->hasAttr("qnode")) {
        traverseCallGraph(calleeOp, /*symbolTable=*/nullptr, [&](func::FuncOp funcOp) {
            if (!funcOp->hasAttr("qnode")) {

                rewriter.setInsertionPointToStart(moduleOp.getBody());
                TypeRange originalTypes = funcOp.getArgumentTypes();
                SmallVector<Type> typesFolded(originalTypes.begin(), originalTypes.end());
                Type indexType = rewriter.getIndexType();
                typesFolded.push_back(indexType);
                FunctionType fnFoldedType = FunctionType::get(op.getContext(),
                                                              /*inputs=*/typesFolded,
                                                              /*outputs=*/funcOp.getResultTypes());
                std::string fnFoldedName = funcOp.getName().str() + ".zne";
                auto currentFnFoldedOp =
                    rewriter.create<func::FuncOp>(loc, fnFoldedName, fnFoldedType);

                rewriter.cloneRegionBefore(funcOp.getBody(), currentFnFoldedOp.getBody(),
                                           currentFnFoldedOp.end());

                Block *currentFnFoldedOpBlock = &currentFnFoldedOp.getBody().front();
                rewriter.setInsertionPointToStart(currentFnFoldedOpBlock);
                currentFnFoldedOpBlock->addArgument(currentFnFoldedOp.getArgumentTypes().back(),
                                                    loc);

                bool containQnodes = false;
                funcOp.walk([&](func::CallOp op) {
                    auto insideFuncOp =
                        SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(op, op.getCalleeAttr());
                    if (insideFuncOp->hasAttr("qnode")) {
                        containQnodes = true;
                        return WalkResult::interrupt();
                    }
                    return WalkResult::advance();
                });

                // Replace calls with calls.zne
                if (containQnodes) {
                    currentFnFoldedOp.walk([&](func::CallOp callOp) {
                        PatternRewriter::InsertionGuard insertGuard(rewriter);
                        func::FuncOp funcOp = SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(
                            op, callOp.getCalleeAttr());
                        std::string foldedName = callOp.getCalleeAttrName().str() + ".folded";
                        func::FuncOp foldedOp = SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(
                            moduleOp, rewriter.getStringAttr(foldedName));
                        if (funcOp->hasAttr("qnode") and !foldedOp) {
                            // Create the folded circuit function
                            auto foldedCircuitAttr =
                                getOrInsertFoldedCircuit(loc, rewriter, funcOp, foldingAlgorithm);
                            func::FuncOp foldedOp =
                                SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(
                                    moduleOp, foldedCircuitAttr);
                            std::vector<Value> args = {callOp.getArgOperands().begin(),
                                                       callOp.getArgOperands().end()};
                            args.push_back(currentFnFoldedOp.getArguments().back());
                            rewriter.setInsertionPoint(callOp);
                            rewriter.replaceOpWithNewOp<func::CallOp>(callOp, foldedOp, args);
                        }
                        else if (funcOp->hasAttr("qnode") and foldedOp) {
                            rewriter.setInsertionPoint(callOp);
                            std::vector<Value> args = {callOp.getArgOperands().begin(),
                                                       callOp.getArgOperands().end()};
                            args.push_back(currentFnFoldedOp.getArguments().back());
                            rewriter.replaceOpWithNewOp<func::CallOp>(callOp, foldedOp, args);
                        }
                    });
                    rewriter.restoreInsertionPoint(point);
                }
            }
        });
        std::string fnName = calleeOp.getName().str() + ".zne";
        FlatSymbolRefAttr foldedOpRefAttr = SymbolRefAttr::get(op.getContext(), fnName);
        fnFoldedOp = SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(moduleOp, foldedOpRefAttr);

        traverseCallGraph(fnFoldedOp, /*symbolTable=*/nullptr, [&](func::FuncOp funcOp) {
            funcOp.walk([&](func::CallOp callOp) {
                std::string fnName = callOp.getCallee().str() + ".zne";
                FlatSymbolRefAttr foldedOpRefAttr = SymbolRefAttr::get(op.getContext(), fnName);
                auto currentFnFoldedOp =
                    SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(moduleOp, foldedOpRefAttr);
                if (currentFnFoldedOp) {
                    PatternRewriter::InsertionGuard insertionGuard(rewriter);
                    rewriter.setInsertionPointToStart(&currentFnFoldedOp.getFunctionBody().front());

                    callOp.setCallee(currentFnFoldedOp.getName());
                    auto parentFunc = callOp->getParentOfType<func::FuncOp>();
                    callOp.getOperandsMutable().append(parentFunc.getArguments().back());
                }
            });
        });
    }
    else {
        // Create the folded circuit function
        FlatSymbolRefAttr foldedOpRefAttr =
            getOrInsertFoldedCircuit(loc, rewriter, calleeOp, foldingAlgorithm);
        fnFoldedOp = SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(calleeOp, foldedOpRefAttr);
    }
    rewriter.setInsertionPoint(op);
    RankedTensorType resultType = cast<RankedTensorType>(op.getResultTypes().front());

    // Loop over the num fold to create a folded circuit per factor
    Value c0 = rewriter.create<index::ConstantOp>(loc, 0);
    Value c1 = rewriter.create<index::ConstantOp>(loc, 1);
    Value size = rewriter.create<index::ConstantOp>(loc, sizeInt);
    // Initialize the results as empty tensor

    Value results =
        rewriter.create<tensor::EmptyOp>(loc, resultType.getShape(), resultType.getElementType());
    Value resultValues =
        rewriter
            .create<scf::ForOp>(
                loc, c0, size, c1, /*iterArgsInit=*/results,
                [&](OpBuilder &builder, Location loc, Value i, ValueRange iterArgs) {
                    std::vector<Value> newArgs(op.getArgs().begin(), op.getArgs().end());
                    SmallVector<Value> index = {i};
                    Value numFold = builder.create<tensor::ExtractOp>(loc, numFolds, index);
                    Value numFoldCasted =
                        builder.create<index::CastSOp>(loc, builder.getIndexType(), numFold);
                    newArgs.push_back(numFoldCasted);
                    func::CallOp callOp = builder.create<func::CallOp>(loc, fnFoldedOp, newArgs);

                    int64_t numResults = callOp.getNumResults();

                    // Measurements
                    ValueRange resultValuesMulti = callOp.getResults();
                    SmallVector<Value> vectorResultsMulti;
                    // Create a tensor
                    for (Value resultValue : resultValuesMulti) {
                        Value resultExtracted;
                        if (isa<RankedTensorType>(resultValue.getType())) {
                            resultExtracted = builder.create<tensor::ExtractOp>(loc, resultValue);
                        }
                        else {
                            resultExtracted = resultValue;
                        }
                        vectorResultsMulti.push_back(resultExtracted);
                    }
                    SmallVector<int64_t> resShape = {numResults};
                    Type type = RankedTensorType::get(resShape, vectorResultsMulti[0].getType());
                    auto tensorResults =
                        builder.create<tensor::FromElementsOp>(loc, type, vectorResultsMulti);
                    Value sizeResultsValue = rewriter.create<index::ConstantOp>(loc, numResults);
                    Value resultValuesFor =
                        rewriter
                            .create<scf::ForOp>(
                                loc, c0, sizeResultsValue, c1,
                                /*iterArgsInit=*/iterArgs.front(),
                                [&](OpBuilder &builder, Location loc, Value j,
                                    ValueRange iterArgsIn) {
                                    Value resultExtracted =
                                        builder.create<tensor::ExtractOp>(loc, tensorResults, j);
                                    SmallVector<Value> indices;
                                    if (numResults == 1) {
                                        indices = {i};
                                    }
                                    else {
                                        indices = {i, j};
                                    }
                                    Value resultInserted = builder.create<tensor::InsertOp>(
                                        loc, resultExtracted, iterArgsIn.front(), indices);

                                    builder.create<scf::YieldOp>(loc, resultInserted);
                                })
                            .getResult(0);
                    builder.create<scf::YieldOp>(loc, resultValuesFor);
                })
            .getResult(0);
    // Replace the original results
    rewriter.replaceOp(op, resultValues);
}
// In *.cpp module only, to keep extraneous headers out of *.hpp
FlatSymbolRefAttr globalFolding(Location loc, PatternRewriter &rewriter, std::string fnFoldedName,
                                StringAttr lib, StringAttr name, StringAttr kwargs,
                                int64_t numberQubits, FunctionType fnFoldedType,
                                SmallVector<Type> typesFolded, func::FuncOp fnFoldedOp,
                                func::FuncOp fnAllocOp, func::FuncOp fnWithoutMeasurementsOp,
                                func::FuncOp fnWithMeasurementsOp)
{
    // Function folded: Create the folded circuit (withoutMeasurement *
    // Adjoint(withoutMeasurement))**num_fold * withMeasurements
    Type qregType = quantum::QuregType::get(rewriter.getContext());

    rewriter.setInsertionPointToStart(fnFoldedOp.addEntryBlock());
    // Loop control variables
    Value c0 = rewriter.create<index::ConstantOp>(loc, 0);
    Value c1 = rewriter.create<index::ConstantOp>(loc, 1);
    TypedAttr numberQubitsAttr = rewriter.getI64IntegerAttr(numberQubits);
    Value numberQubitsValue = rewriter.create<arith::ConstantOp>(loc, numberQubitsAttr);
    rewriter.create<quantum::DeviceInitOp>(loc, lib, name, kwargs);

    Value allocQreg = rewriter.create<func::CallOp>(loc, fnAllocOp, numberQubitsValue).getResult(0);

    int64_t sizeArgs = fnFoldedOp.getArguments().size();
    Value size = fnFoldedOp.getArgument(sizeArgs - 1);
    // Add scf for loop to create the folding
    Value loopedQreg =
        rewriter
            .create<scf::ForOp>(
                loc, c0, size, c1, /*iterArgsInit=*/allocQreg,
                [&](OpBuilder &builder, Location loc, Value i, ValueRange iterArgs) {
                    Value qreg = iterArgs.front();
                    std::vector<Value> argsAndQreg(fnFoldedOp.getArguments().begin(),
                                                   fnFoldedOp.getArguments().end());
                    argsAndQreg.pop_back();
                    argsAndQreg.push_back(qreg);

                    // Call the function without measurements
                    Value fnWithoutMeasurementsQreg =
                        builder.create<func::CallOp>(loc, fnWithoutMeasurementsOp, argsAndQreg)
                            .getResult(0);

                    // Call the function without measurements in an adjoint region
                    auto adjointOp = builder.create<quantum::AdjointOp>(loc, qregType,
                                                                        fnWithoutMeasurementsQreg);
                    Region *adjointRegion = &adjointOp.getRegion();
                    Block *adjointBlock = builder.createBlock(adjointRegion, {}, qregType, loc);

                    std::vector<Value> argsAndQregAdjoint(fnFoldedOp.getArguments().begin(),
                                                          fnFoldedOp.getArguments().end());
                    argsAndQregAdjoint.pop_back();
                    argsAndQregAdjoint.push_back(adjointBlock->getArgument(0));
                    Value fnWithoutMeasurementsAdjointQreg =
                        builder
                            .create<func::CallOp>(loc, fnWithoutMeasurementsOp, argsAndQregAdjoint)
                            .getResult(0);
                    builder.create<quantum::YieldOp>(loc, fnWithoutMeasurementsAdjointQreg);
                    builder.setInsertionPointAfter(adjointOp);
                    builder.create<scf::YieldOp>(loc, adjointOp.getResult());
                })
            .getResult(0);
    std::vector<Value> argsAndRegMeasurement(fnFoldedOp.getArguments().begin(),
                                             fnFoldedOp.getArguments().end());
    argsAndRegMeasurement.pop_back();
    argsAndRegMeasurement.push_back(loopedQreg);
    ValueRange funcFolded =
        rewriter.create<func::CallOp>(loc, fnWithMeasurementsOp, argsAndRegMeasurement)
            .getResults();
    // Remove device
    rewriter.create<quantum::DeviceReleaseOp>(loc);
    rewriter.create<func::ReturnOp>(loc, funcFolded);
    return SymbolRefAttr::get(rewriter.getContext(), fnFoldedName);
}
// In *.cpp module only, to keep extraneous headers out of *.hpp
FlatSymbolRefAttr randomLocalFolding(PatternRewriter &rewriter, std::string fnFoldedName,
                                     func::FuncOp fnFoldedOp, Value c0, Value c1)
{
    // TODO: Implement.

    // Can't throw, because disabled by compilation.
    // throw std::logic_error("Random local folding not implemented!");

    return FlatSymbolRefAttr();
}
// In *.cpp module only, to keep extraneous headers out of *.hpp
FlatSymbolRefAttr allLocalFolding(PatternRewriter &rewriter, std::string fnFoldedName,
                                  func::FuncOp fnFoldedOp, Value c0, Value c1)
{
    int64_t sizeArgs = fnFoldedOp.getArguments().size();
    Value size = fnFoldedOp.getArgument(sizeArgs - 1);

    // Walk through the operations in fnFoldedOp
    fnFoldedOp.walk([&](quantum::QuantumGate op) {
        rewriter.setInsertionPoint(op);
        auto loc = op->getLoc();
        const std::vector<Value> opQubitArgs = op.getQubitOperands();

        // Insert a for loop immediately before each quantum::QuantumGate
        const auto forVal =
            rewriter
                .create<scf::ForOp>(
                    loc, c0, size, c1, /*iterArgsInit=*/opQubitArgs,
                    [&](OpBuilder &builder, Location loc, Value i, ValueRange iterArgs) {
                        // Create adjoint and original operations
                        quantum::QuantumGate origOp =
                            dyn_cast<quantum::QuantumGate>(builder.clone(*op));
                        origOp.setQubitOperands(iterArgs);
                        auto origOpVal = origOp->getResults();

                        quantum::QuantumGate adjointOp =
                            dyn_cast<quantum::QuantumGate>(builder.clone(*origOp));
                        adjointOp.setQubitOperands(origOpVal);
                        adjointOp.setAdjointFlag(!adjointOp.getAdjointFlag());
                        auto adjointOpVal = adjointOp->getResults();

                        // Yield the qubits.
                        builder.create<scf::YieldOp>(loc, adjointOpVal);
                    })
                .getResults();

        op.setQubitOperands(forVal);

        return WalkResult::advance();
    });

    // Return the function symbol reference
    return SymbolRefAttr::get(rewriter.getContext(), fnFoldedName);
}
FlatSymbolRefAttr ZneLowering::getOrInsertFoldedCircuit(Location loc, PatternRewriter &rewriter,
                                                        func::FuncOp op, Folding foldingAlgorithm)
{
    OpBuilder::InsertionGuard guard(rewriter);
    ModuleOp moduleOp = op->getParentOfType<ModuleOp>();
    std::string fnFoldedName = op.getName().str() + ".folded";

    MLIRContext *ctx = rewriter.getContext();

    if (moduleOp.lookupSymbol<func::FuncOp>(fnFoldedName)) {
        return SymbolRefAttr::get(ctx, fnFoldedName);
    }

    // Original function
    func::FuncOp fnOp = SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(op, op.getNameAttr());

    // Set insertion in the module
    rewriter.setInsertionPointToStart(moduleOp.getBody());

    // Get the number of qubits
    const int64_t numberQubits =
        (*fnOp.getOps<quantum::AllocOp>().begin()).getNqubitsAttr().value_or(0);
    // Get the device
    quantum::DeviceInitOp deviceInitOp = *fnOp.getOps<quantum::DeviceInitOp>().begin();

    StringAttr lib = deviceInitOp.getLibAttr();
    StringAttr name = deviceInitOp.getNameAttr();
    StringAttr kwargs = deviceInitOp.getKwargsAttr();

    TypeRange originalTypes = op.getArgumentTypes();
    SmallVector<Type> typesFolded(originalTypes.begin(), originalTypes.end());
    Type indexType = rewriter.getIndexType();
    typesFolded.push_back(indexType);

    rewriter.setInsertionPointToStart(moduleOp.getBody());

    FunctionType fnFoldedType = FunctionType::get(ctx, /*inputs=*/
                                                  typesFolded,
                                                  /*outputs=*/fnOp.getResultTypes());

    func::FuncOp fnFoldedOp = rewriter.create<func::FuncOp>(loc, fnFoldedName, fnFoldedType);
    fnFoldedOp.setPrivate();
    if (foldingAlgorithm == Folding(1)) {
        // Quantum Alloc function
        FlatSymbolRefAttr quantumAllocRefAttr = getOrInsertQuantumAlloc(loc, rewriter, op);
        func::FuncOp fnAllocOp =
            SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(op, quantumAllocRefAttr);

        // Function without measurements: Create function without measurements and with qreg as last
        // argument
        FlatSymbolRefAttr fnWithoutMeasurementsRefAttr =
            getOrInsertFnWithoutMeasurements(loc, rewriter, op);
        func::FuncOp fnWithoutMeasurementsOp =
            SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(op, fnWithoutMeasurementsRefAttr);

        // Function with measurements: Modify the original function to take a quantum register as
        // last arg and keep measurements
        FlatSymbolRefAttr fnWithMeasurementsRefAttr =
            getOrInsertFnWithMeasurements(loc, rewriter, op);
        func::FuncOp fnWithMeasurementsOp =
            SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(op, fnWithMeasurementsRefAttr);

        return globalFolding(loc, rewriter, fnFoldedName, lib, name, kwargs, numberQubits,
                             fnFoldedType, typesFolded, fnFoldedOp, fnAllocOp,
                             fnWithoutMeasurementsOp, fnWithMeasurementsOp);
    }
    rewriter.cloneRegionBefore(fnOp.getBody(), fnFoldedOp.getBody(), fnFoldedOp.end());

    Block *fnFoldedOpBlock = &fnFoldedOp.getBody().front();
    rewriter.setInsertionPointToStart(fnFoldedOpBlock);
    // Loop control variables
    Value c0 = rewriter.create<index::ConstantOp>(loc, 0);
    Value c1 = rewriter.create<index::ConstantOp>(loc, 1);

    fnFoldedOpBlock->addArgument(fnFoldedOp.getArgumentTypes().back(), loc);

    if (foldingAlgorithm == Folding(2)) {
        return allLocalFolding(rewriter, fnFoldedName, fnFoldedOp, c0, c1);
    }
    // Else, if (foldingAlgorithm == Folding(3)):
    return randomLocalFolding(rewriter, fnFoldedName, fnFoldedOp, c0, c1);
}
FlatSymbolRefAttr ZneLowering::getOrInsertQuantumAlloc(Location loc, PatternRewriter &rewriter,
                                                       func::FuncOp op)
{
    // Quantum Alloc function
    MLIRContext *ctx = rewriter.getContext();
    OpBuilder::InsertionGuard guard(rewriter);
    ModuleOp moduleOp = op->getParentOfType<ModuleOp>();
    Type qregType = quantum::QuregType::get(rewriter.getContext());

    std::string fnAllocName = op.getName().str() + ".quantumAlloc";

    if (moduleOp.lookupSymbol<func::FuncOp>(fnAllocName)) {
        return SymbolRefAttr::get(ctx, fnAllocName);
    }
    Type i64Type = rewriter.getI64Type();
    FunctionType fnAllocType = FunctionType::get(ctx, /*inputs=*/
                                                 i64Type,
                                                 /*outputs=*/qregType);
    func::FuncOp fnAlloc = rewriter.create<func::FuncOp>(loc, fnAllocName, fnAllocType);
    fnAlloc.setPrivate();
    Block *allocBloc = fnAlloc.addEntryBlock();
    rewriter.setInsertionPointToStart(allocBloc);
    Value nQubits = allocBloc->getArgument(0);
    IntegerAttr intAttr{};
    auto qreg = rewriter.create<quantum::AllocOp>(loc, qregType, nQubits, intAttr);
    rewriter.create<func::ReturnOp>(loc, qreg.getResult());
    return SymbolRefAttr::get(ctx, fnAllocName);
}
FlatSymbolRefAttr ZneLowering::getOrInsertFnWithoutMeasurements(Location loc,
                                                                PatternRewriter &rewriter,
                                                                func::FuncOp op)
{
    MLIRContext *ctx = rewriter.getContext();
    OpBuilder::InsertionGuard guard(rewriter);
    ModuleOp moduleOp = op->getParentOfType<ModuleOp>();
    std::string fnWithoutMeasurementsName = op.getName().str() + ".withoutMeasurements";
    if (moduleOp.lookupSymbol<func::FuncOp>(fnWithoutMeasurementsName)) {
        return SymbolRefAttr::get(ctx, fnWithoutMeasurementsName);
    }
    Type qregType = quantum::QuregType::get(rewriter.getContext());
    func::FuncOp fnOp = SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(op, op.getNameAttr());
    TypeRange originalTypes = op.getArgumentTypes();

    SmallVector<Type> typesWithoutMeasurements(originalTypes.begin(), originalTypes.end());
    typesWithoutMeasurements.push_back(qregType);

    FunctionType fnWithoutMeasurementsType = FunctionType::get(ctx, /*inputs=*/
                                                               typesWithoutMeasurements,
                                                               /*outputs=*/qregType);
    func::FuncOp fnWithoutMeasurementsOp =
        rewriter.create<func::FuncOp>(loc, fnWithoutMeasurementsName, fnWithoutMeasurementsType);
    fnWithoutMeasurementsOp.setPrivate();
    rewriter.cloneRegionBefore(fnOp.getBody(), fnWithoutMeasurementsOp.getBody(),
                               fnWithoutMeasurementsOp.end());
    Block *fnWithoutMeasurementsBlock = &fnWithoutMeasurementsOp.front();
    fnWithoutMeasurementsBlock->addArgument(qregType, loc);
    quantum::AllocOp allocOp = *fnWithoutMeasurementsOp.getOps<quantum::AllocOp>().begin();

    auto lastArgIndex = fnWithoutMeasurementsBlock->getArguments().size();
    allocOp.replaceAllUsesWith(fnWithoutMeasurementsBlock->getArgument(lastArgIndex - 1));

    rewriter.eraseOp(allocOp);
    quantum::DeviceInitOp deviceInitOp =
        *fnWithoutMeasurementsOp.getOps<quantum::DeviceInitOp>().begin();
    rewriter.eraseOp(deviceInitOp);
    quantum::DeviceReleaseOp deviceReleaseOp =
        *fnWithoutMeasurementsOp.getOps<quantum::DeviceReleaseOp>().begin();
    rewriter.eraseOp(deviceReleaseOp);
    rewriter.setInsertionPointToStart(&fnWithoutMeasurementsOp.getBody().front());

    Operation *lastOp;
    fnWithoutMeasurementsOp.walk([&](quantum::DeallocOp deallocOp) { lastOp = deallocOp; });
    fnWithoutMeasurementsOp.walk(
        [&](func::ReturnOp returnOp) { returnOp->setOperands(lastOp->getOperands()); });

    quantum::DeallocOp localDealloc = *fnWithoutMeasurementsOp.getOps<quantum::DeallocOp>().begin();
    rewriter.eraseOp(localDealloc);
    quantum::replaceQuantumMeasurements(fnWithoutMeasurementsOp, rewriter);
    return SymbolRefAttr::get(ctx, fnWithoutMeasurementsName);
}
FlatSymbolRefAttr
ZneLowering::getOrInsertFnWithMeasurements(Location loc, PatternRewriter &rewriter, func::FuncOp op)
{
    MLIRContext *ctx = rewriter.getContext();
    OpBuilder::InsertionGuard guard(rewriter);
    ModuleOp moduleOp = op->getParentOfType<ModuleOp>();

    std::string fnWithMeasurementsName = op.getName().str() + ".withMeasurements";

    if (moduleOp.lookupSymbol<func::FuncOp>(fnWithMeasurementsName)) {
        return SymbolRefAttr::get(ctx, fnWithMeasurementsName);
    }

    Type qregType = quantum::QuregType::get(rewriter.getContext());
    func::FuncOp fnOp = SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(op, op.getNameAttr());
    TypeRange originalTypes = op.getArgumentTypes();

    SmallVector<Type> typesWithQreg(originalTypes.begin(), originalTypes.end());
    typesWithQreg.push_back(qregType);

    FunctionType fnWithMeasurementsType = FunctionType::get(ctx, /*inputs=*/
                                                            typesWithQreg,
                                                            /*outputs=*/fnOp.getResultTypes());
    func::FuncOp fnWithMeasurementsOp =
        rewriter.create<func::FuncOp>(loc, fnWithMeasurementsName, fnWithMeasurementsType);
    fnWithMeasurementsOp.setPrivate();
    rewriter.cloneRegionBefore(fnOp.getBody(), fnWithMeasurementsOp.getBody(),
                               fnWithMeasurementsOp.end());
    Block *fnWithMeasurementsBlock = &fnWithMeasurementsOp.front();
    fnWithMeasurementsBlock->addArgument(qregType, loc);
    quantum::DeviceInitOp deviceInitOp =
        *fnWithMeasurementsOp.getOps<quantum::DeviceInitOp>().begin();
    rewriter.eraseOp(deviceInitOp);
    quantum::DeviceReleaseOp deviceReleaseOp =
        *fnWithMeasurementsOp.getOps<quantum::DeviceReleaseOp>().begin();
    rewriter.eraseOp(deviceReleaseOp);
    quantum::AllocOp allocOpWithMeasurements =
        *fnWithMeasurementsOp.getOps<quantum::AllocOp>().begin();

    auto lastArgQregIndex = fnWithMeasurementsBlock->getArguments().size();
    allocOpWithMeasurements.replaceAllUsesWith(
        fnWithMeasurementsBlock->getArgument(lastArgQregIndex - 1));
    rewriter.eraseOp(allocOpWithMeasurements);
    return SymbolRefAttr::get(ctx, fnWithMeasurementsName);
}
} // namespace mitigation
} // namespace catalyst
