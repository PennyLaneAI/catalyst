// Copyright 2022-2023 Xanadu Quantum Technologies Inc.

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

#include "Mitigation/IR/MitigationOps.h"
#include "Quantum/IR/QuantumOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Index/IR/IndexOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/IRMapping.h"

namespace catalyst {
namespace mitigation {

LogicalResult ZneLowering::match(mitigation::ZneOp op) const { return success(); }

void ZneLowering::rewrite(mitigation::ZneOp op, PatternRewriter &rewriter) const
{
    Location loc = op.getLoc();

    auto scalarFactors = op.getScalarFactors();
    RankedTensorType scalarFactorType = scalarFactors.getType().cast<RankedTensorType>();
    auto sizeInt = scalarFactorType.getDimSize(0);

    // Create the folded circuit function

    FlatSymbolRefAttr foldedCircuitRefAttr =
        getOrInsertFoldedCircuit(loc, rewriter, op, scalarFactorType.getElementType());
    func::FuncOp foldedCircuit =
        SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(op, foldedCircuitRefAttr);

    // Loop over the scalars
    Value c0 = rewriter.create<index::ConstantOp>(loc, 0);
    Value c1 = rewriter.create<index::ConstantOp>(loc, 1);
    Value size = rewriter.create<index::ConstantOp>(loc, sizeInt);
    Value results;

    // Value resultValues = rewriter
    //                          .create<scf::ForOp>(loc, c0, size, c1, /*iterArgsInit=*/results,
    //                                              [&](OpBuilder &builder, Location loc, Value i,
    //                                                  ValueRange iterArgs) {
    //                                                  // Extract scalar factor
    //                                                  // Call the folded circuit function for each
    //                                                  // scalar factor and insert in the tensor
    //                                                  // results
    //                                                  builder.create<scf::YieldOp>(loc, iterArgs);
    //                                              })
    //                          .getResult(0);
    std::vector<Value> newArgs{op.getArgs().begin(), op.getArgs().end()};
    Value indexValue = rewriter.create<index::ConstantOp>(loc, 0);
    auto scalarFactor = rewriter.create<tensor::ExtractOp>(loc, scalarFactors, indexValue);
    newArgs.push_back(scalarFactor);
    auto resultsValues = rewriter.create<func::CallOp>(loc, foldedCircuit, newArgs);
    // Call the folded circuit
    rewriter.replaceOp(op, resultsValues);
}

FlatSymbolRefAttr ZneLowering::getOrInsertFoldedCircuit(Location loc, PatternRewriter &rewriter,
                                                        mitigation::ZneOp op, Type scalarType)
{
    MLIRContext *ctx = rewriter.getContext();

    OpBuilder::InsertionGuard guard(rewriter);
    ModuleOp moduleOp = op->getParentOfType<ModuleOp>();
    rewriter.setInsertionPointToStart(moduleOp.getBody());
    // Original function
    func::FuncOp fnOp = SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(op, op.getCalleeAttr());
    TypeRange originalTypes = op.getArgs().getTypes();

    Type qregType = quantum::QuregType::get(rewriter.getContext());
    // Alloc function
    std::string fnNameAlloc = op.getCallee().str() + ".alloc";
    Type i64Type = rewriter.getI64Type();
    FunctionType fnAllocType = FunctionType::get(ctx, /*inputs=*/
                                                 i64Type,
                                                 /*outputs=*/qregType);

    func::FuncOp fnAlloc = rewriter.create<func::FuncOp>(loc, fnNameAlloc, fnAllocType);
    fnAlloc.setPrivate();

    Block *allocBloc = fnAlloc.addEntryBlock();
    rewriter.setInsertionPointToStart(allocBloc);
    Value nQubits = allocBloc->getArgument(0);
    IntegerAttr intAttr;
    auto qreg = rewriter.create<quantum::AllocOp>(loc, qregType, nQubits, intAttr);
    rewriter.create<func::ReturnOp>(loc, qreg.getResult());

    // Modify the function to take a qreg
    std::string fnNameWithQregArg = op.getCallee().str() + ".qregarg";
    SmallVector<Type> typesWithQreg = {originalTypes.begin(), originalTypes.end()};
    typesWithQreg.push_back(qregType);

    FunctionType fnWithQregType = FunctionType::get(ctx, /*inputs=*/
                                                    typesWithQreg,
                                                    /*outputs=*/fnOp.getResultTypes());
    rewriter.setInsertionPointToStart(moduleOp.getBody());
    func::FuncOp fnWithQregArg =
        rewriter.create<func::FuncOp>(loc, fnNameWithQregArg, fnWithQregType);
    fnWithQregArg.setPrivate();
    rewriter.cloneRegionBefore(fnOp.getBody(), fnWithQregArg.getBody(), fnWithQregArg.end());
    Block *fnWithoutQregBlock = &fnWithQregArg.front();
    fnWithoutQregBlock->addArgument(qregType, loc);
    quantum::AllocOp allocOpWithQreg = *fnWithQregArg.getOps<quantum::AllocOp>().begin();

    auto lastArgWithQreg = fnWithoutQregBlock->getArguments().size();
    allocOpWithQreg.replaceAllUsesWith(fnWithoutQregBlock->getArgument(lastArgWithQreg - 1));
    rewriter.eraseOp(allocOpWithQreg);

    // Function without measurements
    std::string fnNameWithoutMeasurements = op.getCallee().str() + ".withoutmeasurements";
    SmallVector<Type> typesWithoutMeasurements = {originalTypes.begin(), originalTypes.end()};
    typesWithoutMeasurements.push_back(qregType);

    FunctionType fnWithoutMeasurementsType = FunctionType::get(ctx, /*inputs=*/
                                                               typesWithoutMeasurements,
                                                               /*outputs=*/qregType);
    func::FuncOp fnWithoutMeasurementsOp =
        rewriter.create<func::FuncOp>(loc, fnNameWithoutMeasurements, fnWithoutMeasurementsType);
    fnWithoutMeasurementsOp.setPrivate();
    rewriter.cloneRegionBefore(fnOp.getBody(), fnWithoutMeasurementsOp.getBody(),
                               fnWithoutMeasurementsOp.end());
    Block *fnWithoutMeasurementsBlock = &fnWithoutMeasurementsOp.front();
    fnWithoutMeasurementsBlock->addArgument(qregType, loc);
    quantum::AllocOp allocOp = *fnWithoutMeasurementsOp.getOps<quantum::AllocOp>().begin();

    auto lastArg = fnWithoutMeasurementsBlock->getArguments().size();
    allocOp.replaceAllUsesWith(fnWithoutMeasurementsBlock->getArgument(lastArg - 1));

    std::optional<int64_t> nQubitsOpt = allocOp.getNqubitsAttr();
    int64_t nQubitsInt = nQubitsOpt.value_or(0);

    rewriter.eraseOp(allocOp);
    rewriter.setInsertionPointToStart(&fnWithoutMeasurementsOp.getBody().front());

    std::vector<Operation *> insertOps;
    fnWithoutMeasurementsOp.walk(
        [&](quantum::InsertOp insertOp) { insertOps.push_back(insertOp); });
    fnWithoutMeasurementsOp.walk(
        [&](func::ReturnOp returnOp) { returnOp->setOperands(insertOps.back()->getResult(0)); });

    quantum::DeallocOp localDealloc = *fnWithoutMeasurementsOp.getOps<quantum::DeallocOp>().begin();
    rewriter.eraseOp(localDealloc);

    // Function folded
    rewriter.setInsertionPointToStart(moduleOp.getBody());
    std::string fnNameFolded = op.getCallee().str() + ".folded";
    SmallVector<Type> typesFolded = {originalTypes.begin(), originalTypes.end()};
    Type indexType = rewriter.getIndexType();
    typesFolded.push_back(indexType);
    FunctionType fnFoldedType = FunctionType::get(ctx, /*inputs=*/
                                                  typesFolded,
                                                  /*outputs=*/fnOp.getResultTypes());

    func::FuncOp fnFoldedOp = rewriter.create<func::FuncOp>(loc, fnNameFolded, fnFoldedType);
    fnFoldedOp.setPrivate();

    Block *foldedBloc = fnFoldedOp.addEntryBlock();
    rewriter.setInsertionPointToStart(foldedBloc);
    TypedAttr nQubitsAttr = rewriter.getI64IntegerAttr(nQubitsInt);
    Value nQubitsValue = rewriter.create<arith::ConstantOp>(loc, nQubitsAttr);
    Value allocQreg = rewriter.create<func::CallOp>(loc, fnAlloc, nQubitsValue).getResult(0);

    Value c0 = rewriter.create<index::ConstantOp>(loc, 0);
    Value c1 = rewriter.create<index::ConstantOp>(loc, 1);
    int64_t sizeArgs = fnFoldedOp.getArguments().size();
    Value size = fnFoldedOp.getArgument(sizeArgs - 1);
    // Add scf for loop
    Value loopedQreg =
        rewriter
            .create<scf::ForOp>(
                loc, c0, size, c1, /*iterArgsInit=*/allocQreg,
                [&](OpBuilder &builder, Location loc, Value i, ValueRange iterArgs) {
                    Value qreg = iterArgs.front();
                    std::vector<Value> argsAndReg = {fnFoldedOp.getArguments().begin(),
                                                     fnFoldedOp.getArguments().end()};
                    argsAndReg.pop_back();
                    argsAndReg.push_back(qreg);
                    Value funcQreg =
                        builder.create<func::CallOp>(loc, fnWithoutMeasurementsOp, argsAndReg)
                            .getResult(0);

                    std::vector<Value> argsAndRegAdjoint = {fnFoldedOp.getArguments().begin(),
                                                            fnFoldedOp.getArguments().end()};
                    argsAndRegAdjoint.pop_back();
                    argsAndRegAdjoint.push_back(funcQreg);
                    Value funcQregAdjoint =
                        builder
                            .create<func::CallOp>(loc, fnWithoutMeasurementsOp, argsAndRegAdjoint)
                            .getResult(0);
                    builder.create<scf::YieldOp>(loc, funcQregAdjoint);
                })
            .getResult(0);
    std::vector<Value> argsAndRegMeasurement = {fnFoldedOp.getArguments().begin(),
                                                fnFoldedOp.getArguments().end()};
    argsAndRegMeasurement.pop_back();
    argsAndRegMeasurement.push_back(loopedQreg);
    ValueRange funcFolded =
        rewriter.create<func::CallOp>(loc, fnWithQregArg, argsAndRegMeasurement).getResults();
    rewriter.create<func::ReturnOp>(loc, funcFolded);
    return SymbolRefAttr::get(ctx, fnNameFolded);
}

} // namespace mitigation
} // namespace catalyst

// Block *currentBlock = rewriter.getInsertionBlock();
// rewriter.atBlockEnd(currentBlock);
// Readd the quantum part
// for (Operation &op : originalBlock) {

//     auto newOp = rewriter.clone(op, regMap);
//     originalBlock.push_back(newOp);
// }
// static void exploreTreeAndStoreLeafValues(Operation* op, Value& leafValue);
//     static std::map<catalyst::quantum::ExtractOp, Value> createQubitMap(std::vector<Operation *>
//     extractOps);
// std::map<catalyst::quantum::ExtractOp, Value>
// ZneLowering::createQubitMap(std::vector<Operation *> extractOps)
// {
//     std::map<catalyst::quantum::ExtractOp, Value> qubitMap;
//     for (Operation *op : extractOps) {
//         Value qubit;
//         exploreTreeAndStoreLeafValues(op, qubit);
//         quantum::ExtractOp extractOp = dyn_cast<quantum::ExtractOp>(op);
//         qubitMap.insert(std::make_pair(extractOp, qubit));
//     }
//     return qubitMap;
// }
// Recursive search in the tree
// void ZneLowering::exploreTreeAndStoreLeafValues(Operation *op, Value &leafValue)
// {
//     op->dump();
//     if (op->getUsers().empty()) {
//         leafValue = op->getResult(0);
//     }
//     else {
//         for (Operation *user : op->getUsers()) {
//             exploreTreeAndStoreLeafValues(user, leafValue);
//         }
//     }
// }

// func.func @simpleCircuit.split(%arg0: tensor<3xf64>, %arg1 !quantum.reg) -> f64 attributes
// {qnode} {
//     %c0 = arith.constant 0 : index
//     %c1 = arith.constant 1 : index
//     %c2 = arith.constant 2 : index
//     %f0 = tensor.extract %arg0[%c0] : tensor<3xf64>
//     %f1 = tensor.extract %arg0[%c1] : tensor<3xf64>
//     %f2 = tensor.extract %arg0[%c2] : tensor<3xf64>

//     %idx = arith.index_cast %c0 : index to i64

//     %q_0 = quantum.extract %arg1[%idx] : !quantum.reg -> !quantum.bit
//     %q_1 = quantum.custom "h"() %q_0 : !quantum.bit
//     %q_2 = quantum.custom "rz"(%f0) %q_1 : !quantum.bit
//     %q_3 = quantum.custom "u3"(%f0, %f1, %f2) %q_2 : !quantum.bit

//     %12 = quantum.insert %r[ 0], %q_3 : !quantum.reg, !quantum.bit

//     func.return %12 : !quantum.reg
// }

// func.func @simpleCircuit.measurements(%arg0: tensor<3xf64>, %arg1 !quantum.reg, %arg2 index) ->
// f64 attributes {qnode} {
//     %q_0 = quantum.extract %arg1[%arg2] : !quantum.reg -> !quantum.bit
//     %obs = quantum.namedobs %q_3[PauliX] : !quantum.obs
//     %expval = quantum.expval %obs : f64
//     func.return %expval : f64
// }

// IRMapping ZneLowering::createQubitMap(Block &block)
// {

//     std::vector<quantum::AllocOp> allocOps;
//     block.walk([&](quantum::AllocOp op) { allocOps.push_back(op); });

//     // TODO add failure if more than one
//     Value firstReg = allocOps.front().getResult();

//     std::vector<quantum::InsertOp> insertOps;
//     block.walk([&](quantum::InsertOp op) { insertOps.push_back(op); });
//     Value lastReg = insertOps.back().getResult();

//     IRMapping regMap;
//     regMap.map(firstReg, lastReg);

//     return regMap;
// }

// std::vector<Operation *> ZneLowering::removeAndStoreQuantumMeasurements(Block &block)
// {
//     // Delete measurement operations.
//     std::deque<Operation *> opsToDelete;
//     block.walk([&](quantum::NamedObsOp op) { opsToDelete.push_back(op); });

//     std::vector<Operation *> ops;
//     std::copy(opsToDelete.begin(), opsToDelete.end(), std::back_inserter(ops));

//     SmallPtrSet<Operation *, 4> visited{opsToDelete.begin(), opsToDelete.end()};

//     while (!opsToDelete.empty()) {
//         Operation *currentOp = opsToDelete.front();
//         opsToDelete.pop_front();
//         currentOp->dropAllReferences();
//         for (Operation *user : currentOp->getUsers()) {
//             if (!visited.contains(user)) {
//                 visited.insert(user);
//                 opsToDelete.push_back(user);
//             }
//         }
//         if (currentOp->use_empty()) {
//             currentOp->remove();
//         }
//         else {
//             opsToDelete.push_back(currentOp);
//         }
//     }

//     return ops;
// }