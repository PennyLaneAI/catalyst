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
#include <sstream>
#include <vector>

#include "Mitigation/IR/MitigationOps.h"
#include "Quantum/IR/QuantumOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Index/IR/IndexOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

#include "Quantum/Utils/RemoveQuantumMeasurements.h"

namespace catalyst {
namespace mitigation {

LogicalResult ZneLowering::match(mitigation::ZneOp op) const { return success(); }

void ZneLowering::rewrite(mitigation::ZneOp op, PatternRewriter &rewriter) const
{
    Location loc = op.getLoc();

    auto scalarFactor = op.getScalarFactors();
    RankedTensorType scalarFactorType = scalarFactor.getType().cast<RankedTensorType>();
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
    func::FuncOp circuit =
        SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(op, op.getCalleeAttr());
    auto resultsValues = rewriter.create<func::CallOp>(loc, circuit, op.getArgs());
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

    // Function Name
    std::string fnName = op.getCallee().str() + ".folded";

    // Get callee arg types
    auto originalTypes = op.getArgs().getTypes();
    SmallVector<Type> typesFolded = {originalTypes.begin(), originalTypes.end()};
    typesFolded.push_back(scalarType);

    // Get the scalar factor type

    // Get the arguments and outputs types from the original block
    FunctionType updateFnType = FunctionType::get(ctx, /*inputs=*/
                                                  typesFolded,
                                                  /*outputs=*/op.getResultTypes());

    func::FuncOp updateFnOp = rewriter.create<func::FuncOp>(loc, fnName, updateFnType);
    updateFnOp.setPrivate();
    rewriter.cloneRegionBefore(fnOp.getBody(), updateFnOp.getBody(), updateFnOp.end());
    Block &argMapBlock = updateFnOp.getFunctionBody().front();
    argMapBlock.addArgument(scalarType, loc);

    // // Walk through the blocks
    // std::deque<Operation *> opsToDelete;

    // argMapBlock.walk([&](Operation *op) { opsToDelete.push_back(op); });
    // for (Operation *ptr : opsToDelete) {
    //     ptr->dump();
    //     // auto successors = ptr->getUsers();
    //     // auto succ = *successors.begin();
    //     // succ->dump();
    // }
    // // quantum::removeQuantumMeasurements(argMapBlock);

    // rewriter.create<func::ReturnOp>(loc, c0float);
    return SymbolRefAttr::get(ctx, fnName);
}
} // namespace mitigation
} // namespace catalyst
