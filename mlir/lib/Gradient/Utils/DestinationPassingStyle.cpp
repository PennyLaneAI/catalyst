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

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"

#include "Gradient/Utils/DestinationPassingStyle.h"

using namespace mlir;

void catalyst::convertToDestinationPassingStyle(func::FuncOp callee, OpBuilder &builder)
{
    if (callee.getNumResults() == 0) {
        // Callee is already in destination-passing style
        return;
    }

    MLIRContext *ctx = callee.getContext();
    SmallVector<Type> memRefReturnTypes;
    SmallVector<unsigned> outputIndices;
    SmallVector<Type> nonMemRefReturns;

    for (const auto &[idx, resultType] : llvm::enumerate(callee.getResultTypes())) {
        if (isa<MemRefType>(resultType)) {
            memRefReturnTypes.push_back(resultType);
            outputIndices.push_back(idx);
        }
        else {
            nonMemRefReturns.push_back(resultType);
        }
    }

    SmallVector<Type> dpsArgumentTypes{callee.getArgumentTypes()};
    dpsArgumentTypes.append(memRefReturnTypes);
    auto dpsFunctionType = FunctionType::get(ctx, dpsArgumentTypes, nonMemRefReturns);

    if (callee.isDeclaration()) {
        // If the function does not have a body, we are done after modifying the function type.
        callee.setFunctionType(dpsFunctionType);
        return;
    }

    // Insert the new output arguments to the function.
    unsigned dpsOutputIdx = callee.getNumArguments();
    SmallVector<unsigned> argIndices(/*size=*/memRefReturnTypes.size(),
                                     /*values=*/dpsOutputIdx);
    SmallVector<DictionaryAttr> argAttrs{memRefReturnTypes.size()};
    SmallVector<Location> argLocs{memRefReturnTypes.size(), callee.getLoc()};

    // insertArguments modifies the function type, so we need to update the function type *after*
    // inserting the arguments.
    callee.insertArguments(argIndices, memRefReturnTypes, argAttrs, argLocs);
    callee.setFunctionType(dpsFunctionType);

    // Update return sites to copy over the memref that would have been returned to the output.
    callee.walk([&](func::ReturnOp returnOp) {
        OpBuilder::InsertionGuard insertionGuard(builder);
        builder.setInsertionPoint(returnOp);
        SmallVector<Value> nonMemRefReturns;
        size_t idx = 0;
        for (Value operand : returnOp.getOperands()) {
            if (isa<MemRefType>(operand.getType())) {
                BlockArgument output = callee.getArgument(idx + dpsOutputIdx);
                // We need a linalg.copy instead of a memref.copy here because it provides better
                // type information at the LLVM level for Enzyme.
                builder.create<linalg::CopyOp>(returnOp.getLoc(), operand, output);
                idx++;
            }
            else {
                nonMemRefReturns.push_back(operand);
            }
        }
        returnOp.getOperandsMutable().assign(nonMemRefReturns);
    });
}
