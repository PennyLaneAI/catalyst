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
    SmallVector<Value> memRefReturns;
    SmallVector<unsigned> outputIndices;
    SmallVector<Type> nonMemRefReturns;
    callee.walk([&](func::ReturnOp returnOp) {
        // This is the first return op we've seen.
        if (memRefReturns.empty()) {
            for (const auto &[idx, operand] : llvm::enumerate(returnOp.getOperands())) {
                if (isa<MemRefType>(operand.getType())) {
                    memRefReturns.push_back(operand);
                    outputIndices.push_back(idx);
                }
                else {
                    nonMemRefReturns.push_back(operand.getType());
                }
            }
            return WalkResult::interrupt();
        }
        return WalkResult::advance();
    });

    // Insert the new output arguments to the function.
    unsigned dpsOutputIdx = callee.getNumArguments();
    SmallVector<unsigned> argIndices(/*size=*/memRefReturns.size(),
                                     /*values=*/dpsOutputIdx);
    SmallVector<Type> memRefTypes{memRefReturns.size()};
    SmallVector<DictionaryAttr> argAttrs{memRefReturns.size()};
    SmallVector<Location> argLocs{memRefReturns.size(), UnknownLoc::get(ctx)};

    llvm::transform(memRefReturns, memRefTypes.begin(),
                    [](Value memRef) { return memRef.getType(); });
    llvm::transform(memRefReturns, argLocs.begin(), [](Value memRef) { return memRef.getLoc(); });

    callee.insertArguments(argIndices, memRefTypes, argAttrs, argLocs);
    callee.setFunctionType(FunctionType::get(ctx, callee.getArgumentTypes(), nonMemRefReturns));

    // Update return sites to copy over the memref that would have been returned to the output.
    callee.walk([&](func::ReturnOp returnOp) {
        OpBuilder::InsertionGuard insertionGuard(builder);
        builder.setInsertionPoint(returnOp);
        SmallVector<Value> nonMemRefReturns;
        size_t idx = 0;
        for (Value operand : returnOp.getOperands()) {
            if (isa<MemRefType>(operand.getType())) {
                BlockArgument output = callee.getArgument(idx + dpsOutputIdx);
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
