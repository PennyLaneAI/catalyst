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

#include "Adjoint.hpp"

#include <algorithm>
#include <sstream>
#include <vector>

#include "mlir/Dialect/Func/IR/FuncOps.h"

#include "Gradient/IR/GradientOps.h"
#include "Gradient/Utils/DifferentialQNode.h"
#include "Gradient/Utils/GradientShape.h"
#include "Quantum/IR/QuantumOps.h"

namespace catalyst {
namespace gradient {

LogicalResult AdjointLowering::match(func::FuncOp op) const
{
    if (getQNodeDiffMethod(op) == "adjoint" && requiresCustomGradient(op))
        return success();

    return failure();
}

void AdjointLowering::rewrite(func::FuncOp op, PatternRewriter &rewriter) const
{
    Location loc = op.getLoc();
    rewriter.setInsertionPointAfter(op);

    // Generate the quantum gradient function, relying on the backend to implement the adjoint
    // computation.
    func::FuncOp qGradFn = genQGradFunction(rewriter, loc, op);

    // Register the quantum gradient on the quantum-only split-out QNode.
    registerCustomGradient(op, FlatSymbolRefAttr::get(qGradFn));
}

func::FuncOp AdjointLowering::discardAndReturnReg(PatternRewriter &rewriter, Location loc,
                                                  func::FuncOp callee)
{
    SmallVector<quantum::DeallocOp> deallocs;
    for (auto op : callee.getOps<quantum::DeallocOp>()) {
        deallocs.push_back(op);
    };

    // If there are no deallocs leave early then this transformation
    // is invalid. This is because the caller will expect a quantum register
    // as a return value.
    // Also, let's handle the simple case that is guaranteed at the moment.
    size_t numDeallocs = deallocs.size();
    if (numDeallocs != 1) {
        callee.emitOpError() << "Invalid number of quantum registers: " << numDeallocs;
        return callee;
    }

    // Since the return value is guaranteed to be discarded, then let's change the return type
    // to be only the quantum register.
    std::string fnName = callee.getName().str() + ".nodealloc";
    Type qregType = quantum::QuregType::get(rewriter.getContext());
    FunctionType fnType = rewriter.getFunctionType(callee.getArgumentTypes(), qregType);
    StringAttr visibility = rewriter.getStringAttr("private");

    func::FuncOp unallocFn =
        SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(callee, rewriter.getStringAttr(fnName));

    if (!unallocFn) {
        PatternRewriter::InsertionGuard insertGuard(rewriter);
        rewriter.setInsertionPointAfter(callee);
        unallocFn =
            rewriter.create<func::FuncOp>(loc, fnName, fnType, visibility, nullptr, nullptr);
        // clone the body.
        rewriter.cloneRegionBefore(callee.getBody(), unallocFn.getBody(), unallocFn.end());
        rewriter.setInsertionPointToStart(&unallocFn.getBody().front());

        // Let's capture the qreg.
        quantum::DeallocOp localDealloc = *unallocFn.getOps<quantum::DeallocOp>().begin();

        // Let's return the qreg.
        unallocFn.walk(
            [&](func::ReturnOp returnOp) { returnOp->setOperands(localDealloc.getOperand()); });

        // Let's erase the deallocation.
        rewriter.eraseOp(localDealloc);
    }

    unallocFn.walk([&](Operation *op) {
        if (isa<quantum::DeviceReleaseOp>(op)) {
            rewriter.eraseOp(op);
        }
    });

    return unallocFn;
}

func::FuncOp AdjointLowering::genQGradFunction(PatternRewriter &rewriter, Location loc,
                                               func::FuncOp callee)
{

    func::FuncOp unallocFn = discardAndReturnReg(rewriter, loc, callee);

    std::string fnName = callee.getName().str() + ".adjoint";
    std::vector<Type> fnArgTypes = callee.getArgumentTypes().vec();
    Type gradientSizeType = rewriter.getIndexType();
    fnArgTypes.push_back(gradientSizeType);
    FunctionType fnType = rewriter.getFunctionType(fnArgTypes, computeQGradTypes(callee));
    StringAttr visibility = rewriter.getStringAttr("private");

    func::FuncOp qGradFn =
        SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(callee, rewriter.getStringAttr(fnName));
    if (!qGradFn) {
        PatternRewriter::InsertionGuard insertGuard(rewriter);
        rewriter.setInsertionPointAfter(callee);

        qGradFn = rewriter.create<func::FuncOp>(loc, fnName, fnType, visibility, nullptr, nullptr);
        rewriter.setInsertionPointToStart(qGradFn.addEntryBlock());

        AdjointOp qGradOp = rewriter.create<AdjointOp>(
            loc, computeQGradTypes(callee), unallocFn.getName(), qGradFn.getArguments().back(),
            qGradFn.getArguments().drop_back(), ValueRange{});

        rewriter.create<func::ReturnOp>(loc, qGradOp.getResults());
    }

    return qGradFn;
}

} // namespace gradient
} // namespace catalyst
