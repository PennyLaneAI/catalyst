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
#include "ClassicalJacobian.hpp"
#include "HybridGradient.hpp"

#include <algorithm>
#include <map>
#include <sstream>
#include <vector>

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"

#include "Gradient/Utils/GradientShape.h"
#include "Quantum/IR/QuantumOps.h"

namespace catalyst {
namespace gradient {

LogicalResult AdjointLowering::match(GradOp op) const
{
    if (op.getMethod() == "adj")
        return success();

    return failure();
}

void AdjointLowering::rewrite(GradOp op, PatternRewriter &rewriter) const
{
    Location loc = op.getLoc();
    func::FuncOp callee =
        SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(op, op.getCalleeAttr());
    rewriter.setInsertionPointAfter(callee);

    // In order to allocate memory for various tensors relating to the number of gate parameters
    // at runtime we run a function that merely counts up for each gate parameter encountered.
    func::FuncOp paramCountFn = genParamCountFunction(rewriter, loc, callee);

    // Generate the classical argument map from function arguments to gate parameters. This
    // function will be differentiated to produce the classical jacobian.
    func::FuncOp argMapFn = genArgMapFunction(rewriter, loc, callee);

    // Generate the quantum gradient function, relying on the backend to implement the adjoint
    // computation.
    func::FuncOp qGradFn = genQGradFunction(rewriter, loc, callee);

    // Generate the full gradient function, computing the partial derivates with respect to the
    // original function arguments from the classical Jacobian and quantum gradient.
    func::FuncOp fullGradFn =
        genFullGradFunction(rewriter, loc, op, paramCountFn, argMapFn, qGradFn, "adj");

    rewriter.setInsertionPoint(op);
    rewriter.replaceOpWithNewOp<func::CallOp>(op, fullGradFn, op.getArgOperands());
}

func::FuncOp AdjointLowering::discardAndReturnReg(PatternRewriter &rewriter, Location loc,
                                                  func::FuncOp callee)
{

    std::vector<quantum::DeallocOp> deallocs;
    callee.walk([&](quantum::DeallocOp dealloc) { deallocs.push_back(dealloc); });

    // If there are no deallocs leave early then this transformation
    // is invalid. This is because the caller will expect a quantum register
    // as a return value.
    // Also, let's handle the simple case that is guaranteed at the moment.
    bool invalid_transformation = deallocs.size() != 1;
    if (invalid_transformation) {
        callee.emitOpError() << "Invalid number of quantum registers: " << deallocs.size();
        return callee;
    }

    quantum::DeallocOp deallocToReturn = deallocs.front();
    std::string fnName = callee.getName().str() + ".nodealloc";
    std::vector<Type> fnArgTypes = callee.getArgumentTypes().vec();
    Value qreg = deallocToReturn.getQreg();
    Type qregType = qreg.getType();

    // Since the return value is guaranteed to be discarded, then let's change the return type
    // to be only the quantum register.
    std::vector<Type> resultTypes;
    resultTypes.push_back(qregType);
    FunctionType fnType = rewriter.getFunctionType(fnArgTypes, resultTypes);
    StringAttr visibility = rewriter.getStringAttr("private");

    func::FuncOp unallocFn =
        SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(callee, rewriter.getStringAttr(fnName));

    if (!unallocFn) {
        PatternRewriter::InsertionGuard insertGuard(rewriter);
        rewriter.setInsertionPointAfter(callee);
        unallocFn = rewriter.create<func::FuncOp>(loc, fnName, fnType, visibility);
        // clone the body.
        rewriter.cloneRegionBefore(callee.getBody(), unallocFn.getBody(), unallocFn.end());
        rewriter.setInsertionPointToStart(&unallocFn.getBody().front());
        // Let's capture the qreg
        std::vector<quantum::DeallocOp> localDeallocs;
        unallocFn.walk([&](quantum::DeallocOp deallocOp) { localDeallocs.push_back(deallocOp); });
        // Let's return the qreg.
        unallocFn.walk([&](func::ReturnOp returnOp) {
            PatternRewriter::InsertionGuard insertGuard(rewriter);
            rewriter.setInsertionPoint(returnOp);
            returnOp->setOperands(localDeallocs.front().getOperand());
        });
        // Let's erase the deallocation
        unallocFn.walk([&](quantum::DeallocOp deallocOp) { deallocOp.erase(); });
    }

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

        qGradFn = rewriter.create<func::FuncOp>(loc, fnName, fnType, visibility);
        rewriter.setInsertionPointToStart(qGradFn.addEntryBlock());

        AdjointOp qGradOp = rewriter.create<AdjointOp>(
            loc, computeQGradTypes(callee), unallocFn.getName(),
            qGradFn.getArguments().drop_back()); // device has no use for the gradient size value

        rewriter.create<func::ReturnOp>(loc, qGradOp.getResults());
    }

    return qGradFn;
}

} // namespace gradient
} // namespace catalyst
