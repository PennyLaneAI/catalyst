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

#include <algorithm>
#include <sstream>
#include <vector>

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/IRMapping.h"

#include "Gradient/IR/GradientOps.h"
#include "Gradient/Utils/DifferentialQNode.h"
#include "Gradient/Utils/GradientShape.h"
#include "Quantum/IR/QuantumOps.h"

#include "Adjoint.hpp"

namespace catalyst {
namespace gradient {

LogicalResult AdjointLowering::matchAndRewrite(func::FuncOp op, PatternRewriter &rewriter) const
{
    if (!(getQNodeDiffMethod(op) == "adjoint" && requiresCustomGradient(op))) {
        return failure();
    }

    Location loc = op.getLoc();
    rewriter.setInsertionPointAfter(op);

    // Generate the quantum gradient function, relying on the backend to implement the adjoint
    // computation.
    func::FuncOp qGradFn = genQGradFunction(rewriter, loc, op);

    // Register the quantum gradient on the quantum-only split-out QNode.
    registerCustomGradient(op, FlatSymbolRefAttr::get(qGradFn));
    return success();
}

func::FuncOp AdjointLowering::discardAndReturnReg(PatternRewriter &rewriter, Location loc,
                                                  func::FuncOp callee)
{
    // TODO: we do not support multiple return statements (which can happen for unstructured
    // control flow), i.e. our gradient functions will have just one block.
    assert(callee.getBody().hasOneBlock() &&
           "Gradients with unstructured control flow are not supported");

    // Since the return value is guaranteed to be discarded, then let's change the return type
    // to be only the quantum register and the expval.
    //
    // We also need to return the expval to avoid dead code elimination downstream from
    // removing the expval op in the body.
    // TODO: we only support grad on expval op for now
    SmallVector<quantum::DeallocOp> deallocs;
    SmallVector<quantum::ExpvalOp> expvalOps;
    SmallVector<quantum::DeviceReleaseOp> deviceReleaseOps;
    for (Operation &op : callee.getBody().getOps()) {
        if (isa<quantum::DeallocOp>(op)) {
            deallocs.push_back(cast<quantum::DeallocOp>(op));
            continue;
        }
        else if (isa<quantum::MeasurementProcess>(op)) {
            if (isa<quantum::ExpvalOp>(op)) {
                expvalOps.push_back(cast<quantum::ExpvalOp>(op));
                continue;
            }
            else {
                callee.emitOpError() << "Adjoint gradient is only supported on expval measurements";
                return callee;
            }
        }
        else if (isa<quantum::DeviceReleaseOp>(op)) {
            deviceReleaseOps.push_back(cast<quantum::DeviceReleaseOp>(op));
        }
    }

    // If there are no deallocs leave early then this transformation
    // is invalid. This is because the caller will expect a quantum register
    // as a return value.
    // Also, let's handle the simple case that is guaranteed at the moment.
    size_t numDeallocs = deallocs.size();
    if (numDeallocs != 1) {
        callee.emitOpError() << "Invalid number of quantum registers: " << numDeallocs;
        return callee;
    }

    size_t numDeviceReleases = deviceReleaseOps.size();
    if (numDeviceReleases > 1) {
        callee.emitOpError() << "Invalid number of device release ops: " << numDeviceReleases;
        return callee;
    }

    // Create clone, return type is qreg and float for the expvals
    std::string fnName = callee.getName().str() + ".nodealloc";
    Type qregType = quantum::QuregType::get(rewriter.getContext());
    Type f64Type = rewriter.getF64Type();
    SmallVector<Type> retTypes{qregType};
    std::for_each(expvalOps.begin(), expvalOps.end(),
                  [&](const quantum::ExpvalOp &) { retTypes.push_back(f64Type); });
    FunctionType fnType = rewriter.getFunctionType(callee.getArgumentTypes(), retTypes);
    StringAttr visibility = rewriter.getStringAttr("private");

    func::FuncOp unallocFn =
        SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(callee, rewriter.getStringAttr(fnName));

    if (!unallocFn) {
        PatternRewriter::InsertionGuard insertGuard(rewriter);
        rewriter.setInsertionPointAfter(callee);
        unallocFn =
            rewriter.create<func::FuncOp>(loc, fnName, fnType, visibility, nullptr, nullptr);

        // Clone the body.
        IRMapping mapper;
        rewriter.cloneRegionBefore(callee.getBody(), unallocFn.getBody(), unallocFn.end(), mapper);
        rewriter.setInsertionPointToStart(&unallocFn.getBody().front());

        // Let's return the qreg+expval and erase the device release.
        // Fine for now: only one block in body so only one dealloc and one expval
        SmallVector<Value> returnVals{mapper.lookup(deallocs[0])->getOperand(0)};
        std::for_each(expvalOps.begin(), expvalOps.end(), [&](const quantum::ExpvalOp &expval) {
            returnVals.push_back(mapper.lookup(expval));
        });

        // Create the return
        // Again, assume just one block for now
        Operation *returnOp = unallocFn.getBody().front().getTerminator();
        assert(isa<func::ReturnOp>(returnOp) && "adjoint block must terminate with return op");
        returnOp->setOperands(returnVals);

        // Erase the device release.
        for (auto op : deviceReleaseOps) {
            rewriter.eraseOp(mapper.lookup(op));
        }

        // Let's erase the deallocation.
        rewriter.eraseOp(mapper.lookup(deallocs[0]));
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

        qGradFn = rewriter.create<func::FuncOp>(loc, fnName, fnType, visibility, nullptr, nullptr);
        rewriter.setInsertionPointToStart(qGradFn.addEntryBlock());

        AdjointOp qGradOp = rewriter.create<AdjointOp>(
            loc, computeQGradTypes(callee), SymbolRefAttr::get(unallocFn),
            qGradFn.getArguments().back(), qGradFn.getArguments().drop_back(), ValueRange{});

        rewriter.create<func::ReturnOp>(loc, qGradOp.getResults());
    }

    return qGradFn;
}

} // namespace gradient
} // namespace catalyst
