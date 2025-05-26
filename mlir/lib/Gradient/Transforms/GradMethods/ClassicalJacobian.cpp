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

#include "ClassicalJacobian.hpp"

#include <deque>
#include <string>
#include <vector>

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Index/IR/IndexOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"

#include "Catalyst/Utils/StaticAllocas.h"
#include "Quantum/IR/QuantumInterfaces.h"
#include "Quantum/IR/QuantumOps.h"
#include "Quantum/Utils/RemoveQuantum.h"

namespace catalyst {
namespace gradient {

/// Generate a new mlir function that counts the (runtime) number of gate parameters.
///
/// This enables other functions to allocate memory for vectors of gate
/// parameters without having to deal with dynamic memory management. The function works
/// by eliminating all quantum code and running the classical preprocessing,
/// but instead of storing gate parameters it merely counts them.
/// The impact on execution time is expected to be non-dominant, as the classical pre-processing is
/// already run multiple times, such as to differentiate the ArgMap and on every execution of
/// quantum function for the parameter-shift method. However, if this is inefficient in certain
/// use-cases, other approaches can be employed.
///
func::FuncOp genParamCountFunction(PatternRewriter &rewriter, Location loc, func::FuncOp callee)
{
    // Define the properties of the gate parameter counting version of the function to be
    // differentiated.
    std::string fnName = callee.getSymName().str() + ".pcount";
    FunctionType fnType =
        rewriter.getFunctionType(callee.getArgumentTypes(), rewriter.getIndexType());
    StringAttr visibility = rewriter.getStringAttr("private");

    func::FuncOp paramCountFn =
        SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(callee, rewriter.getStringAttr(fnName));
    if (!paramCountFn) {
        // First copy the original function as is, then we can replace all quantum ops by counting
        // their gate parameters instead.
        rewriter.setInsertionPointAfter(callee);
        paramCountFn =
            rewriter.create<func::FuncOp>(loc, fnName, fnType, visibility, nullptr, nullptr);
        rewriter.cloneRegionBefore(callee.getBody(), paramCountFn.getBody(), paramCountFn.end());

        PatternRewriter::InsertionGuard insertGuard(rewriter);
        rewriter.setInsertionPointToStart(&paramCountFn.getBody().front());

        // Store the counter in memory since we don't want to deal with returning the SSA value
        // for updated parameter counts from arbitrary regions/ops.
        MemRefType paramCountType = MemRefType::get({}, rewriter.getIndexType());
        Value paramCountBuffer = getStaticMemrefAlloca(loc, rewriter, paramCountType);
        Value cZero = rewriter.create<index::ConstantOp>(loc, 0);
        rewriter.create<memref::StoreOp>(loc, cZero, paramCountBuffer);

        paramCountFn.walk([&](Operation *op) {
            // For each quantum gate add the number of parameters to the counter.
            if (auto gate = dyn_cast<quantum::DifferentiableGate>(op)) {
                PatternRewriter::InsertionGuard insertGuard(rewriter);
                rewriter.setInsertionPoint(gate);

                ValueRange diffParams = gate.getDiffParams();
                if (!diffParams.empty()) {
                    Value currCount = rewriter.create<memref::LoadOp>(loc, paramCountBuffer);
                    Value numParams = rewriter.create<index::ConstantOp>(loc, diffParams.size());
                    Value newCount = rewriter.create<index::AddOp>(loc, currCount, numParams);
                    rewriter.create<memref::StoreOp>(loc, newCount, paramCountBuffer);
                }

                rewriter.replaceOp(gate, gate.getQubitOperands());
            }
            // Any other gates or quantum instructions can also be stripped.
            // Measurements are handled separately.
            else if (isa<quantum::DeviceInitOp>(op)) {
                rewriter.eraseOp(op);
            }
            else if (auto gate = dyn_cast<quantum::QuantumOperation>(op)) {
                rewriter.replaceOp(op, gate.getQubitOperands());
            }
            else if (auto region = dyn_cast<quantum::QuantumRegion>(op)) {
                rewriter.replaceOp(op, region.getRegisterOperand());
            }
            else if (isa<quantum::DeallocOp>(op)) {
                rewriter.eraseOp(op);
            }
            else if (isa<quantum::DeviceReleaseOp>(op)) {
                rewriter.eraseOp(op);
            }

            // Replace any return statements from the original function with the parameter count.
            else if (isa<func::ReturnOp>(op)) {
                PatternRewriter::InsertionGuard insertGuard(rewriter);
                rewriter.setInsertionPoint(op);

                Value paramCount = rewriter.create<memref::LoadOp>(loc, paramCountBuffer);
                op->setOperands(paramCount);
            }
        });

        quantum::removeQuantumMeasurements(paramCountFn, rewriter);
        paramCountFn->setAttr("QuantumFree", rewriter.getUnitAttr());
    }

    return paramCountFn;
}

func::FuncOp genSplitPreprocessed(PatternRewriter &rewriter, Location loc, func::FuncOp qnode,
                                  func::FuncOp qnodeQuantum)
{
    // Define the properties of the split-out preprocessing-only QNode.
    std::string fnName = qnode.getSymName().str() + ".preprocess";
    SmallVector<Type> fnArgTypes(qnode.getArgumentTypes());
    auto paramsBufferType = MemRefType::get({ShapedType::kDynamic}, rewriter.getF64Type());
    fnArgTypes.push_back(rewriter.getIndexType()); // parameter count
    FunctionType fnType = rewriter.getFunctionType(fnArgTypes, qnode.getResultTypes());
    StringAttr visibility = rewriter.getStringAttr("private");

    func::FuncOp splitFn =
        SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(qnode, rewriter.getStringAttr(fnName));
    if (!splitFn) {
        // First copy the original function as is, then we can replace all quantum ops by collecting
        // their gate parameters in a memory buffer instead. This buffer is passed into a modified
        // qnodeQuantum.
        splitFn = rewriter.create<func::FuncOp>(loc, fnName, fnType, visibility, nullptr, nullptr);
        rewriter.cloneRegionBefore(qnode.getBody(), splitFn.getBody(), splitFn.end());
        Block &argMapBlock = splitFn.getFunctionBody().front();
        SmallVector<Value> qnodeQuantumArgs{argMapBlock.getArguments()};

        Value paramCount = argMapBlock.addArgument(rewriter.getIndexType(), loc);
        PatternRewriter::InsertionGuard insertGuard(rewriter);
        rewriter.setInsertionPointToStart(&splitFn.getBody().front());
        Value paramsBuffer = rewriter.create<memref::AllocOp>(loc, paramsBufferType, paramCount);
        Value paramsTensor = rewriter.create<bufferization::ToTensorOp>(loc, paramsBuffer, true);

        qnodeQuantumArgs.push_back(paramsTensor);
        MemRefType paramsProcessedType = MemRefType::get({}, rewriter.getIndexType());
        Value paramsProcessed = getStaticMemrefAlloca(loc, rewriter, paramsProcessedType);
        Value cZero = rewriter.create<index::ConstantOp>(loc, 0);
        rewriter.create<memref::StoreOp>(loc, cZero, paramsProcessed);
        Value cOne = rewriter.create<index::ConstantOp>(loc, 1);

        splitFn.walk([&](Operation *op) {
            // Insert gate parameters into the params buffer.
            if (auto gate = dyn_cast<quantum::DifferentiableGate>(op)) {
                PatternRewriter::InsertionGuard insertGuard(rewriter);
                rewriter.setInsertionPoint(gate);

                ValueRange diffParams = gate.getDiffParams();
                if (!diffParams.empty()) {
                    Value paramIdx = rewriter.create<memref::LoadOp>(loc, paramsProcessed);
                    for (auto param : diffParams) {
                        rewriter.create<memref::StoreOp>(loc, param, paramsBuffer, paramIdx);
                        paramIdx = rewriter.create<index::AddOp>(loc, paramIdx, cOne);
                    }
                    rewriter.create<memref::StoreOp>(loc, paramIdx, paramsProcessed);
                }

                rewriter.replaceOp(op, gate.getQubitOperands());
            }
            // Any other gates or quantum instructions also need to be stripped.
            // Measurements are handled separately.
            else if (isa<quantum::DeviceInitOp>(op)) {
                rewriter.eraseOp(op);
            }
            else if (auto gate = dyn_cast<quantum::QuantumOperation>(op)) {
                rewriter.replaceOp(op, gate.getQubitOperands());
            }
            else if (auto region = dyn_cast<quantum::QuantumRegion>(op)) {
                rewriter.replaceOp(op, region.getRegisterOperand());
            }
            else if (isa<quantum::DeallocOp>(op)) {
                rewriter.eraseOp(op);
            }
            else if (isa<quantum::DeviceReleaseOp>(op)) {
                rewriter.eraseOp(op);
            }

            // Return ops should be preceded with calls to the modified QNode
            else if (auto returnOp = dyn_cast<func::ReturnOp>(op)) {
                PatternRewriter::InsertionGuard insertionGuard(rewriter);
                rewriter.setInsertionPoint(returnOp);
                auto modifiedCall =
                    rewriter.create<func::CallOp>(loc, qnodeQuantum, qnodeQuantumArgs);

                returnOp.getOperandsMutable().assign(modifiedCall.getResults());
            }
        });

        quantum::removeQuantumMeasurements(splitFn, rewriter);
        splitFn->setAttr("QuantumFree", rewriter.getUnitAttr());
    }

    return splitFn;
}

} // namespace gradient
} // namespace catalyst
