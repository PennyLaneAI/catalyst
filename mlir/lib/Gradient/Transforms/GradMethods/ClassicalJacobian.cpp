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

#include "Quantum/IR/QuantumInterfaces.h"
#include "Quantum/IR/QuantumOps.h"
#include "Quantum/Utils/RemoveQuantumMeasurements.h"

namespace catalyst {
namespace gradient {

/// Generate a new mlir function that counts the (runtime) number of gate parameters.
///
/// This enables other functions like `genArgMapFunction` to allocate memory for vectors of gate
/// parameters without having to deal with dynamic memory management. The function works similarly
/// to `genArgMapFunction` by eliminating all quantum code and running the classical preprocessing,
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
        Value paramCountBuffer = rewriter.create<memref::AllocaOp>(loc, paramCountType);
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

                rewriter.replaceOp(gate, cast<quantum::QuantumGate>(op).getQubitOperands());
            }
            // Replace any return statements from the original function with the parameter count.
            else if (isa<func::ReturnOp>(op)) {
                PatternRewriter::InsertionGuard insertGuard(rewriter);
                rewriter.setInsertionPoint(op);

                Value paramCount = rewriter.create<memref::LoadOp>(loc, paramCountBuffer);
                op->setOperands(paramCount);
            }
            // Erase redundant device specifications.
            else if (isa<quantum::DeviceOp>(op)) {
                rewriter.eraseOp(op);
            }
        });

        quantum::removeQuantumMeasurements(paramCountFn);
    }

    return paramCountFn;
}

/// Generate a new mlir function that maps qfunc arguments to gate parameters.
///
/// This enables to extract any classical preprocessing done inside the quantum function and compute
/// its jacobian separately in order to combine it with quantum-only gradients such as the
/// parameter-shift or adjoint method.
///
func::FuncOp genArgMapFunction(PatternRewriter &rewriter, Location loc, func::FuncOp callee)
{
    // Define the properties of the classical preprocessing function.
    std::string fnName = callee.getSymName().str() + ".argmap";
    SmallVector<Type> fnArgTypes(callee.getArgumentTypes());
    fnArgTypes.push_back(rewriter.getIndexType());
    auto paramsVectorType = RankedTensorType::get({ShapedType::kDynamic}, rewriter.getF64Type());
    FunctionType fnType = rewriter.getFunctionType(fnArgTypes, paramsVectorType);
    StringAttr visibility = rewriter.getStringAttr("private");

    func::FuncOp argMapFn =
        SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(callee, rewriter.getStringAttr(fnName));
    if (!argMapFn) {
        // First copy the original function as is, then we can replace all quantum ops by collecting
        // their gate parameters in a memory buffer instead. The size of this vector is passed as an
        // input to the new function.
        argMapFn = rewriter.create<func::FuncOp>(loc, fnName, fnType, visibility, nullptr, nullptr);
        rewriter.cloneRegionBefore(callee.getBody(), argMapFn.getBody(), argMapFn.end());
        Block &argMapBlock = argMapFn.getFunctionBody().front();
        // Allocate the memory for the gate parameters collected at runtime
        Value numParams = argMapBlock.addArgument(rewriter.getIndexType(), loc);
        auto paramsBufferType =
            MemRefType::get(paramsVectorType.getShape(), paramsVectorType.getElementType());

        PatternRewriter::InsertionGuard insertGuard(rewriter);
        rewriter.setInsertionPointToStart(&argMapFn.getBody().front());

        Value paramsBuffer = rewriter.create<memref::AllocOp>(loc, paramsBufferType, numParams);
        MemRefType paramsProcessedType = MemRefType::get({}, rewriter.getIndexType());
        Value paramsProcessed = rewriter.create<memref::AllocaOp>(loc, paramsProcessedType);
        Value cZero = rewriter.create<index::ConstantOp>(loc, 0);
        rewriter.create<memref::StoreOp>(loc, cZero, paramsProcessed);
        Value cOne = rewriter.create<index::ConstantOp>(loc, 1);

        argMapFn.walk([&](Operation *op) {
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
            else if (auto returnOp = dyn_cast<func::ReturnOp>(op)) {
                PatternRewriter::InsertionGuard insertionGuard(rewriter);
                rewriter.setInsertionPoint(returnOp);
                Value paramsVector =
                    rewriter.create<bufferization::ToTensorOp>(loc, paramsVectorType, paramsBuffer);
                returnOp.getOperandsMutable().assign(paramsVector);
            }
            // Erase redundant device specifications.
            else if (isa<quantum::DeviceOp>(op)) {
                rewriter.eraseOp(op);
            }
        });

        quantum::removeQuantumMeasurements(argMapFn);
    }

    return argMapFn;
}

} // namespace gradient
} // namespace catalyst
