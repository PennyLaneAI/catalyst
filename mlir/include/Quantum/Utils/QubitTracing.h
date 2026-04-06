// Copyright 2026 Xanadu Quantum Technologies Inc.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include "llvm/ADT/STLExtras.h"

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Value.h"

#include "Quantum/IR/QuantumOps.h"

namespace catalyst::quantum {
namespace detail {

/// The implementation of `traceQubit`. Return a null `Value` if the chain cannot be resolved.
inline mlir::Value traceQubitImpl(mlir::Value qubit, llvm::function_ref<void(mlir::Value)> visitor)
{
    while (qubit) {
        visitor(qubit);

        if (auto extractOp = qubit.getDefiningOp<ExtractOp>()) {
            return extractOp.getResult();
        }

        if (auto gate = mlir::dyn_cast_or_null<QuantumGate>(qubit.getDefiningOp())) {
            auto out = mlir::cast<mlir::OpResult>(qubit);
            auto operands = gate.getQubitOperands();
            unsigned r = out.getResultNumber();
            if (r < operands.size()) {
                qubit = operands[r];
                continue;
            }
            return mlir::Value();
        }

        if (auto measureOp = mlir::dyn_cast_or_null<MeasureOp>(qubit.getDefiningOp())) {
            qubit = measureOp.getInQubit();
            continue;
        }

        if (auto forOp = mlir::dyn_cast_or_null<mlir::scf::ForOp>(qubit.getDefiningOp())) {
            unsigned r = mlir::cast<mlir::OpResult>(qubit).getResultNumber();
            if (r < forOp.getInitArgs().size()) {
                qubit = forOp.getInitArgs()[r];
                continue;
            }
            return mlir::Value();
        }

        if (auto ifOp = mlir::dyn_cast_or_null<mlir::scf::IfOp>(qubit.getDefiningOp())) {
            if (ifOp.getElseRegion().empty()) {
                return mlir::Value();
            }
            unsigned r = mlir::cast<mlir::OpResult>(qubit).getResultNumber();
            mlir::scf::YieldOp thenYield = ifOp.thenYield();
            mlir::scf::YieldOp elseYield = ifOp.elseYield();
            if (r >= thenYield.getNumOperands() || r >= elseYield.getNumOperands()) {
                return mlir::Value();
            }
            mlir::Value thenV = thenYield.getOperand(r);
            mlir::Value elseV = elseYield.getOperand(r);
            mlir::Value thenToExt = traceQubitImpl(thenV, [](mlir::Value) {});
            mlir::Value elseToExt = traceQubitImpl(elseV, [](mlir::Value) {});
            if (!thenToExt || thenToExt != elseToExt) {
                return mlir::Value();
            }
            qubit = thenV;
            continue;
        }

        if (auto whileOp = mlir::dyn_cast_or_null<mlir::scf::WhileOp>(qubit.getDefiningOp())) {
            unsigned r = mlir::cast<mlir::OpResult>(qubit).getResultNumber();
            mlir::scf::ConditionOp cond = whileOp.getConditionOp();
            mlir::ValueRange exitVals = cond.getOperands().drop_front();
            if (r >= exitVals.size()) {
                return mlir::Value();
            }
            qubit = exitVals[r];
            continue;
        }

        if (auto blockArg = mlir::dyn_cast<mlir::BlockArgument>(qubit)) {
            mlir::Block *owner = blockArg.getOwner();
            mlir::Operation *parent = owner->getParentOp();
            unsigned argNo = blockArg.getArgNumber();

            if (auto forOp = mlir::dyn_cast<mlir::scf::ForOp>(parent)) {
                unsigned numIVs = forOp.getNumInductionVars();
                if (argNo >= numIVs && argNo - numIVs < forOp.getInitArgs().size()) {
                    qubit = forOp.getInitArgs()[argNo - numIVs];
                    continue;
                }
            }
            else if (auto whileOp = mlir::dyn_cast<mlir::scf::WhileOp>(parent)) {
                if (owner == whileOp.getBeforeBody()) {
                    if (argNo < whileOp.getInits().size()) {
                        qubit = whileOp.getInits()[argNo];
                        continue;
                    }
                }
                else if (owner == whileOp.getAfterBody()) {
                    mlir::scf::ConditionOp cond = whileOp.getConditionOp();
                    mlir::ValueRange forwarded = cond.getOperands().drop_front();
                    if (argNo < forwarded.size()) {
                        qubit = forwarded[argNo];
                        continue;
                    }
                }
            }
        }

        break;
    }

    return mlir::Value();
}

} // namespace detail

/// Trace a qubit backward to the result of `quantum.extract` that defined the logical wire.
/// `visitor` is invoked for each `Value` visited along the path.
inline mlir::Value traceQubit(mlir::Value qubit, llvm::function_ref<void(mlir::Value)> visitor)
{
    return detail::traceQubitImpl(qubit, visitor);
}

/// Trace a qubit backward to the result of `quantum.extract` with empty visitor.
inline mlir::Value traceQubit(mlir::Value qubit)
{
    return detail::traceQubitImpl(qubit, [](mlir::Value) {});
}

} // namespace catalyst::quantum
