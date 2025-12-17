// Copyright 2025 Xanadu Quantum Technologies Inc.

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

#include <queue>

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Value.h"

#include "Ion/IR/IonOps.h"
#include "Quantum/IR/QuantumOps.h"
#include "RTIO/IR/RTIOOps.h"

namespace catalyst {
namespace ion {

enum class TraceMode {
    Qreg = 0,
    Event = 1,
};

/// Traces a Value backward through the IR by tracing its dataflow dependencies
/// across control flow and specific quantum operations.
///
/// Template Parameters:
///   - ModeT: TraceMode enum (Qreg or Event) that controls how quantum.insert
///            operations are handled
///            Qreg mode: Trace to find the source qreg of the given value
///            Event mode: Trace to find all events that contribute to the given value
///   - CallbackT: Callable type that will be invoked for each visited value.
///                May optionally return WalkResult for early termination.
///
/// Supported Operations:
///   - scf.for
///   - scf.if
///   - ion.parallelprotocol
///   - unrealized_conversion_cast
///   - quantum.extract
///   - quantum.insert
template <TraceMode ModeT, typename CallbackT>
auto traceValueWithCallback(mlir::Value value, CallbackT &&callback)
{
    using namespace mlir;
    static_assert(std::is_same_v<std::invoke_result_t<CallbackT, Value>, WalkResult>,
                  "Callback must return WalkResult");

    WalkResult walkResult = WalkResult::advance();
    std::queue<Value> worklist;
    worklist.push(value);

    while (!worklist.empty()) {
        Value value = worklist.front();
        worklist.pop();

        if (callback(value).wasInterrupted()) {
            walkResult = WalkResult::interrupt();
            continue;
        }

        if (auto arg = mlir::dyn_cast<BlockArgument>(value)) {
            Block *block = arg.getOwner();
            Operation *parentOp = block->getParentOp();

            if (auto forOp = dyn_cast<scf::ForOp>(parentOp)) {
                unsigned argIndex = arg.getArgNumber();
                Value iterArg = forOp.getInitArgs()[argIndex - 1];
                worklist.push(iterArg);
                continue;
            }
            else if (auto parallelProtocolOp = dyn_cast<ion::ParallelProtocolOp>(parentOp)) {
                unsigned argIndex = arg.getArgNumber();
                Value inQubit = parallelProtocolOp.getInQubits()[argIndex];
                worklist.push(inQubit);
                continue;
            }
            return WalkResult::interrupt();
        }

        Operation *defOp = value.getDefiningOp();
        if (defOp == nullptr) {
            continue;
        }

        if (auto forOp = dyn_cast<scf::ForOp>(defOp)) {
            unsigned resultIdx = llvm::cast<OpResult>(value).getResultNumber();
            BlockArgument iterArg = forOp.getRegionIterArg(resultIdx);
            worklist.push(iterArg);
        }
        else if (auto ifOp = dyn_cast<scf::IfOp>(defOp)) {
            unsigned resultIdx = llvm::cast<OpResult>(value).getResultNumber();
            Value thenValue = ifOp.thenYield().getOperand(resultIdx);
            Value elseValue = ifOp.elseYield().getOperand(resultIdx);
            worklist.push(thenValue);
            worklist.push(elseValue);
        }
        else if (auto parallelProtocolOp = dyn_cast<ion::ParallelProtocolOp>(defOp)) {
            unsigned resultIdx = llvm::cast<OpResult>(value).getResultNumber();
            Value inQubit = parallelProtocolOp.getInQubits()[resultIdx];
            worklist.push(inQubit);
        }
        else if (auto op = dyn_cast<mlir::UnrealizedConversionCastOp>(defOp)) {
            worklist.push(op.getInputs().front());
        }
        else if (auto op = dyn_cast<quantum::ExtractOp>(defOp)) {
            worklist.push(op.getQreg());
        }
        else if (auto op = dyn_cast<rtio::RTIOQubitToChannelOp>(defOp)) {
            worklist.push(op.getQubit());
        }
        else if (auto op = dyn_cast<quantum::InsertOp>(defOp)) {
            Value inQreg = op.getInQreg();
            Value qubit = op.getQubit();
            if constexpr (ModeT == TraceMode::Qreg) {
                worklist.push(inQreg);
            }
            else if constexpr (ModeT == TraceMode::Event) {
                worklist.push(qubit);
                // only trace qreg if it defined op is also come from insert op
                if (llvm::isa_and_present<quantum::InsertOp>(inQreg.getDefiningOp())) {
                    worklist.push(inQreg);
                }
            }
        }
    }

    return walkResult;
}

} // namespace ion
} // namespace catalyst
