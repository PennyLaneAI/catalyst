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

#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/IR/DialectImplementation.h" // needed for generated type parser
#include "mlir/Transforms/InliningUtils.h"
#include "llvm/ADT/TypeSwitch.h" // needed for generated type parser

#include "Quantum/IR/QuantumDialect.h"
#include "Quantum/IR/QuantumOps.h"

using namespace mlir;
using namespace catalyst::quantum;

//===----------------------------------------------------------------------===//
// Quantum dialect definitions.
//===----------------------------------------------------------------------===//

#include "Quantum/IR/QuantumOpsDialect.cpp.inc"

namespace {
struct QNodeInlinerInterface : public DialectInlinerInterface {
    using DialectInlinerInterface::DialectInlinerInterface;

    //===--------------------------------------------------------------------===//
    // Analysis Hooks
    //===--------------------------------------------------------------------===//

    /// All call operations can be inlined.
    bool isLegalToInline(Operation *call, Operation *callable, bool wouldBeCloned) const final
    {
        return true;
    }

    /// All operations can be inlined.
    bool isLegalToInline(Operation *, Region *, bool, BlockAndValueMapping &) const final
    {
        return true;
    }

    /// All functions can be inlined.
    bool isLegalToInline(Region *, Region *, bool, BlockAndValueMapping &) const final
    {
        return true;
    }

    //===--------------------------------------------------------------------===//
    // Transformation Hooks
    //===--------------------------------------------------------------------===//

    /// Handle the given inlined terminator by replacing it with a new operation
    /// as necessary.
    void handleTerminator(Operation *op, Block *newDest) const final
    {
        // Only return needs to be handled here.
        auto returnOp = dyn_cast<ReturnOp>(op);
        if (!returnOp)
            return;

        // Replace the return with a branch to the dest.
        OpBuilder builder(op);
        builder.create<cf::BranchOp>(op->getLoc(), newDest, returnOp.getOperands());
        op->erase();
    }

    /// Handle the given inlined terminator by replacing it with a new operation
    /// as necessary.
    void handleTerminator(Operation *op, ArrayRef<Value> valuesToRepl) const final
    {
        // Only return needs to be handled here.
        auto returnOp = cast<ReturnOp>(op);

        // Replace the values directly with the return operands.
        assert(returnOp.getNumOperands() == valuesToRepl.size());
        for (const auto &it : llvm::enumerate(returnOp.getOperands()))
            valuesToRepl[it.index()].replaceAllUsesWith(it.value());
    }
};
} // namespace

void QuantumDialect::initialize()
{
    addTypes<
#define GET_TYPEDEF_LIST
#include "Quantum/IR/QuantumOpsTypes.cpp.inc"
        >();

    addOperations<
#define GET_OP_LIST
#include "Quantum/IR/QuantumOps.cpp.inc"
        >();
    addInterfaces<QNodeInlinerInterface>();
}

//===----------------------------------------------------------------------===//
// Quantum type definitions.
//===----------------------------------------------------------------------===//

#define GET_TYPEDEF_CLASSES
#include "Quantum/IR/QuantumOpsTypes.cpp.inc"
