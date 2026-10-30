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

#include "Quantum/IR/QuantumDialect.h"

#include "llvm/ADT/TypeSwitch.h" // needed for generated type parser
#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/DialectImplementation.h" // needed for generated type parser
#include "mlir/Transforms/InliningUtils.h"

#include "Quantum/IR/QuantumOps.h"

using namespace mlir;
using namespace catalyst::quantum;

//===----------------------------------------------------------------------===//
// Quantum Dialect Interfaces
//===----------------------------------------------------------------------===//

namespace {

struct QuantumInlinerInterface : public DialectInlinerInterface {
    using DialectInlinerInterface::DialectInlinerInterface;

    /// Returns true if the given operation 'callable' can be inlined into the
    /// position given by the 'call'. Quantum operations impose no restrictions.
    bool isLegalToInline(Operation *, Operation *, bool) const final { return true; }

    /// Returns true if the given region 'src' can be inlined into the region 'dest'.
    bool isLegalToInline(Region *, Region *, bool, IRMapping &) const final { return true; }

    /// Operations in the quantum dialect are always legal to inline.
    bool isLegalToInline(Operation *, Region *, bool, IRMapping &) const final { return true; }

    /// Handle the given inlined terminator by replacing it with a new operation
    /// as necessary. Required when the region has only one block.
    void handleTerminator(Operation *op, ValueRange valuesToRepl) const final {
        auto yieldOp = dyn_cast<YieldOp>(op);
        if (!yieldOp) {
            return;
        }

        for (auto retValue : llvm::zip(valuesToRepl, yieldOp.getOperands())) {
            std::get<0>(retValue).replaceAllUsesWith(std::get<1>(retValue));
        }
    }
};
} // namespace

//===----------------------------------------------------------------------===//
// Quantum dialect definitions.
//===----------------------------------------------------------------------===//

#include "Quantum/IR/QuantumOpsDialect.cpp.inc"

void QuantumDialect::initialize() {
    addTypes<
#define GET_TYPEDEF_LIST
#include "Quantum/IR/QuantumOpsTypes.cpp.inc"
        >();

    addAttributes<
#define GET_ATTRDEF_LIST
#include "Quantum/IR/QuantumAttributes.cpp.inc"
        >();

    addOperations<
#define GET_OP_LIST
#include "Quantum/IR/QuantumOps.cpp.inc"
        >();

    addInterfaces<QuantumInlinerInterface>();

    declarePromisedInterfaces<bufferization::BufferizableOpInterface, QubitUnitaryOp, HermitianOp,
                              HamiltonianOp, SampleOp, CountsOp, ProbsOp, StateOp, SetStateOp,
                              SetBasisStateOp>();
}

/// Verify the QNode and kernel entry point attribute invariants
LogicalResult QuantumDialect::verifyOperationAttribute(Operation *op, NamedAttribute namedAttr) {
    StringRef attrName = namedAttr.getName().getValue();
    if (attrName != "quantum.node" && attrName != "quantum.kernel_entry_point") {
        return success();
    }

    if (!isa<func::FuncOp>(op)) {
        return op->emitOpError() << "attribute '" << attrName << "' is only valid on 'func.func'";
    }

    if (!isa<UnitAttr>(namedAttr.getValue())) {
        return op->emitOpError() << "attribute '" << attrName << "' must be a unit attribute";
    }

    // Cannot enforce this yet as Catalyst allows QNodes without MPs.
    // auto funcOp = cast<func::FuncOp>(op);
    // auto measurement = funcOp.walk([&](MeasurementProcess) { return WalkResult::interrupt(); });
    // if (!measurement.wasInterrupted()) {
    //     return op->emitOpError()
    //            << "attribute '" << attrName
    //            << "' requires at least one measurement process operation in the function body";
    // }

    return success();
}

//===----------------------------------------------------------------------===//
// Quantum type definitions.
//===----------------------------------------------------------------------===//

#define GET_TYPEDEF_CLASSES
#include "Quantum/IR/QuantumOpsTypes.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "Quantum/IR/QuantumAttributes.cpp.inc"
