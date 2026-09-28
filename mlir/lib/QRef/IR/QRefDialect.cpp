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

#include "QRef/IR/QRefDialect.h"

#include <cstdint>

#include "llvm/ADT/TypeSwitch.h" // needed for generated type parser
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h" // needed for generated type parser
#include "mlir/Transforms/InliningUtils.h"

#include "QRef/IR/QRefOps.h"

using namespace mlir;
using namespace catalyst::qref;

//===----------------------------------------------------------------------===//
// QRef dialect definitions.
//===----------------------------------------------------------------------===//

#include "QRef/IR/QRefOpsDialect.cpp.inc"

namespace {

static ParseResult parseQuregTypeBody(AsmParser &parser, IntegerAttr &size) {
    // Parse allocation size: `?` or non-negative integer
    if (succeeded(parser.parseOptionalQuestion())) {
        size = parser.getBuilder().getI64IntegerAttr(ShapedType::kDynamic);
        return success();
    }

    int64_t id = -1;
    if (failed(parser.parseInteger(id))) {
        return failure();
    }

    if (id < 0) {
        return parser.emitError(parser.getCurrentLocation(),
                                "Static allocation size must be non-negative");
    }

    size = parser.getBuilder().getI64IntegerAttr(id);
    return success();
}

static void printQuregTypeBody(AsmPrinter &printer, IntegerAttr size) {
    if (size) {
        int64_t id = size.getInt();
        if (id >= 0) {
            printer << id;
        } else {
            printer << "?";
        }
    } else {
        printer << "?";
    }
}

// This class defines the interface for handling inlining for qref
// dialect operations.
// Similar to the scf dialect, we allow all inlining.
struct QrefInlinerInterface : public DialectInlinerInterface {
    using DialectInlinerInterface::DialectInlinerInterface;

    // We don't have any special restrictions on what can be inlined into
    // destination regions (e.g. qref.adjoint/ctrl bodies). Always allow it.
    bool isLegalToInline(Region *dest, Region *src, bool wouldBeCloned,
                         IRMapping &valueMapping) const final {
        return true;
    }

    // Operations in qref dialect are always legal to inline.
    bool isLegalToInline(Operation *, Region *, bool, IRMapping &) const final { return true; }
};
} // namespace

void QRefDialect::initialize() {
    addTypes<
#define GET_TYPEDEF_LIST
#include "QRef/IR/QRefOpsTypes.cpp.inc"
        >();

    addOperations<
#define GET_OP_LIST
#include "QRef/IR/QRefOps.cpp.inc"
        >();

    addInterfaces<QrefInlinerInterface>();
}

//===----------------------------------------------------------------------===//
// QRef type definitions.
//===----------------------------------------------------------------------===//

#define GET_TYPEDEF_CLASSES
#include "QRef/IR/QRefOpsTypes.cpp.inc"
