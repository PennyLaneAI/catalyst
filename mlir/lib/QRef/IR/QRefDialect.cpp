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

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h" // needed for generated type parser
#include "llvm/ADT/TypeSwitch.h"           // needed for generated type parser

#include "QRef/IR/QRefDialect.h"
#include "QRef/IR/QRefOps.h"

using namespace mlir;
using namespace catalyst::qref;

//===----------------------------------------------------------------------===//
// QRef dialect definitions.
//===----------------------------------------------------------------------===//

#include "QRef/IR/QRefOpsDialect.cpp.inc"

static ParseResult parseQuregTypeBody(AsmParser &parser, IntegerAttr &size)
{
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

static void printQuregTypeBody(AsmPrinter &printer, IntegerAttr size)
{
    if (size) {
        int64_t id = size.getInt();
        if (id >= 0) {
            printer << id;
        }
        else {
            printer << "?";
        }
    }
    else {
        printer << "?";
    }
}

void QRefDialect::initialize()
{
    addTypes<
#define GET_TYPEDEF_LIST
#include "QRef/IR/QRefOpsTypes.cpp.inc"
        >();

    addOperations<
#define GET_OP_LIST
#include "QRef/IR/QRefOps.cpp.inc"
        >();
}

//===----------------------------------------------------------------------===//
// QRef type definitions.
//===----------------------------------------------------------------------===//

#define GET_TYPEDEF_CLASSES
#include "QRef/IR/QRefOpsTypes.cpp.inc"
