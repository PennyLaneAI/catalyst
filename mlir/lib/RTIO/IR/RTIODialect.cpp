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

#include "RTIO/IR/RTIODialect.h"
#include "RTIO/IR/RTIOOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace catalyst::rtio;

//===----------------------------------------------------------------------===//
// RTIO Dialect
//===----------------------------------------------------------------------===//

#include "RTIO/IR/RTIOOpsDialect.cpp.inc"

static ParseResult parseChannelTypeBody(AsmParser &parser, std::string &kind, ArrayAttr &qualifiers,
                                        IntegerAttr &channelId)
{
    // 1. Parse kind (string)
    if (failed(parser.parseString(&kind)))
        return failure();

    // 2. Parse optional qualifiers: `, [...]`
    qualifiers = nullptr;
    if (succeeded(parser.parseOptionalComma())) {
        if (succeeded(parser.parseOptionalLSquare())) {
            SmallVector<Attribute> quals;
            if (failed(parser.parseOptionalRSquare())) {
                do {
                    Attribute attr;
                    if (failed(parser.parseAttribute(attr)))
                        return failure();
                    quals.push_back(attr);
                } while (succeeded(parser.parseOptionalComma()));

                if (failed(parser.parseRSquare()))
                    return failure();
            }
            qualifiers = parser.getBuilder().getArrayAttr(quals);

            // After qualifiers, parse comma for channelId
            if (failed(parser.parseOptionalComma())) {
                channelId = parser.getBuilder().getI64IntegerAttr(ShapedType::kDynamic);
                return success();
            }
        }
        // Comma but no `[`, so this comma is for channelId
    }
    else {
        // No comma at all, no qualifiers and no channelId
        channelId = parser.getBuilder().getI64IntegerAttr(ShapedType::kDynamic);
        return success();
    }

    // 3. Parse channelId: `?` or non-negative integer
    if (succeeded(parser.parseOptionalQuestion())) {
        channelId = parser.getBuilder().getI64IntegerAttr(ShapedType::kDynamic);
        return success();
    }

    int64_t id = -1;
    if (failed(parser.parseInteger(id))) {
        return failure();
    }

    if (id < 0) {
        return parser.emitError(parser.getCurrentLocation(),
                                "static channel ID must be non-negative");
    }

    channelId = parser.getBuilder().getI64IntegerAttr(id);
    return success();
}

// Custom printer for the entire channel type body
static void printChannelTypeBody(AsmPrinter &printer, StringRef kind, ArrayAttr qualifiers,
                                 IntegerAttr channelId)
{
    // 1. Print kind
    printer << "\"" << kind << "\"";

    // 2. Print qualifiers if present
    if (qualifiers && !qualifiers.empty()) {
        printer << ", [";
        llvm::interleaveComma(qualifiers, printer,
                              [&](Attribute attr) { printer.printAttribute(attr); });
        printer << "]";
    }

    // 3. Print channelId if present (and not default ShapedType::kDynamic)
    if (channelId) {
        int64_t id = channelId.getInt();
        printer << ", ";
        if (id >= 0) {
            printer << id;
        }
        else {
            printer << "?";
        }
    }
    else if (qualifiers && !qualifiers.empty()) {
        printer << ", ?";
    }
}

void catalyst::rtio::RTIODialect::initialize()
{
    addTypes<
#define GET_TYPEDEF_LIST
#include "RTIO/IR/RTIOOpsTypes.cpp.inc"
        >();

    addAttributes<
#define GET_ATTRDEF_LIST
#include "RTIO/IR/RTIOAttributes.cpp.inc"
        >();

    addOperations<
#define GET_OP_LIST
#include "RTIO/IR/RTIOOps.cpp.inc"
        >();
}

//===----------------------------------------------------------------------===//
// RTIO Type Definitions
//===----------------------------------------------------------------------===//

#define GET_TYPEDEF_CLASSES
#include "RTIO/IR/RTIOOpsTypes.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "RTIO/IR/RTIOAttributes.cpp.inc"
