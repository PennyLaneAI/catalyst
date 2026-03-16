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

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"

#include "RTIO/IR/RTIOOps.h"

using namespace mlir;
using namespace catalyst::rtio;

//===----------------------------------------------------------------------===//
// RPC Tag Verification
//===----------------------------------------------------------------------===//

namespace {
/// ARTIQ RPC tag format: <return_type>:<arg_types>
/// Type codes: n=void, i=i32, I=i64, f=f64, s=string, O=object
bool isTypeCompatibleWithTagCode(Type type, char code)
{
    switch (code) {
    case 'i': {
        auto intTy = dyn_cast<IntegerType>(type);
        return intTy && intTy.getWidth() == 32;
    }
    case 'I': {
        auto intTy = dyn_cast<IntegerType>(type);
        return intTy && intTy.getWidth() == 64;
    }
    case 'f': {
        return isa<Float64Type>(type);
    }
    case 's':
    case 'O':
        return isa<LLVM::LLVMPointerType>(type);
    default:
        return false;
    }
}
} // namespace

//===----------------------------------------------------------------------===//
// RTIO Operations
//===----------------------------------------------------------------------===//

LogicalResult RTIOSyncOp::verify()
{
    // Ensure at least one event is provided
    if (getEvents().empty()) {
        return emitOpError("requires at least one event to synchronize");
    }
    return success();
}

LogicalResult RTIORPCOp::verify()
{
    StringRef tag = getTag();
    size_t colon = tag.find(':');
    if (colon == StringRef::npos) {
        return emitOpError("tag must be in format '<return>:<args>' (e.g. n:IIf)");
    }

    StringRef returnPart = tag.take_front(colon);
    StringRef argsPart = tag.drop_front(colon + 1);

    // Validate return type vs results
    if (returnPart == "n") {
        if (!getResults().empty()) {
            return emitOpError("tag return type 'n' (void) requires no results");
        }
    }
    else {
        // Non-void return: must be sync and have exactly one result
        if (getIsAsync()) {
            return emitOpError("RPC with return value must be synchronous (remove 'async')");
        }
        if (getResults().size() != 1) {
            return emitOpError("tag return type '")
                   << returnPart << "' requires exactly one result";
        }
        char retCode = returnPart[0];
        if (returnPart.size() != 1) {
            return emitOpError("tag return type must be single character (i, I, f, s, O)");
        }
        if (!isTypeCompatibleWithTagCode(getResults()[0].getType(), retCode)) {
            return emitOpError("result type ")
                   << getResults()[0].getType() << " incompatible with tag return code '" << retCode
                   << "' (i=i32, I=i64, f=f64, s/O=ptr)";
        }
    }

    // Validate argument count
    size_t numArgs = getArgs().size();
    if (argsPart.size() != numArgs) {
        return emitOpError("tag has ")
               << argsPart.size() << " arg type code(s) but " << numArgs << " argument(s) provided";
    }

    // Validate each argument type
    for (size_t i = 0; i < numArgs; ++i) {
        Type argTy = getArgs()[i].getType();
        char code = argsPart[i];
        if (!isTypeCompatibleWithTagCode(argTy, code)) {
            return emitOpError("argument ")
                   << i << " has type " << argTy << " which is incompatible with tag code '" << code
                   << "' (i=i32, I=i64, f=f64, s/O=ptr)";
        }
    }

    return success();
}

#define GET_OP_CLASSES
#include "RTIO/IR/RTIOOps.cpp.inc"
