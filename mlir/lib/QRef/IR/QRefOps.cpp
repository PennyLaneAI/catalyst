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

#include "mlir/IR/OpImplementation.h"
#include "llvm/ADT/StringSet.h"

#include "QRef/IR/QRefDialect.h"
#include "QRef/IR/QRefOps.h"

using namespace mlir;
using namespace catalyst::qref;

//===----------------------------------------------------------------------===//
// QRef op definitions.
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "QRef/IR/QRefOps.cpp.inc"

namespace catalyst::qref {

// Utils
static LogicalResult verifyTensorResult(Type ty, int64_t length0, int64_t length1)
{
    ShapedType tensor = cast<ShapedType>(ty);
    if (!tensor.hasStaticShape() || tensor.getShape().size() != 2 ||
        tensor.getShape()[0] != length0 || tensor.getShape()[1] != length1) {
        return failure();
    }

    return success();
}

//===----------------------------------------------------------------------===//
// QRef op verifiers.
//===----------------------------------------------------------------------===//

static const mlir::StringSet<> validPauliWords = {"X", "Y", "Z", "I"};

LogicalResult PauliRotOp::verify()
{
    size_t pauliWordLength = getPauliProduct().size();
    size_t numWires = getWires().size();
    if (pauliWordLength != numWires) {
        return emitOpError() << "length of Pauli word (" << pauliWordLength
                             << ") and number of wires (" << numWires << ") must be the same";
    }

    if (!llvm::all_of(getPauliProduct(), [](mlir::Attribute attr) {
            auto pauliStr = llvm::cast<mlir::StringAttr>(attr);
            return validPauliWords.contains(pauliStr.getValue());
        })) {
        return emitOpError() << "Only \"X\", \"Y\", \"Z\", and \"I\" are valid Pauli words.";
    }

    return success();
}

LogicalResult QubitUnitaryOp::verify()
{
    size_t dim = 1 << getWires().size();
    if (failed(verifyTensorResult(cast<ShapedType>(getMatrix().getType()), dim, dim))) {
        return emitOpError("The Unitary matrix must be of size 2^(num_wires) * 2^(num_wires)");
    }

    return success();
}

} // namespace catalyst::qref
