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
// QRef op canonicalizers.
//===----------------------------------------------------------------------===//

LogicalResult AllocOp::canonicalize(AllocOp alloc, mlir::PatternRewriter &rewriter)
{
    if (alloc->use_empty()) {
        rewriter.eraseOp(alloc);
        return success();
    }

    return failure();
}

//===----------------------------------------------------------------------===//
// QRef op verifiers.
//===----------------------------------------------------------------------===//

static const mlir::StringSet<> validPauliWords = {"X", "Y", "Z", "I"};

LogicalResult AllocOp::verify()
{
    if (getNqubits() && getNqubitsAttr()) {
        return emitOpError() << "must have a single allocation size";
    }

    QuregType type = getQreg().getType();
    if (auto size = getNqubits()) {
        // Dynamic
        if (!type.isDynamic() || type.getSize().getInt() != mlir::ShapedType::kDynamic) {
            return emitOpError() << "expected result to have dynamic allocation size !qref.qreg<?>";
        }
    }

    else if (auto size = getNqubitsAttr()) {
        // Static
        if (!type.isStatic() || type.getSize().getInt() != size) {
            return emitOpError() << "expected result to have allocation size !qref.qreg<" << *size
                                 << ">";
        }
    }

    return success();
}

LogicalResult PauliRotOp::verify()
{
    size_t pauliWordLength = getPauliProduct().size();
    size_t numQubits = getQubits().size();
    if (pauliWordLength != numQubits) {
        return emitOpError() << "length of Pauli word (" << pauliWordLength
                             << ") and number of qubits (" << numQubits << ") must be the same";
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
    size_t dim = 1 << getQubits().size();
    if (failed(verifyTensorResult(cast<ShapedType>(getMatrix().getType()), dim, dim))) {
        return emitOpError("The Unitary matrix must be of size 2^(num_qubits) * 2^(num_qubits)");
    }

    return success();
}

} // namespace catalyst::qref
