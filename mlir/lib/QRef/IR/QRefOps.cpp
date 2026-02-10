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

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"
#include "llvm/ADT/StringSet.h"

#include "QRef/IR/QRefDialect.h"
#include "QRef/IR/QRefOps.h"
#include "Quantum/IR/QuantumInterfaces.h"

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

static const mlir::StringSet<> hermitianOps = {"Hadamard", "PauliX", "PauliY", "PauliZ", "CNOT",
                                               "CY",       "CZ",     "SWAP",   "Toffoli"};
static const mlir::StringSet<> rotationsOps = {"RX",  "RY",  "RZ",  "PhaseShift",
                                               "CRX", "CRY", "CRZ", "ControlledPhaseShift"};

LogicalResult CustomOp::canonicalize(CustomOp op, mlir::PatternRewriter &rewriter)
{
    if (op.getAdjoint()) {
        auto name = op.getGateName();
        if (hermitianOps.contains(name)) {
            op.setAdjoint(false);
            return success();
        }
        else if (rotationsOps.contains(name)) {
            auto params = op.getParams();
            SmallVector<Value> paramsNeg;
            for (auto param : params) {
                auto paramNeg = mlir::arith::NegFOp::create(rewriter, op.getLoc(), param);
                paramsNeg.push_back(paramNeg);
            }

            rewriter.replaceOpWithNewOp<CustomOp>(op, paramsNeg, op.getQubits(), name, false,
                                                  op.getCtrlQubits(), op.getCtrlValues());

            return success();
        }
        return failure();
    }
    return failure();
}

LogicalResult MultiRZOp::canonicalize(MultiRZOp op, mlir::PatternRewriter &rewriter)
{
    if (op.getAdjoint()) {
        auto paramNeg = mlir::arith::NegFOp::create(rewriter, op.getLoc(), op.getTheta());

        rewriter.replaceOpWithNewOp<MultiRZOp>(op, paramNeg, op.getQubits(), nullptr,
                                               op.getCtrlQubits(), op.getCtrlValues());

        return success();
    };
    return failure();
}

LogicalResult PCPhaseOp::canonicalize(PCPhaseOp op, mlir::PatternRewriter &rewriter)
{
    if (op.getAdjoint()) {
        auto paramNeg = mlir::arith::NegFOp::create(rewriter, op.getLoc(), op.getTheta());

        rewriter.replaceOpWithNewOp<PCPhaseOp>(op, paramNeg, op.getDim(), op.getQubits(), nullptr,
                                               op.getCtrlQubits(), op.getCtrlValues());

        return success();
    };
    return failure();
}

LogicalResult AllocOp::canonicalize(AllocOp alloc, mlir::PatternRewriter &rewriter)
{
    if (alloc->use_empty()) {
        rewriter.eraseOp(alloc);
        return success();
    }

    return failure();
}

LogicalResult DeallocOp::canonicalize(DeallocOp dealloc, mlir::PatternRewriter &rewriter)
{
    if (auto alloc = dyn_cast_if_present<AllocOp>(dealloc.getQreg().getDefiningOp())) {
        if (dealloc.getQreg().hasOneUse()) {
            rewriter.eraseOp(dealloc);
            rewriter.eraseOp(alloc);
            return success();
        }
    }

    return failure();
}

//===----------------------------------------------------------------------===//
// QRef op verifiers.
//===----------------------------------------------------------------------===//

static const mlir::StringSet<> validPauliWords = {"X", "Y", "Z", "I"};

LogicalResult AllocOp::verify()
{
    if (!(getNqubits() || getNqubitsAttr().has_value())) {
        return emitOpError() << "expected op to have a non-null allocation size";
    }

    if (getNqubits() && getNqubitsAttr().has_value()) {
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

LogicalResult CustomOp::verify()
{
    if (getQubits().size() == 0) {
        return emitOpError("expected op to have at least one qubit");
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

LogicalResult AdjointOp::verify()
{
    auto res = this->getRegion().walk(
        [](catalyst::quantum::MeasurementProcess op) { return WalkResult::interrupt(); });

    if (res.wasInterrupted()) {
        return emitOpError("quantum measurements are not allowed in the adjoint regions");
    }

    return success();
}

LogicalResult ComputationalBasisOp::verify()
{
    if ((getQubits().size() != 0) && (getQreg() != nullptr)) {
        return emitOpError()
               << "computational basis op cannot simultaneously take in both qubits and quregs";
    }

    return success();
}

LogicalResult HermitianOp::verify()
{
    size_t dim = std::pow(2, getQubits().size());
    if (failed(verifyTensorResult(cast<ShapedType>(getMatrix().getType()), dim, dim)))
        return emitOpError("The Hermitian matrix must be of size 2^(num_qubits) * 2^(num_qubits)");

    return success();
}

} // namespace catalyst::qref
