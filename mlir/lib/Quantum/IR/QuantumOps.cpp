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

#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"

#include "Quantum/IR/QuantumDialect.h"
#include "Quantum/IR/QuantumOps.h"

using namespace mlir;
using namespace catalyst::quantum;

//===----------------------------------------------------------------------===//
// Quantum op definitions.
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "Quantum/IR/QuantumOps.cpp.inc"

//===----------------------------------------------------------------------===//
// Quantum op interfaces.
//===----------------------------------------------------------------------===//

Optional<Operation *> AllocOp::buildDealloc(OpBuilder &builder, Value alloc)
{
    return builder.create<DeallocOp>(alloc.getLoc(), alloc).getOperation();
}

//===----------------------------------------------------------------------===//
// Quantum op canonicalizers.
//===----------------------------------------------------------------------===//

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

LogicalResult InsertOp::canonicalize(InsertOp insert, mlir::PatternRewriter &rewriter)
{
    if (auto extract = dyn_cast_if_present<ExtractOp>(insert.getQubit().getDefiningOp())) {
        bool bothStatic = extract.getIdxAttr().has_value() && insert.getIdxAttr().has_value();
        bool bothDynamic = !extract.getIdxAttr().has_value() && !insert.getIdxAttr().has_value();
        bool staticallyEqual = bothStatic && extract.getIdxAttrAttr() == insert.getIdxAttrAttr();
        bool dynamicallyEqual = bothDynamic && extract.getIdx() == insert.getIdx();

        if (staticallyEqual || dynamicallyEqual) {
            rewriter.replaceOp(insert, extract.getQreg());
            rewriter.eraseOp(extract);
            return success();
        }
    }

    return failure();
}

//===----------------------------------------------------------------------===//
// Quantum op verifiers.
//===----------------------------------------------------------------------===//

static LogicalResult verifyObservable(Value obs, size_t *numQubits)
{
    if (auto compOp = obs.getDefiningOp<ComputationalBasisOp>()) {
        *numQubits = compOp.getQubits().size();
        return success();
    }
    else if (obs.getDefiningOp<NamedObsOp>() || obs.getDefiningOp<HermitianOp>() ||
             obs.getDefiningOp<TensorOp>() || obs.getDefiningOp<HamiltonianOp>()) {
        return success();
    }

    return failure();
}

static LogicalResult verifyTensorResult(Type ty, int64_t length)
{
    ShapedType tensor = ty.cast<ShapedType>();
    if (!tensor.hasStaticShape() || tensor.getShape().size() != 1 ||
        tensor.getShape()[0] != length) {
        return failure();
    }

    return success();
}

static LogicalResult verifyTensorResult(Type ty, int64_t length0, int64_t length1)
{
    ShapedType tensor = ty.cast<ShapedType>();
    if (!tensor.hasStaticShape() || tensor.getShape().size() != 2 ||
        tensor.getShape()[0] != length0 || tensor.getShape()[1] != length1) {
        return failure();
    }

    return success();
}

// ----- gates

LogicalResult CustomOp::verify() { return verifyQubitNumbers(); }

LogicalResult MultiRZOp::verify()
{
    if (getInQubits().size() < 1) {
        return emitOpError("must have at least 1 qubit");
    }

    return verifyQubitNumbers();
}

LogicalResult QubitUnitaryOp::verify()
{
    if (getInQubits().size() < 1) {
        return emitOpError("must have at least 1 qubit");
    }

    size_t dim = std::pow(2, getInQubits().size());
    if (failed(verifyTensorResult(getMatrix().getType().cast<ShapedType>(), dim, dim))) {
        return emitOpError("The Unitary matrix must be of size 2^(num_qubits) * 2^(num_qubits)");
    }

    return verifyQubitNumbers();
}

// ----- measurements

LogicalResult HermitianOp::verify()
{
    size_t dim = std::pow(2, getQubits().size());
    if (failed(verifyTensorResult(getMatrix().getType().cast<ShapedType>(), dim, dim)))
        return emitOpError("The Hermitian matrix must be of size 2^(num_qubits) * 2^(num_qubits)");

    return success();
}

LogicalResult SampleOp::verify()
{
    size_t numQubits;
    if (failed(verifyObservable(getObs(), &numQubits))) {
        return emitOpError("observable must be locally defined");
    }

    if (getObs().getDefiningOp<ComputationalBasisOp>() &&
        failed(verifyTensorResult(getSamples().getType(), getShots(), numQubits))) {
        // In the computational basis, Pennylane adds a second dimension for the number of qubits.
        return emitOpError("return tensor must have 2D static shape equal to "
                           "(number of shots, number of qubits in observable)");
    }
    else if (!getObs().getDefiningOp<ComputationalBasisOp>() &&
             failed(verifyTensorResult(getSamples().getType(), getShots()))) {
        // For any given observables, Pennylane always returns a 1D tensor.
        return emitOpError("return tensor must have 1D static shape equal to (number of shots)");
    }

    return success();
}

LogicalResult CountsOp::verify()
{
    size_t numQubits = 0;
    if (failed(verifyObservable(getObs(), &numQubits))) {
        return emitOpError("observable must be locally defined");
    }

    size_t numEigvals = 0;
    if (getObs().getDefiningOp<NamedObsOp>()) {
        // Any named observable has 2 eigenvalues.
        numEigvals = 2;
    }
    else if (getObs().getDefiningOp<ComputationalBasisOp>()) {
        // In the computational basis, the "eigenvalues" are all possible bistrings one can measure.
        numEigvals = std::pow(2, numQubits);
    }
    else {
        return emitOpError("cannot determine the number of eigenvalues for general observable");
    }

    if (failed(verifyTensorResult(getEigvals().getType(), numEigvals)) ||
        failed(verifyTensorResult(getCounts().getType(), numEigvals))) {
        return emitOpError("number of eigenvalues or counts did not match observable");
    }

    return success();
}

LogicalResult ProbsOp::verify()
{
    size_t numQubits = 0;
    if (failed(verifyObservable(getObs(), &numQubits))) {
        return emitOpError("observable must be locally defined");
    }

    if (!numQubits) {
        return emitOpError("only computational basis observables are supported");
    }

    if (!getProbabilities()) {
        return success();
    }

    size_t dim = std::pow(2, numQubits);
    if (failed(verifyTensorResult(getProbabilities().getType().cast<ShapedType>(), dim))) {
        return emitOpError("return tensor must have static length equal to 2^(number of qubits)");
    }

    return success();
}

LogicalResult StateOp::verify()
{
    size_t numQubits = 0;
    if (failed(verifyObservable(getObs(), &numQubits))) {
        return emitOpError("observable must be locally defined");
    }

    if (!numQubits) {
        return emitOpError("only computational basis observables are supported");
    }

    if (!getState()) {
        return success();
    }

    size_t dim = std::pow(2, numQubits);
    if (failed(verifyTensorResult(getState().getType().cast<ShapedType>(), dim))) {
        return emitOpError("return tensor must have static length equal to 2^(number of qubits)");
    }

    return success();
}
