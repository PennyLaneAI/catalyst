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

#include <optional>
#include <type_traits>

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/TypeSwitch.h"

#include "Quantum/IR/QuantumDialect.h"
#include "Quantum/IR/QuantumOps.h"

using namespace mlir;
using namespace catalyst::quantum;

//===----------------------------------------------------------------------===//
// Quantum op definitions.
//===----------------------------------------------------------------------===//

#include "Quantum/IR/QuantumEnums.cpp.inc"
#define GET_OP_CLASSES
#include "Quantum/IR/QuantumOps.cpp.inc"

//===----------------------------------------------------------------------===//
// Quantum op canonicalizers.
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
                auto paramNeg = rewriter.create<mlir::arith::NegFOp>(op.getLoc(), param);
                paramsNeg.push_back(paramNeg);
            }

            rewriter.replaceOpWithNewOp<CustomOp>(
                op, op.getOutQubits().getTypes(), op.getOutCtrlQubits().getTypes(), paramsNeg,
                op.getInQubits(), name, false, op.getInCtrlQubits(), op.getInCtrlValues());

            return success();
        }
        return failure();
    }
    return failure();
}

LogicalResult MultiRZOp::canonicalize(MultiRZOp op, mlir::PatternRewriter &rewriter)
{
    if (op.getAdjoint()) {
        auto paramNeg = rewriter.create<mlir::arith::NegFOp>(op.getLoc(), op.getTheta());

        rewriter.replaceOpWithNewOp<MultiRZOp>(
            op, op.getOutQubits().getTypes(), op.getOutCtrlQubits().getTypes(), paramNeg,
            op.getInQubits(), nullptr, op.getInCtrlQubits(), op.getInCtrlValues());

        return success();
    };
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

template <typename IndexingOp> LogicalResult foldConstantIndexingOp(IndexingOp op, Attribute idx)
{
    // Prefer using an attribute when the index is constant.
    bool hasNoIdxAttr = !op.getIdxAttr().has_value();
    bool isConstantIdx = isa_and_nonnull<IntegerAttr>(idx);
    if (hasNoIdxAttr && isConstantIdx) {
        auto constantIdx = cast<IntegerAttr>(idx);
        op.setIdxAttr(constantIdx.getValue().getSExtValue());

        // Remove the dynamic Value
        op.getIdxMutable().clear();
        return success();
    }
    return failure();
}

OpFoldResult ExtractOp::fold(FoldAdaptor adaptor)
{
    if (succeeded(foldConstantIndexingOp(*this, adaptor.getIdx()))) {
        return getResult();
    }
    // Returning nullptr tells the caller the op was unchanged.
    return nullptr;
}

LogicalResult InsertOp::canonicalize(InsertOp insert, mlir::PatternRewriter &rewriter)
{
    if (auto extract = dyn_cast_if_present<ExtractOp>(insert.getQubit().getDefiningOp())) {
        bool bothStatic = extract.getIdxAttr().has_value() && insert.getIdxAttr().has_value();
        bool bothDynamic = !extract.getIdxAttr().has_value() && !insert.getIdxAttr().has_value();
        bool staticallyEqual = bothStatic && extract.getIdxAttrAttr() == insert.getIdxAttrAttr();
        bool dynamicallyEqual = bothDynamic && extract.getIdx() == insert.getIdx();
        bool oneUse = extract.getResult().hasOneUse();

        if ((staticallyEqual || dynamicallyEqual) && oneUse) {
            rewriter.replaceOp(insert, insert.getInQreg());
            rewriter.eraseOp(extract);
            return success();
        }
    }

    return failure();
}

OpFoldResult InsertOp::fold(FoldAdaptor adaptor)
{
    if (succeeded(foldConstantIndexingOp(*this, adaptor.getIdx()))) {
        return getResult();
    }
    // Returning nullptr tells the caller the op was unchanged.
    return nullptr;
}

//===----------------------------------------------------------------------===//
// Quantum op verifiers.
//===----------------------------------------------------------------------===//

LogicalResult ExtractOp::verify()
{
    if (!(getIdx() || getIdxAttr().has_value())) {
        return emitOpError() << "expected op to have a non-null index";
    }
    return success();
}

LogicalResult InsertOp::verify()
{
    if (!(getIdx() || getIdxAttr().has_value())) {
        return emitOpError() << "expected op to have a non-null index";
    }
    return success();
}

static LogicalResult verifyObservable(Value obs, std::optional<size_t> &numQubits)
{
    if (auto compOp = obs.getDefiningOp<ComputationalBasisOp>()) {
        numQubits = compOp.getQubits().size();
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
    ShapedType tensor = cast<ShapedType>(ty);
    if (!tensor.hasStaticShape() || tensor.getShape().size() != 1 ||
        tensor.getShape()[0] != length) {
        return failure();
    }

    return success();
}

static LogicalResult verifyTensorResult(Type ty, int64_t length0, int64_t length1)
{
    ShapedType tensor = cast<ShapedType>(ty);
    if (!tensor.hasStaticShape() || tensor.getShape().size() != 2 ||
        tensor.getShape()[0] != length0 || tensor.getShape()[1] != length1) {
        return failure();
    }

    return success();
}

// ----- gates

LogicalResult QubitUnitaryOp::verify()
{
    size_t dim = std::pow(2, getInQubits().size());
    if (failed(verifyTensorResult(cast<ShapedType>(getMatrix().getType()), dim, dim))) {
        return emitOpError("The Unitary matrix must be of size 2^(num_qubits) * 2^(num_qubits)");
    }

    return success();
}

// ----- measurements

template <typename T>
static LogicalResult verifyMeasurementOpDynamism(T *op, bool hasObs, bool hasDynShape,
                                                 bool hasBufferIn, bool hasOutTensor)
{
    // `obs` operand must always be present
    if (!hasObs) {
        return (*op)->emitOpError("must take an observable");
    }

    // If a tensor is not returned, must be bufferized.
    // If a tensor is returned, must be unbufferized.
    if (!hasOutTensor ^ hasBufferIn) {
        return (*op)->emitOpError(
            "either tensors must be returned or memrefs must be used as inputs");
    }

    // Two cases are allowed when a tensor is returned.
    // 1. Either return shape is completely static, and no length is specified in argument,
    // 2. Or return shape is dynamic and a length argument is specified.
    if (hasOutTensor) {
        ShapedType outTensor;
        if constexpr (std::is_same_v<T, ProbsOp>) {
            outTensor = cast<ShapedType>(op->getProbabilities().getType());
        }
        else if constexpr (std::is_same_v<T, StateOp>) {
            outTensor = cast<ShapedType>(op->getState().getType());
        }
        else if constexpr (std::is_same_v<T, SampleOp>) {
            outTensor = cast<ShapedType>(op->getSamples().getType());
        }
        else if constexpr (std::is_same_v<T, CountsOp>) {
            outTensor = cast<ShapedType>(op->getCounts().getType());
        }

        if (outTensor.hasStaticShape() && hasDynShape) {
            return (*op)->emitOpError(
                "with static return shapes should not specify dynamic shape in arguments");
        }
        if (!outTensor.hasStaticShape() && !hasDynShape) {
            return (*op)->emitOpError(
                "with dynamic return shapes must specify dynamic shape in arguments");
        }
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

LogicalResult SampleOp::verify()
{
    std::optional<size_t> numQubits = 0;

    if (failed(verifyObservable(getObs(), numQubits))) {
        return emitOpError("observable must be locally defined");
    }

    if (!((bool)getSamples() ^ (bool)getInData())) {
        return emitOpError("either tensors must be returned or memrefs must be used as inputs");
    }

    bool hasObs = (bool)getObs();
    bool hasDynShape = (bool)(getDynamicShape().size());
    bool hasBufferIn = (bool)getInData();
    bool hasOutTensor = (bool)getSamples();
    return verifyMeasurementOpDynamism<SampleOp>(this, hasObs, hasDynShape, hasBufferIn,
                                                 hasOutTensor);
}

LogicalResult CountsOp::verify()
{
    std::optional<size_t> numQubits = 0;

    if (failed(verifyObservable(getObs(), numQubits))) {
        return emitOpError("observable must be locally defined");
    }

    bool hasObs = (bool)getObs();
    bool hasDynShape = (bool)getDynamicShape();
    bool hasBufferIn = (bool)getInCounts();
    bool hasOutTensor = (bool)getCounts();
    if (failed(verifyMeasurementOpDynamism<CountsOp>(this, hasObs, hasDynShape, hasBufferIn,
                                                     hasOutTensor))) {
        return failure();
    }

    size_t numEigvals = 0;
    if (getObs().getDefiningOp<NamedObsOp>()) {
        // Any named observable has 2 eigenvalues.
        numEigvals = 2;
    }
    else if (getObs().getDefiningOp<ComputationalBasisOp>()) {
        // In the computational basis, the "eigenvalues" are all possible bistrings one can measure.
        numEigvals = std::pow(2, numQubits.value());
    }
    else {
        return emitOpError("cannot determine the number of eigenvalues for general observable");
    }

    ShapedType outTensor = getEigvals() ? cast<ShapedType>(getEigvals().getType())
                                        : cast<ShapedType>(getInEigvals().getType());
    if (!outTensor.isDynamicDim(0)) {
        if (getObs().getDefiningOp<NamedObsOp>() || numQubits.value() != 0) {
            Type eigvalsToVerify =
                getEigvals() ? (Type)getEigvals().getType() : (Type)getInEigvals().getType();
            Type countsToVerify =
                getCounts() ? (Type)getCounts().getType() : (Type)getInCounts().getType();

            if (failed(verifyTensorResult(eigvalsToVerify, numEigvals)) ||
                failed(verifyTensorResult(countsToVerify, numEigvals))) {
                return emitOpError("number of eigenvalues or counts did not match observable");
            }
        }
    }

    return success();
}

LogicalResult ProbsOp::verify()
{
    std::optional<size_t> numQubits;
    if (failed(verifyObservable(getObs(), numQubits))) {
        return emitOpError("observable must be locally defined");
    }

    if (!numQubits.has_value()) {
        return emitOpError("only computational basis observables are supported");
    }

    bool hasObs = (bool)getObs();
    bool hasDynShape = (bool)getDynamicShape();
    bool hasStateIn = (bool)getStateIn();
    bool hasOutTensor = (bool)getProbabilities();
    return verifyMeasurementOpDynamism<ProbsOp>(this, hasObs, hasDynShape, hasStateIn,
                                                hasOutTensor);
}

LogicalResult StateOp::verify()
{
    std::optional<size_t> numQubits;
    if (failed(verifyObservable(getObs(), numQubits))) {
        return emitOpError("observable must be locally defined");
    }

    if (!numQubits.has_value()) {
        return emitOpError("only computational basis observables are supported");
    }

    bool hasObs = (bool)getObs();
    bool hasDynShape = (bool)getDynamicShape();
    bool hasStateIn = (bool)getStateIn();
    bool hasOutTensor = (bool)getState();
    return verifyMeasurementOpDynamism<StateOp>(this, hasObs, hasDynShape, hasStateIn,
                                                hasOutTensor);
}

LogicalResult AdjointOp::verify()
{
    auto res =
        this->getRegion().walk([](MeasurementProcess op) { return WalkResult::interrupt(); });

    if (res.wasInterrupted()) {
        return emitOpError("quantum measurements are not allowed in the adjoint regions");
    }

    return success();
}
