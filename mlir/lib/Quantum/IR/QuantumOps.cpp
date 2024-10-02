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
#include "llvm/ADT/TypeSwitch.h"
#include <optional>

#include "Quantum/IR/QuantumDialect.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
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
            rewriter.replaceOp(insert, extract.getQreg());
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

    Type toVerify = getSamples() ? getSamples().getType() : getInData().getType();
    if (getObs().getDefiningOp<ComputationalBasisOp>() &&
        failed(verifyTensorResult(toVerify, getShots(), numQubits.value()))) {
        // In the computational basis, Pennylane adds a second dimension for the number of qubits.
        return emitOpError("return tensor must have 2D static shape equal to "
                           "(number of shots, number of qubits in observable)");
    }
    else if (!getObs().getDefiningOp<ComputationalBasisOp>() &&
             failed(verifyTensorResult(toVerify, getShots()))) {
        // For any given observables, Pennylane always returns a 1D tensor.
        return emitOpError("return tensor must have 1D static shape equal to (number of shots)");
    }

    return success();
}

LogicalResult CountsOp::verify()
{
    std::optional<size_t> numQubits = 0;
    if (failed(verifyObservable(getObs(), numQubits))) {
        return emitOpError("observable must be locally defined");
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

    bool xor_eigvals = (bool)getEigvals() ^ (bool)getInEigvals();
    bool xor_counts = (bool)getCounts() ^ (bool)getInCounts();
    bool is_valid = xor_eigvals && xor_counts;
    if (!is_valid) {
        return emitOpError("either tensors must be returned or memrefs must be used as inputs");
    }

    Type eigvalsToVerify =
        getEigvals() ? (Type)getEigvals().getType() : (Type)getInEigvals().getType();
    Type countsToVerify = getCounts() ? (Type)getCounts().getType() : (Type)getInCounts().getType();

    if (failed(verifyTensorResult(eigvalsToVerify, numEigvals)) ||
        failed(verifyTensorResult(countsToVerify, numEigvals))) {
        return emitOpError("number of eigenvalues or counts did not match observable");
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

    if (!(bool)getProbabilities() ^ (bool)getStateIn()) {
        return emitOpError("either tensors must be returned or memrefs must be used as inputs");
    }

    Type toVerify =
        getProbabilities() ? (Type)getProbabilities().getType() : (Type)getStateIn().getType();
    size_t dim = std::pow(2, numQubits.value());
    if (failed(verifyTensorResult(cast<ShapedType>(toVerify), dim))) {
        return emitOpError("return tensor must have static length equal to 2^(number of qubits)");
    }

    return success();
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

    if (!(bool)getState() ^ (bool)getStateIn()) {
        return emitOpError("either tensors must be returned or memrefs must be used as inputs");
    }

    Type toVerify = getState() ? (Type)getState().getType() : (Type)getStateIn().getType();
    size_t dim = std::pow(2, numQubits.value());
    if (failed(verifyTensorResult(cast<ShapedType>(toVerify), dim))) {
        return emitOpError("return tensor must have static length equal to 2^(number of qubits)");
    }

    return success();
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

void InitializeOp::getEffects(
    llvm::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>> &effects)
{
    effects.emplace_back(mlir::MemoryEffects::Allocate::get());
}

void FinalizeOp::getEffects(
    llvm::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>> &effects)
{
    effects.emplace_back(mlir::MemoryEffects::Free::get());
}

void DeviceInitOp::getEffects(
    llvm::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>> &effects)
{
    effects.emplace_back(mlir::MemoryEffects::Allocate::get());
}

void DeviceReleaseOp::getEffects(
    llvm::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>> &effects)
{
    effects.emplace_back(mlir::MemoryEffects::Free::get());
}

void DeallocOp::getEffects(
    llvm::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>> &effects)
{
    effects.emplace_back(mlir::MemoryEffects::Free::get());
}

void SetStateOp::getEffects(
    llvm::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>> &effects)
{
    effects.emplace_back(mlir::MemoryEffects::Write::get());
}

void SetBasisStateOp::getEffects(
    llvm::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>> &effects)
{
    effects.emplace_back(mlir::MemoryEffects::Write::get());
}

void GlobalPhaseOp::getEffects(
    llvm::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>> &effects)
{
    effects.emplace_back(mlir::MemoryEffects::Read::get());
    effects.emplace_back(mlir::MemoryEffects::Write::get());
}

void MeasureOp::getEffects(
    llvm::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>> &effects)
{
    effects.emplace_back(mlir::MemoryEffects::Read::get());
    effects.emplace_back(mlir::MemoryEffects::Write::get());
}

void ExpvalOp::getEffects(
    llvm::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>> &effects)
{
    effects.emplace_back(mlir::MemoryEffects::Read::get());
    effects.emplace_back(mlir::MemoryEffects::Write::get());
}

void ProbsOp::getEffects(
    llvm::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>> &effects)
{
    effects.emplace_back(mlir::MemoryEffects::Read::get());
    effects.emplace_back(mlir::MemoryEffects::Write::get());
}

void SampleOp::getEffects(
    llvm::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>> &effects)
{
    effects.emplace_back(mlir::MemoryEffects::Read::get());
    effects.emplace_back(mlir::MemoryEffects::Write::get());
}

void CountsOp::getEffects(
    llvm::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>> &effects)
{
    effects.emplace_back(mlir::MemoryEffects::Read::get());
    effects.emplace_back(mlir::MemoryEffects::Write::get());
}

void StateOp::getEffects(
    llvm::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>> &effects)
{
    effects.emplace_back(mlir::MemoryEffects::Read::get());
    effects.emplace_back(mlir::MemoryEffects::Write::get());
}
