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

#include "Quantum/IR/QuantumOps.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Support/WalkResult.h"

#include "QRef/IR/QRefOps.h"
#include "Quantum/IR/QuantumAttrDefs.h"
#include "Quantum/IR/QuantumDialect.h"
#include "Quantum/IR/QuantumInterfaces.h"
#include "Quantum/IR/QuantumTypes.h"

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
                auto paramNeg = mlir::arith::NegFOp::create(rewriter, op.getLoc(), param);
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
        auto paramNeg = mlir::arith::NegFOp::create(rewriter, op.getLoc(), op.getTheta());

        rewriter.replaceOpWithNewOp<MultiRZOp>(
            op, op.getOutQubits().getTypes(), op.getOutCtrlQubits().getTypes(), paramNeg,
            op.getInQubits(), nullptr, op.getInCtrlQubits(), op.getInCtrlValues());

        return success();
    };
    return failure();
}

LogicalResult PCPhaseOp::canonicalize(PCPhaseOp op, mlir::PatternRewriter &rewriter)
{
    if (op.getAdjoint()) {
        auto paramNeg = mlir::arith::NegFOp::create(rewriter, op.getLoc(), op.getTheta());

        rewriter.replaceOpWithNewOp<PCPhaseOp>(
            op, op.getOutQubits().getTypes(), op.getOutCtrlQubits().getTypes(), paramNeg,
            op.getDimAttr(), op.getInQubits(), nullptr, op.getInCtrlQubits(), op.getInCtrlValues());

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

LogicalResult ExtractOp::canonicalize(ExtractOp extract, mlir::PatternRewriter &rewriter)
{
    // Handle the pattern: %reg2 = insert %reg1[idx], %qubit -> %q = extract %reg2[idx]
    // Convert to: %q = %qubit, and replace other uses of %reg2 with %reg1
    if (auto insert = dyn_cast_if_present<InsertOp>(extract.getQreg().getDefiningOp())) {
        bool bothStatic = extract.getIdxAttr().has_value() && insert.getIdxAttr().has_value();
        bool bothDynamic = !extract.getIdxAttr().has_value() && !insert.getIdxAttr().has_value();
        bool staticallyEqual = bothStatic && extract.getIdxAttrAttr() == insert.getIdxAttrAttr();
        bool dynamicallyEqual = bothDynamic && extract.getIdx() == insert.getIdx();

        bool inSameBlock = extract->getBlock() == insert->getBlock();

        if ((staticallyEqual || dynamicallyEqual) && inSameBlock) {
            rewriter.replaceOp(extract, insert.getQubit());
            rewriter.replaceOp(insert, insert.getInQreg());
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
        bool sameQreg = extract.getQreg() == insert.getInQreg();
        bool oneUse = extract.getResult().hasOneUse();

        if ((staticallyEqual || dynamicallyEqual) && oneUse && sameQreg) {
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

LogicalResult CustomOp::verify()
{
    if (getInQubits().size() == 0) {
        return emitOpError("expected op to have at least one qubit");
    }
    return success();
}

LogicalResult OperatorOp::verify()
{
    const bool hasQregInput = static_cast<bool>(getInQreg());
    const bool hasQregOutput = static_cast<bool>(getOutQreg());
    const bool hasQregMode = hasQregInput || hasQregOutput;

    const bool hasQubitInput = !getInQubits().empty();
    const bool hasQubitOutput = !getOutQubits().empty();
    const bool hasQubitMode = hasQubitInput || hasQubitOutput;

    // At most one mode must be used: explicit qubits or qreg-based addressing.
    if (hasQregMode && hasQubitMode) {
        return emitOpError() << "must use either qubits or registers, but not both";
    }

    if (hasQregInput != hasQregOutput) {
        return emitOpError() << "in_qreg and out_qreg must either both be present or absent";
    }

    if (!getForwardArgs().empty() && !getUID()) {
        return emitOpError() << "forward_args can only be present when UID is provided";
    }

    const bool hasQubitControls =
        !getInCtrlQubits().empty() || !getInCtrlValues().empty() || !getOutCtrlQubits().empty();
    const bool hasQregControls =
        static_cast<bool>(getArrCtrlIndices()) || static_cast<bool>(getArrCtrlValues());

    if (hasQubitControls && hasQregControls) {
        return emitOpError()
               << "cannot mix qubit controls (in_ctrl_qubits/in_ctrl_values/out_ctrl_qubits) "
               << "with register controls (arr_ctrl_indices/arr_ctrl_values)";
    }

    if (static_cast<bool>(getArrCtrlIndices()) != static_cast<bool>(getArrCtrlValues())) {
        return emitOpError()
               << "arr_ctrl_indices and arr_ctrl_values must either both be present or both absent";
    }

    if (hasQregControls) {
        auto ctrlIndType = cast<ShapedType>(getArrCtrlIndices().getType());
        auto ctrlValType = cast<ShapedType>(getArrCtrlValues().getType());
        if (ctrlIndType.getShape()[0] != ctrlValType.getShape()[0]) {
            return emitOpError() << "number of input control qubits (" << ctrlIndType.getShape()[0]
                                 << ") and control values (" << ctrlValType.getShape()[0]
                                 << ") must be the same";
        }
    }

    if (getParamMapAttr()) {
        const size_t numParams = getParams().size();
        std::vector<bool> coveredParams(numParams, false);
        size_t coveredCount = 0;
        for (NamedAttribute namedAttr : getParamMap()) {
            auto denseArray = cast<DenseI64ArrayAttr>(namedAttr.getValue());
            for (int64_t idx : denseArray.asArrayRef()) {
                if (idx < 0 || idx >= static_cast<int64_t>(numParams)) {
                    return emitOpError() << "param_map index is out of bounds with respect to "
                                            "params: "
                                         << idx << " is not in [0, " << numParams << ")";
                }
                if (!coveredParams[static_cast<size_t>(idx)]) {
                    coveredParams[static_cast<size_t>(idx)] = true;
                    ++coveredCount;
                }
            }
        }
        if (coveredCount != numParams) {
            return emitOpError() << "param_map must cover all params when provided: expected "
                                 << numParams << ", got " << coveredCount;
        }
    }

    auto qubitMap = getQubitMap();
    if (qubitMap) {
        const size_t numTargets = hasQregMode ? getArrQubitIndices().size() : getInQubits().size();
        const char *boundsNoun = hasQregMode ? "index arrays" : "qubits";
        const char *coverageNoun =
            hasQregMode ? "index arrays in register mode" : "qubit values in qubit mode";

        llvm::SmallDenseSet<int64_t> coveredTargets;
        for (NamedAttribute namedAttr : qubitMap) {
            auto denseArray = cast<DenseI64ArrayAttr>(namedAttr.getValue());
            for (int64_t idx : denseArray.asArrayRef()) {
                if (idx < 0 || idx >= static_cast<int64_t>(numTargets)) {
                    return emitOpError()
                           << "qubit_map index is out of bounds with respect to " << boundsNoun
                           << ": " << idx << " is not in [0, " << numTargets << ")";
                }
                coveredTargets.insert(idx);
            }
        }
        if (getQubitMapAttr() && coveredTargets.size() != numTargets) {
            return emitOpError() << "qubit_map must cover all " << coverageNoun << ": expected "
                                 << numTargets << ", got " << coveredTargets.size();
        }
    }

    return success();
}

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
    if (auto compOp = obs.getDefiningOp<catalyst::qref::ComputationalBasisOp>()) {
        numQubits = compOp.getQubits().size();
        return success();
    }
    else if (obs.getDefiningOp<NamedObsOp>() || obs.getDefiningOp<HermitianOp>() ||
             obs.getDefiningOp<TensorOp>() || obs.getDefiningOp<HamiltonianOp>() ||
             obs.getDefiningOp<catalyst::qref::NamedObsOp>() ||
             obs.getDefiningOp<catalyst::qref::HermitianOp>()) {
        return success();
    }
    else if (auto mcmObsOp = obs.getDefiningOp<MCMObsOp>()) {
        numQubits = mcmObsOp.getMcms().size();
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

static const mlir::StringSet<> validPauliWords = {"X", "Y", "Z", "I"};

LogicalResult PauliRotOp::verify()
{
    size_t pauliWordLength = getPauliProduct().size();
    size_t numQubits = getInQubits().size();
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
    size_t dim = std::pow(2, getInQubits().size());
    if (failed(verifyTensorResult(cast<ShapedType>(getMatrix().getType()), dim, dim))) {
        return emitOpError("The Unitary matrix must be of size 2^(num_qubits) * 2^(num_qubits)");
    }

    return success();
}

// ----- measurements

static LogicalResult verifyInQNodeFunction(Operation *op)
{
    // strict verification only for quantum kernels
    // detection is a bit tricky until we have a dedicated operation, use heuristics for now
    auto kernelModule = op->getParentOfType<ModuleOp>();
    if (!kernelModule) {
        return success();
    }

    auto modIt = kernelModule.getOps<ModuleOp>();
    if (modIt.empty() || !(*modIt.begin())->hasAttr("transform.with_named_sequence")) {
        return success();
    }

    auto parentFunc = op->getParentOfType<func::FuncOp>();
    if (!parentFunc) {
        return op->emitOpError("must be nested inside a 'func.func' operation");
    }
    if (!parentFunc->hasAttrOfType<UnitAttr>("quantum.node")) {
        return op->emitOpError("requires parent function to carry 'quantum.node' attribute");
    }
    return success();
}

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
    if (failed(verifyInQNodeFunction(getOperation()))) {
        return failure();
    }

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
    if (failed(verifyInQNodeFunction(getOperation()))) {
        return failure();
    }

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
    if (getObs().getDefiningOp<quantum::NamedObsOp>() ||
        getObs().getDefiningOp<qref::NamedObsOp>()) {
        // Any named observable has 2 eigenvalues.
        numEigvals = 2;
    }
    else if (getObs().getDefiningOp<quantum::ComputationalBasisOp>() ||
             getObs().getDefiningOp<qref::ComputationalBasisOp>()) {
        // In the computational basis, the "eigenvalues" are all possible bistrings one can measure.
        numEigvals = std::pow(2, numQubits.value());
    }
    else if (getObs().getDefiningOp<MCMObsOp>()) {
        // When counting MCMs, the "eigenvalues" are all possible bistrings one can measure.
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
    if (failed(verifyInQNodeFunction(getOperation()))) {
        return failure();
    }

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
    if (failed(verifyInQNodeFunction(getOperation()))) {
        return failure();
    }

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

LogicalResult ExpvalOp::verify()
{
    if (failed(verifyInQNodeFunction(getOperation()))) {
        return failure();
    }

    return success();
}

LogicalResult VarianceOp::verify()
{
    if (failed(verifyInQNodeFunction(getOperation()))) {
        return failure();
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

    Block &b = this->getRegion().front();
    if (b.getNumArguments() != this->getArgs().size()) {
        return emitOpError("Adjoint op number of operands must be the same as the number of "
                           "arguments on its block");
    }

    for (auto [operand, bbArg] : llvm::zip_equal(this->getArgs(), b.getArguments())) {
        if (operand.getType() != bbArg.getType()) {
            return emitOpError(
                "Adjoint op operand types must be the same as the argument types on its block");
        }
    }

    return success();
}

//===----------------------------------------------------------------------===//
// Quantum op builders.
//===----------------------------------------------------------------------===//

void OperatorOp::build(OpBuilder &odsBuilder, OperationState &odsState, llvm::StringRef op_name,
                       ValueRange params, ValueRange in_qubits, ValueRange in_ctrl_qubits,
                       ValueRange in_ctrl_values, ValueRange forward_args, bool adjoint,
                       std::optional<int64_t> UID, DictionaryAttr static_data,
                       DictionaryAttr param_map, DictionaryAttr qubit_map)
{
    SmallVector<Type> resultTypes;
    TypeRange qubitTypes = TypeRange(in_qubits);
    TypeRange ctrlQubitTypes = TypeRange(in_ctrl_qubits);
    resultTypes.append(qubitTypes.begin(), qubitTypes.end());
    resultTypes.append(ctrlQubitTypes.begin(), ctrlQubitTypes.end());

    IntegerAttr uidAttr = UID ? odsBuilder.getI64IntegerAttr(*UID) : IntegerAttr();

    build(odsBuilder, odsState,
          /*resultTypes=*/resultTypes,
          /*op_name=*/op_name,
          /*params=*/params,
          /*forward_args=*/forward_args,
          /*in_qubits=*/in_qubits,
          /*in_ctrl_qubits=*/in_ctrl_qubits,
          /*in_ctrl_values=*/in_ctrl_values,
          /*in_qreg=*/Value(),
          /*arr_qubit_indices=*/ValueRange(),
          /*arr_ctrl_indices=*/Value(),
          /*arr_ctrl_values=*/Value(),
          /*adjoint=*/adjoint,
          /*UID=*/uidAttr,
          /*static_data=*/static_data,
          /*param_map=*/param_map,
          /*qubit_map=*/qubit_map);
}

void OperatorOp::build(OpBuilder &odsBuilder, OperationState &odsState, llvm::StringRef op_name,
                       ValueRange params, Value in_qreg, ValueRange arr_qubit_indices,
                       Value arr_ctrl_indices, Value arr_ctrl_values, ValueRange forward_args,
                       bool adjoint, std::optional<int64_t> UID, DictionaryAttr static_data,
                       DictionaryAttr param_map, DictionaryAttr qubit_map)
{
    SmallVector<Type> resultTypes = {in_qreg.getType()};

    IntegerAttr uidAttr = UID ? odsBuilder.getI64IntegerAttr(*UID) : IntegerAttr();

    build(odsBuilder, odsState,
          /*resultTypes=*/resultTypes,
          /*op_name=*/op_name,
          /*params=*/params,
          /*forward_args=*/forward_args,
          /*in_qubits=*/ValueRange(),
          /*in_ctrl_qubits=*/ValueRange(),
          /*in_ctrl_values=*/ValueRange(),
          /*in_qreg=*/in_qreg,
          /*arr_qubit_indices=*/arr_qubit_indices,
          /*arr_ctrl_indices=*/arr_ctrl_indices,
          /*arr_ctrl_values=*/arr_ctrl_values,
          /*adjoint=*/adjoint,
          /*UID=*/uidAttr,
          /*static_data=*/static_data,
          /*param_map=*/param_map,
          /*qubit_map=*/qubit_map);
}

//===----------------------------------------------------------------------===//
// Quantum op printers/parsers.
//===----------------------------------------------------------------------===//

static void printDenseI64ArrayAsList(OpAsmPrinter &p, DenseI64ArrayAttr attr)
{
    p << "[";
    llvm::interleaveComma(attr.asArrayRef(), p, [&](int64_t value) { p << value; });
    p << "]";
}

static void printDictionaryWithDenseI64Lists(OpAsmPrinter &p, DictionaryAttr dict)
{
    p << "{";
    llvm::interleaveComma(dict, p, [&](NamedAttribute namedAttr) {
        p.printKeywordOrString(namedAttr.getName().strref());
        p << " = ";
        if (auto denseArray = dyn_cast<DenseI64ArrayAttr>(namedAttr.getValue())) {
            printDenseI64ArrayAsList(p, denseArray);
        }
        else {
            p << namedAttr.getValue();
        }
    });
    p << "}";
}

void OperatorOp::print(OpAsmPrinter &p)
{
    // 1. Template Name
    p << " \"" << getOpName() << "\"";

    // 2. Variadic Inputs: (%arg0 : type, ...)
    p << "(";
    llvm::interleaveComma(llvm::zip(getParams(), getParams().getTypes()), p,
                          [&](auto pair) { p << std::get<0>(pair) << ": " << std::get<1>(pair); });
    p << ")";

    // 3. Adjoint
    if (getAdjoint()) {
        p << " adj";
    }

    // 4. Qubits
    if (!getInQreg()) {
        p << " qubits(" << getInQubits() << ")";
    }

    // 5. Attribute Dictionary
    SmallVector<StringRef> elidedAttrs = {
        "static_data",        "param_map", "qubit_map", "operandSegmentSizes",
        "resultSegmentSizes", "op_name",   "adjoint",   "UID"};
    p.printOptionalAttrDict(getOperation()->getAttrs(), elidedAttrs);

    p.increaseIndent();

    // 6. Python-only data
    if (getUID()) {
        p.printNewline();
        p << "UID(" << *getUID() << ")";
        if (!getForwardArgs().empty()) {
            p << " forward(";
            llvm::interleaveComma(
                llvm::zip(getForwardArgs(), getForwardArgs().getTypes()), p,
                [&](auto pair) { p << std::get<0>(pair) << ": " << std::get<1>(pair); });
            p << ")";
        }
    }

    // 7. Quantum register
    if (getInQreg()) {
        p.printNewline();
        p << "quregs(" << getInQreg() << ") indices(";
        llvm::interleaveComma(
            llvm::zip(getArrQubitIndices(), getArrQubitIndices().getTypes()), p,
            [&](auto pair) { p << std::get<0>(pair) << ": " << std::get<1>(pair); });
        p << ")";
    }

    // 8. Control qubits
    if (!getInCtrlQubits().empty()) {
        p.printNewline();
        p << "ctrls(" << getInCtrlQubits() << ") ";
        p << "ctrl_vals(" << getInCtrlValues() << ")";
    }
    else if (getArrCtrlIndices()) {
        p.printNewline();
        p << "ctrls(" << getArrCtrlIndices() << ": " << getArrCtrlIndices().getType() << ") ";
        p << "ctrl_vals(" << getArrCtrlValues() << ": " << getArrCtrlValues().getType() << ")";
    }

    // 9. Compilable static data
    if (getStaticDataAttr()) {
        p.printNewline();
        p << "static_data = " << getStaticData();
    }

    // 10. Optional metadata
    if (getParamMapAttr() || getQubitMapAttr()) {
        p.printNewline();
    }
    if (getParamMapAttr()) {
        p << "param_map = ";
        printDictionaryWithDenseI64Lists(p, getParamMap());
    }
    if (getParamMapAttr() && getQubitMapAttr()) {
        p << " ";
    }
    if (getQubitMapAttr()) {
        p << "qubit_map = ";
        printDictionaryWithDenseI64Lists(p, getQubitMap());
    }

    p.decreaseIndent();
}

static ParseResult parseOperandTypePair(OpAsmParser &parser,
                                        OpAsmParser::UnresolvedOperand &operand, Type &type)
{
    return failure(parser.parseOperand(operand) || parser.parseColon() || parser.parseType(type));
}

static ParseResult parseDenseI64ArrayDictionary(OpAsmParser &parser, Builder &builder,
                                                DictionaryAttr &dict)
{
    NamedAttrList attrs;
    auto parseDictEntry = [&]() -> ParseResult {
        StringRef key;
        SmallVector<int64_t> values;
        auto parseArrayValue = [&]() -> ParseResult {
            int64_t value = 0;
            if (parser.parseInteger(value)) {
                return failure();
            }
            values.push_back(value);
            return success();
        };

        if (parser.parseKeyword(&key) || parser.parseEqual() ||
            parser.parseCommaSeparatedList(OpAsmParser::Delimiter::Square, parseArrayValue)) {
            return failure();
        }
        attrs.append(builder.getStringAttr(key), builder.getDenseI64ArrayAttr(values));
        return success();
    };

    if (parser.parseCommaSeparatedList(OpAsmParser::Delimiter::Braces, parseDictEntry)) {
        return failure();
    }
    dict = attrs.getDictionary(parser.getContext());
    return success();
}

ParseResult OperatorOp::parse(OpAsmParser &parser, OperationState &result)
{
    Builder &builder = parser.getBuilder();
    MLIRContext *ctx = parser.getContext();

    // 1. Parse operation name string: "foo"
    std::string opName;
    if (parser.parseString(&opName)) {
        return failure();
    }
    result.addAttribute("op_name", builder.getStringAttr(opName));

    // 2. Parse variadic params: (%arg0: type, ...)
    SmallVector<OpAsmParser::UnresolvedOperand> params;
    SmallVector<Type> paramTypes;
    auto parseParamAndType = [&]() -> ParseResult {
        OpAsmParser::UnresolvedOperand operand;
        Type type;
        if (parseOperandTypePair(parser, operand, type)) {
            return failure();
        }
        params.push_back(operand);
        paramTypes.push_back(type);
        return success();
    };
    if (parser.parseCommaSeparatedList(OpAsmParser::Delimiter::Paren, parseParamAndType)) {
        return failure();
    }

    // 3. Optional adjoint marker.
    if (succeeded(parser.parseOptionalKeyword("adj"))) {
        result.addAttribute("adjoint", builder.getUnitAttr());
    }

    SmallVector<OpAsmParser::UnresolvedOperand> inQubits;
    SmallVector<OpAsmParser::UnresolvedOperand> forwardArgs;
    SmallVector<Type> forwardArgTypes;
    SmallVector<OpAsmParser::UnresolvedOperand> inCtrlQubits;
    SmallVector<OpAsmParser::UnresolvedOperand> inCtrlValues;
    std::optional<OpAsmParser::UnresolvedOperand> inQreg;
    SmallVector<OpAsmParser::UnresolvedOperand> arrQubitIndices;
    SmallVector<Type> arrQubitIndexTypes;
    std::optional<OpAsmParser::UnresolvedOperand> arrCtrlIndices;
    std::optional<Type> arrCtrlIndicesType;
    std::optional<OpAsmParser::UnresolvedOperand> arrCtrlValues;
    std::optional<Type> arrCtrlValuesType;

    // 4. Optional qubit section.
    if (succeeded(parser.parseOptionalKeyword("qubits"))) {
        auto parseQubitOperand = [&]() -> ParseResult {
            OpAsmParser::UnresolvedOperand operand;
            if (parser.parseOperand(operand)) {
                return failure();
            }
            inQubits.push_back(operand);
            return success();
        };
        if (parser.parseCommaSeparatedList(OpAsmParser::Delimiter::Paren, parseQubitOperand)) {
            return failure();
        }
    }

    // 5. Parse optional generic attr-dict that printer emits before trailing sections.
    NamedAttrList genericAttrs;
    if (parser.parseOptionalAttrDict(genericAttrs)) {
        return failure();
    }
    result.addAttributes(genericAttrs);

    // 6. Optional Python-only metadata: UID(...) [forward(...)].
    if (succeeded(parser.parseOptionalKeyword("UID"))) {
        int64_t uid = 0;
        if (parser.parseLParen() || parser.parseInteger(uid) || parser.parseRParen()) {
            return failure();
        }
        result.addAttribute("UID", builder.getI64IntegerAttr(uid));

        if (succeeded(parser.parseOptionalKeyword("forward"))) {
            auto parseForwardArgAndType = [&]() -> ParseResult {
                OpAsmParser::UnresolvedOperand operand;
                Type type;
                if (parseOperandTypePair(parser, operand, type)) {
                    return failure();
                }
                forwardArgs.push_back(operand);
                forwardArgTypes.push_back(type);
                return success();
            };
            if (parser.parseCommaSeparatedList(OpAsmParser::Delimiter::Paren,
                                               parseForwardArgAndType)) {
                return failure();
            }
        }
    }

    // 7. Optional qreg section.
    if (succeeded(parser.parseOptionalKeyword("quregs"))) {
        OpAsmParser::UnresolvedOperand operand;
        if (parser.parseLParen() || parser.parseOperand(operand) || parser.parseRParen()) {
            return failure();
        }
        inQreg = operand;

        auto parseIndexAndType = [&]() -> ParseResult {
            OpAsmParser::UnresolvedOperand indexOperand;
            Type indexType;
            if (parseOperandTypePair(parser, indexOperand, indexType)) {
                return failure();
            }
            arrQubitIndices.push_back(indexOperand);
            arrQubitIndexTypes.push_back(indexType);
            return success();
        };
        if (parser.parseKeyword("indices") ||
            parser.parseCommaSeparatedList(OpAsmParser::Delimiter::Paren, parseIndexAndType)) {
            return failure();
        }
    }

    // 8. Optional controls section (qubit controls or register controls).
    if (succeeded(parser.parseOptionalKeyword("ctrls"))) {
        if (inQreg) {
            OpAsmParser::UnresolvedOperand ctrlIndices;
            Type ctrlIndicesTy;
            if (parser.parseLParen() || parseOperandTypePair(parser, ctrlIndices, ctrlIndicesTy) ||
                parser.parseRParen()) {
                return failure();
            }
            arrCtrlIndices = ctrlIndices;
            arrCtrlIndicesType = ctrlIndicesTy;

            OpAsmParser::UnresolvedOperand ctrlValues;
            Type ctrlValuesTy;
            if (parser.parseKeyword("ctrl_vals") || parser.parseLParen() ||
                parseOperandTypePair(parser, ctrlValues, ctrlValuesTy) || parser.parseRParen()) {
                return failure();
            }
            arrCtrlValues = ctrlValues;
            arrCtrlValuesType = ctrlValuesTy;
        }
        else {
            auto parseCtrlQubit = [&]() -> ParseResult {
                OpAsmParser::UnresolvedOperand operand;
                if (parser.parseOperand(operand)) {
                    return failure();
                }
                inCtrlQubits.push_back(operand);
                return success();
            };
            if (parser.parseCommaSeparatedList(OpAsmParser::Delimiter::Paren, parseCtrlQubit) ||
                parser.parseKeyword("ctrl_vals")) {
                return failure();
            }

            auto parseCtrlValue = [&]() -> ParseResult {
                OpAsmParser::UnresolvedOperand operand;
                if (parser.parseOperand(operand)) {
                    return failure();
                }
                inCtrlValues.push_back(operand);
                return success();
            };
            if (parser.parseCommaSeparatedList(OpAsmParser::Delimiter::Paren, parseCtrlValue)) {
                return failure();
            }
        }
    }

    // 9. Optional inherent metadata blocks.
    if (succeeded(parser.parseOptionalKeyword("static_data"))) {
        DictionaryAttr staticData;
        if (parser.parseEqual() || parser.parseAttribute(staticData)) {
            return failure();
        }
        result.addAttribute("static_data", staticData);
    }
    if (succeeded(parser.parseOptionalKeyword("param_map"))) {
        DictionaryAttr paramMap;
        if (parser.parseEqual() || parseDenseI64ArrayDictionary(parser, builder, paramMap)) {
            return failure();
        }
        result.addAttribute("param_map", paramMap);
    }
    if (succeeded(parser.parseOptionalKeyword("qubit_map"))) {
        DictionaryAttr qubitMap;
        if (parser.parseEqual() || parseDenseI64ArrayDictionary(parser, builder, qubitMap)) {
            return failure();
        }
        result.addAttribute("qubit_map", qubitMap);
    }

    // 10. Resolve operands in segment order.
    if (parser.resolveOperands(params, paramTypes, parser.getCurrentLocation(), result.operands)) {
        return failure();
    }
    if (parser.resolveOperands(forwardArgs, forwardArgTypes, parser.getCurrentLocation(),
                               result.operands)) {
        return failure();
    }
    if (parser.resolveOperands(inQubits, QubitType::get(ctx), result.operands)) {
        return failure();
    }
    if (parser.resolveOperands(inCtrlQubits, QubitType::get(ctx), result.operands)) {
        return failure();
    }
    if (parser.resolveOperands(inCtrlValues, builder.getI1Type(), result.operands)) {
        return failure();
    }
    if (inQreg && parser.resolveOperand(*inQreg, QuregType::get(ctx), result.operands)) {
        return failure();
    }
    if (parser.resolveOperands(arrQubitIndices, arrQubitIndexTypes, parser.getCurrentLocation(),
                               result.operands)) {
        return failure();
    }
    if (arrCtrlIndices &&
        parser.resolveOperand(*arrCtrlIndices, *arrCtrlIndicesType, result.operands)) {
        return failure();
    }
    if (arrCtrlValues &&
        parser.resolveOperand(*arrCtrlValues, *arrCtrlValuesType, result.operands)) {
        return failure();
    }

    // 11. Add inferred results in segment order.
    result.addTypes(SmallVector<Type>(inQubits.size(), QubitType::get(ctx)));
    result.addTypes(SmallVector<Type>(inCtrlQubits.size(), QubitType::get(ctx)));
    if (inQreg) {
        result.addTypes(QuregType::get(ctx));
    }

    // 12. Add explicit segment sizes.
    result.addAttribute(
        "operandSegmentSizes",
        builder.getDenseI32ArrayAttr(
            {static_cast<int32_t>(params.size()), static_cast<int32_t>(forwardArgs.size()),
             static_cast<int32_t>(inQubits.size()), static_cast<int32_t>(inCtrlQubits.size()),
             static_cast<int32_t>(inCtrlValues.size()), inQreg ? 1 : 0,
             static_cast<int32_t>(arrQubitIndices.size()), arrCtrlIndices ? 1 : 0,
             arrCtrlValues ? 1 : 0}));
    result.addAttribute(
        "resultSegmentSizes",
        builder.getDenseI32ArrayAttr({static_cast<int32_t>(inQubits.size()),
                                      static_cast<int32_t>(inCtrlQubits.size()), inQreg ? 1 : 0}));

    return success();
}

//===----------------------------------------------------------------------===//
// Quantum op interface methods.
//===----------------------------------------------------------------------===//

// CustomOp

std::string CustomOp::getOperatorName() { return getGateName().str(); }

mlir::TypeRange CustomOp::getDynamicShape() { return getAllParams().getTypes(); }

std::vector<size_t> CustomOp::getWireLens() { return {getNonCtrlQubitOperands().size()}; }

mlir::DictionaryAttr CustomOp::getStaticData()
{
    return mlir::DictionaryAttr::get(getContext(), {});
}

// MultiRZOp

std::string MultiRZOp::getOperatorName() { return "MultiRZ"; }

mlir::TypeRange MultiRZOp::getDynamicShape() { return getAllParams().getTypes(); }

std::vector<size_t> MultiRZOp::getWireLens() { return {getNonCtrlQubitOperands().size()}; }

mlir::DictionaryAttr MultiRZOp::getStaticData()
{
    return mlir::DictionaryAttr::get(getContext(), {});
}

// PauliRotOp

std::string PauliRotOp::getOperatorName() { return "PauliRot"; }

mlir::TypeRange PauliRotOp::getDynamicShape() { return getAllParams().getTypes(); }

std::vector<size_t> PauliRotOp::getWireLens() { return {getNonCtrlQubitOperands().size()}; }

mlir::DictionaryAttr PauliRotOp::getStaticData()
{
    mlir::MLIRContext *ctx = getContext();
    mlir::NamedAttribute pauliWordEntry = mlir::NamedAttribute(
        mlir::StringAttr::get(ctx, "pauli_word"), mlir::StringAttr::get(ctx, getPauliWord()));
    return mlir::DictionaryAttr::get(ctx, {pauliWordEntry});
}

// PCPhaseOp

std::string PCPhaseOp::getOperatorName() { return "PCPhase"; }

mlir::TypeRange PCPhaseOp::getDynamicShape() { return getAllParams().getTypes(); }

std::vector<size_t> PCPhaseOp::getWireLens() { return {getNonCtrlQubitOperands().size()}; }

mlir::DictionaryAttr PCPhaseOp::getStaticData()
{
    return mlir::DictionaryAttr::get(getContext(), {});
}

// GlobalPhaseOp

std::string GlobalPhaseOp::getOperatorName() { return "GlobalPhase"; }

mlir::TypeRange GlobalPhaseOp::getDynamicShape() { return getAllParams().getTypes(); }

std::vector<size_t> GlobalPhaseOp::getWireLens() { return {0}; }

mlir::DictionaryAttr GlobalPhaseOp::getStaticData()
{
    return mlir::DictionaryAttr::get(getContext(), {});
}

// QubitUnitaryOp

std::string QubitUnitaryOp::getOperatorName() { return "QubitUnitary"; }

mlir::TypeRange QubitUnitaryOp::getDynamicShape() { return getAllParams().getTypes(); }

std::vector<size_t> QubitUnitaryOp::getWireLens() { return {getNonCtrlQubitOperands().size()}; }

mlir::DictionaryAttr QubitUnitaryOp::getStaticData()
{
    return mlir::DictionaryAttr::get(getContext(), {});
}

// OperatorOp

std::string OperatorOp::getOperatorName() { return getOpName().str(); }

mlir::TypeRange OperatorOp::getDynamicShape() { return getParams().getTypes(); }

std::vector<size_t> OperatorOp::getWireLens()
{
    if (getInQreg()) {
        std::vector<size_t> lens;
        // This assumes static lengths!
        // If we enable support for dynamic lengths, we need to update this
        for (mlir::Type indexTensor : getArrQubitIndices().getType()) {
            for (size_t dim : cast<RankedTensorType>(indexTensor).getShape()) {
                lens.push_back(size_t(dim));
            }
        }
        return lens;
    }
    return {getInQubits().size()};
}

std::string OperatorOp::getExtraData()
{
    return getUID().has_value() ? std::to_string(getUID().value()) : "";
}
