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
#include "mlir/IR/FunctionImplementation.h"
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

// ----- QNodeOp

void QNodeOp::build(mlir::OpBuilder &builder, mlir::OperationState &state, llvm::StringRef name,
                    mlir::FunctionType type, llvm::ArrayRef<mlir::NamedAttribute> attrs)
{
    // FunctionOpInterface provides a convenient `build` method that will populate
    // the state of our FuncOp, and create an entry block.
    buildWithEntryBlock(builder, state, name, type, attrs, type.getInputs());
}

mlir::ParseResult QNodeOp::parse(mlir::OpAsmParser &parser, mlir::OperationState &result)
{
    // Dispatch to the FunctionOpInterface provided utility method that parses the
    // function operation.
    auto buildFuncType = [](mlir::Builder &builder, llvm::ArrayRef<mlir::Type> argTypes,
                            llvm::ArrayRef<mlir::Type> results,
                            mlir::function_interface_impl::VariadicFlag,
                            std::string &) { return builder.getFunctionType(argTypes, results); };

    return mlir::function_interface_impl::parseFunctionOp(parser, result, /*allowVariadic=*/false,
                                                          buildFuncType);
}

void QNodeOp::print(mlir::OpAsmPrinter &p)
{
    // Dispatch to the FunctionOpInterface provided utility method that prints the
    // function operation.
    mlir::function_interface_impl::printFunctionOp(p, *this,
                                                   /*isVariadic=*/false);
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

LogicalResult ReturnOp::verify()
{
    auto qnode = cast<QNodeOp>((*this)->getParentOp());
    ArrayRef<Type> results = qnode.getFunctionType().getResults();

    if (getNumOperands() != results.size()) {
        return emitOpError("has ") << getNumOperands() << " operands, but enclosing qnode (@"
                                   << qnode.getName() << ") returns " << results.size();
    }

    for (unsigned i = 0, e = results.size(); i != e; ++i)
        if (getOperand(i).getType() != results[i])
            return emitError() << "type of return operand " << i << " (" << getOperand(i).getType()
                               << ") doesn't match function result type (" << results[i] << ")"
                               << " in qnode @" << qnode.getName();
    return success();
}

LogicalResult CallOp::verifySymbolUses(SymbolTableCollection &symbolTable)
{
    FlatSymbolRefAttr fnAttr = getCalleeAttr();
    if (!fnAttr) {
        return emitOpError("requires a 'callee' symbol reference attribute");
    }
    QNodeOp fn = symbolTable.lookupNearestSymbolFrom<QNodeOp>(*this, fnAttr);
    if (!fn) {
        return emitOpError() << "'" << fnAttr.getValue() << "' does not reference a valid qnode";
    }

    // Verify that the operand and result types match the callee.
    auto fnType = fn.getFunctionType();
    if (fnType.getNumInputs() != getNumOperands()) {
        return emitOpError("incorrect number of operands for callee");
    }

    for (unsigned i = 0, e = fnType.getNumInputs(); i != e; ++i) {
        if (getOperand(i).getType() != fnType.getInput(i)) {
            return emitOpError("operand type mismatch: expected operand type ")
                   << fnType.getInput(i) << ", but provided " << getOperand(i).getType()
                   << " for operand number " << i;
        }
    }

    if (fnType.getNumResults() != getNumResults()) {
        return emitOpError("incorrect number of results for callee");
    }

    for (unsigned i = 0, e = fnType.getNumResults(); i != e; ++i) {
        if (getResult(i).getType() != fnType.getResult(i)) {
            auto diag = emitOpError("result type mismatch at index ") << i;
            diag.attachNote() << "      op result types: " << getResultTypes();
            diag.attachNote() << "function result types: " << fnType.getResults();
            return diag;
        }
    }

    return success();
}

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

LogicalResult QubitUnitaryOp::verify()
{
    size_t dim = std::pow(2, getInQubits().size());
    if (failed(verifyTensorResult(getMatrix().getType().cast<ShapedType>(), dim, dim))) {
        return emitOpError("The Unitary matrix must be of size 2^(num_qubits) * 2^(num_qubits)");
    }

    return success();
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

    if (!((bool)getSamples() ^ (bool)getInData())) {
        return emitOpError("either tensors must be returned or memrefs must be used as inputs");
    }

    Type toVerify = getSamples() ? getSamples().getType() : getInData().getType();
    if (getObs().getDefiningOp<ComputationalBasisOp>() &&
        failed(verifyTensorResult(toVerify, getShots(), numQubits))) {
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
    size_t numQubits = 0;
    if (failed(verifyObservable(getObs(), &numQubits))) {
        return emitOpError("observable must be locally defined");
    }

    if (!numQubits) {
        return emitOpError("only computational basis observables are supported");
    }

    if (!(bool)getProbabilities() ^ (bool)getStateIn()) {
        return emitOpError("either tensors must be returned or memrefs must be used as inputs");
    }

    Type toVerify =
        getProbabilities() ? (Type)getProbabilities().getType() : (Type)getStateIn().getType();
    size_t dim = std::pow(2, numQubits);
    if (failed(verifyTensorResult(toVerify.cast<ShapedType>(), dim))) {
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

    if (!(bool)getState() ^ (bool)getStateIn()) {
        return emitOpError("either tensors must be returned or memrefs must be used as inputs");
    }

    Type toVerify = getState() ? (Type)getState().getType() : (Type)getStateIn().getType();
    size_t dim = std::pow(2, numQubits);
    if (failed(verifyTensorResult(toVerify.cast<ShapedType>(), dim))) {
        return emitOpError("return tensor must have static length equal to 2^(number of qubits)");
    }

    return success();
}
