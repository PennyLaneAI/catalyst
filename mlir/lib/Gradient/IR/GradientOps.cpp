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

#include "mlir/Dialect/Func/IR/FuncOps.h"

#include "Gradient/IR/GradientDialect.h"
#include "Gradient/IR/GradientOps.h"
#include "Gradient/Utils/GradientShape.h"

#define GET_OP_CLASSES
#include "Gradient/IR/GradientOps.cpp.inc"

using namespace mlir;
using namespace catalyst::gradient;

// Gradient input checker
LogicalResult verifyGradInputs(
    OpState *op_state,
    func::FuncOp callee,
    ValueRange callee_operands,
    const std::vector<size_t> &diff_arg_indices)
{
    // Check that the call operand types match the callee operand types.
    ValueRange fnArgs = callee_operands;
    FunctionType fnType = callee.getFunctionType();
    if (fnType.getNumInputs() != fnArgs.size())
        return op_state->emitOpError("incorrect number of operands for callee, ")
               << "expected " << fnType.getNumInputs() << " but got " << fnArgs.size();

    for (unsigned i = 0; i < fnArgs.size(); ++i)
        if (fnArgs[i].getType() != fnType.getInput(i))
            return op_state->emitOpError("operand type mismatch: expected operand type ")
                   << fnType.getInput(i) << ", but provided " << fnArgs[i].getType()
                   << " for operand number " << i;

    // Only differentiation on real numbers is supported.
    const std::vector<size_t> &diffArgIndices = diff_arg_indices;
    for (size_t idx : diffArgIndices) {
        Type diffArgBaseType = fnArgs[idx].getType();
        if (auto tensorType = diffArgBaseType.dyn_cast<TensorType>())
            diffArgBaseType = tensorType.getElementType();

        if (!diffArgBaseType.isa<FloatType>())
            return op_state->emitOpError("invalid numeric base type: callee operand at position ")
                   << idx << " must be floating point to be differentiable";
    }
    return success();
}

// Gradient output checker
LogicalResult verifyGradOutputs(
    OpState *op_state,
    func::FuncOp fn,
    const std::vector<size_t> &diff_arg_indices,
    TypeRange result_types)
{
    const std::vector<Type> &expectedTypes = computeResultTypes(fn, diff_arg_indices);

    // Verify the number of results matches the expected gradient shape.
    // The grad output should contain one set of results (equal in size to
    // the number of function results) for each differentiable argument.
    if (result_types.size() != expectedTypes.size())
        return op_state->emitOpError("incorrect number of results in the gradient of the callee, ")
               << "expected " << expectedTypes.size() << " results "
               << "but got " << result_types.size();

    // Verify the shape of each result. The numeric type should match the numeric type
    // of the corresponding function result. The shape is given by grouping the differentiated
    // argument shape with the corresponding function result shape.
    TypeRange gradResultTypes = result_types;
    for (unsigned i = 0; i < expectedTypes.size(); i++) {
        if (gradResultTypes[i] != expectedTypes[i])
            return op_state->emitOpError("invalid result type: grad result at position ")
                   << i << " must be " << expectedTypes[i] << " but got " << gradResultTypes[i];
    }

    return success();
}


//===----------------------------------------------------------------------===//
// GradOp, CallOpInterface
//===----------------------------------------------------------------------===//

CallInterfaceCallable GradOp::getCallableForCallee() { return getCalleeAttr(); }

Operation::operand_range GradOp::getArgOperands() { return getOperands(); }

//===----------------------------------------------------------------------===//
// GradOp, SymbolUserOpInterface
//===----------------------------------------------------------------------===//

LogicalResult GradOp::verifySymbolUses(SymbolTableCollection &symbolTable)
{
    // Check that the callee attribute refers to a valid function.
    auto fn = ({
         auto callee = this->getCalleeAttr();
         func::FuncOp fn = symbolTable.lookupNearestSymbolFrom<func::FuncOp>(
             this->getOperation(), callee);
         if (!fn)
             return this->emitOpError("invalid function name specified: ") << callee;
         fn;
    });

    auto r1 = ::verifyGradInputs(
        this, fn,
        this->getArgOperands(),
        GradOp::compDiffArgIndices(this->getDiffArgIndices()));

    auto r2 = ::verifyGradOutputs(
        this,
        fn,
        GradOp::compDiffArgIndices(this->getDiffArgIndices()),
        this->getResultTypes());

    return success(succeeded(r1) && succeeded(r2));
}

//===----------------------------------------------------------------------===//
// GradOp Extra methods
//===----------------------------------------------------------------------===//

LogicalResult GradOp::verify()
{
    StringRef method = this->getMethod();
    if (method != "fd" && method != "ps" && method != "adj")
        return emitOpError("got invalid differentiation method: ") << method;
    return success();
}

std::vector<size_t> GradOp::compDiffArgIndices(Optional<DenseIntElementsAttr> indices)
{
    // By default only the first argument is differentiated, otherwise gather indices.
    std::vector<size_t> diffArgIndices{0};
    if (indices.has_value()) {
        auto range = indices.value().getValues<size_t>();
        diffArgIndices = std::vector<size_t>(range.begin(), range.end());
    }
    return diffArgIndices;
}

//===----------------------------------------------------------------------===//
// JVPOp, CallOpInterface
//===----------------------------------------------------------------------===//

CallInterfaceCallable JVPOp::getCallableForCallee() { return getCalleeAttr(); }

Operation::operand_range JVPOp::getArgOperands() { return getOperands(); }

//===----------------------------------------------------------------------===//
// JVPOp, SymbolUserOpInterface
//===----------------------------------------------------------------------===//

LogicalResult JVPOp::verifySymbolUses(SymbolTableCollection &symbolTable)
{
    // Check that the callee attribute refers to a valid function.
    func::FuncOp fn = ({
         auto callee = this->getCalleeAttr();
         auto fn = symbolTable.lookupNearestSymbolFrom<func::FuncOp>(
             this->getOperation(), callee);
         if (!fn)
             return this->emitOpError("invalid function name specified: ") << callee;
         fn;
    });

    auto r1 = ::verifyGradInputs(
        this, fn,
        this->getCalleeOperands(),
        GradOp::compDiffArgIndices(this->getDiffArgIndices()));

    /* const std::vector<Type> &expectedTypes = computeResultTypes(fn, diff_arg_indices); */

    /* // Verify the number of results matches the expected gradient shape. */
    /* // The grad output should contain one set of results (equal in size to */
    /* // the number of function results) for each differentiable argument. */
    /* if (result_types.size() != expectedTypes.size()) */
    /*     return op_state->emitOpError("incorrect number of results in the gradient of the callee, ") */
    /*            << "expected " << expectedTypes.size() << " results " */
    /*            << "but got " << result_types.size(); */

    /* // Verify the shape of each result. The numeric type should match the numeric type */
    /* // of the corresponding function result. The shape is given by grouping the differentiated */
    /* // argument shape with the corresponding function result shape. */
    /* TypeRange gradResultTypes = result_types; */
    /* for (unsigned i = 0; i < expectedTypes.size(); i++) { */
    /*     if (gradResultTypes[i] != expectedTypes[i]) */
    /*         return op_state->emitOpError("invalid result type: grad result at position ") */
    /*                << i << " must be " << expectedTypes[i] << " but got " << gradResultTypes[i]; */
    /* } */

    return r1;
}

//===----------------------------------------------------------------------===//
// JVPOp Extra methods
//===----------------------------------------------------------------------===//

std::vector<mlir::Value> JVPOp::getCalleeOperands()
{
    auto diffArgs = GradOp::compDiffArgIndices(this->getDiffArgIndices());
    return std::vector<mlir::Value>(
        this->operand_begin(),
        this->operand_begin() + (this->getNumOperands() - diffArgs.size()));
}

LogicalResult JVPOp::verify()
{
    StringRef method = this->getMethod();
    if (method != "fd" && method != "ps" && method != "adj")
        return emitOpError("got invalid differentiation method: ") << method;
    return success();
}

//===----------------------------------------------------------------------===//
// VJPOp, CallOpInterface
//===----------------------------------------------------------------------===//

CallInterfaceCallable VJPOp::getCallableForCallee() { return getCalleeAttr(); }

Operation::operand_range VJPOp::getArgOperands() { return getOperands(); }

//===----------------------------------------------------------------------===//
// VJPOp, SymbolUserOpInterface
//===----------------------------------------------------------------------===//

LogicalResult VJPOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
    // Check that the callee attribute refers to a valid function.
    auto callee = ({
         auto cattr = this->getCalleeAttr();
         func::FuncOp fn = symbolTable.lookupNearestSymbolFrom<func::FuncOp>(
             this->getOperation(), cattr);
         if (!fn)
             return this->emitOpError("invalid function name specified: ") << cattr;
         fn;
    });

    auto r1 = ::verifyGradInputs(
        this, callee,
        this->getCalleeOperands(),
        GradOp::compDiffArgIndices(this->getDiffArgIndices()));

    /* callee.getFunctionType() */

    return r1;
}

//===----------------------------------------------------------------------===//
// VJPOp Extra methods
//===----------------------------------------------------------------------===//

std::vector<mlir::Value> VJPOp::getCalleeOperands()
{
    return std::vector<mlir::Value>(
        this->operand_begin(),
        this->operand_begin() + (this->getNumOperands() - this->getCalleeResults().size()));
}

LogicalResult VJPOp::verify()
{
    StringRef method = this->getMethod();
    if (method != "fd" && method != "ps" && method != "adj")
        return emitOpError("got invalid differentiation method: ") << method;
    return success();
}
