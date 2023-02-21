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

//===----------------------------------------------------------------------===//
// CallOpInterface
//===----------------------------------------------------------------------===//

CallInterfaceCallable GradOp::getCallableForCallee() { return getCalleeAttr(); }

Operation::operand_range GradOp::getArgOperands() { return getOperands(); }

//===----------------------------------------------------------------------===//
// SymbolUserOpInterface
//===----------------------------------------------------------------------===//

LogicalResult GradOp::verifySymbolUses(SymbolTableCollection &symbolTable)
{
    // Check that the callee attribute refers to a valid function.
    func::FuncOp fn = symbolTable.lookupNearestSymbolFrom<func::FuncOp>(*this, getCalleeAttr());
    if (!fn)
        return emitOpError("invalid function name specified: ") << getCalleeAttr();

    // Check that the call operand types match the callee operand types.
    OperandRange fnArgs = getArgOperands();
    FunctionType fnType = fn.getFunctionType();
    if (fnType.getNumInputs() != fnArgs.size())
        return emitOpError("incorrect number of operands for callee, ")
               << "expected " << fnType.getNumInputs() << " but got " << fnArgs.size();

    for (unsigned i = 0; i < fnArgs.size(); ++i)
        if (fnArgs[i].getType() != fnType.getInput(i))
            return emitOpError("operand type mismatch: expected operand type ")
                   << fnType.getInput(i) << ", but provided " << fnArgs[i].getType()
                   << " for operand number " << i;

    // Only differentiation on real numbers is supported.
    const std::vector<size_t> &diffArgIndices = compDiffArgIndices();
    for (size_t idx : diffArgIndices) {
        Type diffArgBaseType = fnArgs[idx].getType();
        if (auto tensorType = diffArgBaseType.dyn_cast<TensorType>())
            diffArgBaseType = tensorType.getElementType();

        if (!diffArgBaseType.isa<FloatType>())
            return emitOpError("invalid numeric base type: callee operand at position ")
                   << idx << " must be floating point to be differentiable";
    }

    const std::vector<Type> &expectedTypes = computeResultTypes(fn, diffArgIndices);

    // Verify the number of results matches the expected gradient shape.
    // The grad output should contain one set of results (equal in size to
    // the number of function results) for each differentiable argument.
    if (getNumResults() != expectedTypes.size())
        return emitOpError("incorrect number of results in the gradient of the callee, ")
               << "expected " << expectedTypes.size() << " results "
               << "but got " << getNumResults();

    // Verify the shape of each result. The numeric type should match the numeric type
    // of the corresponding function result. The shape is given by grouping the differentiated
    // argument shape with the corresponding function result shape.
    TypeRange gradResultTypes = getResultTypes();
    for (unsigned i = 0; i < expectedTypes.size(); i++) {
        if (gradResultTypes[i] != expectedTypes[i])
            return emitOpError("invalid result type: grad result at position ")
                   << i << " must be " << expectedTypes[i] << " but got " << gradResultTypes[i];
    }

    return success();
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

std::vector<uint64_t> GradOp::compDiffArgIndices()
{
    // By default only the first argument is differentiated, otherwise gather indices.
    std::vector<size_t> diffArgIndices{0};
    if (getDiffArgIndices().has_value()) {
        auto range = getDiffArgIndicesAttr().getValues<size_t>();
        diffArgIndices = std::vector<size_t>(range.begin(), range.end());
    }
    return diffArgIndices;
}
