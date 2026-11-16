// Copyright 2023 Xanadu Quantum Technologies Inc.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "Mitigation/IR/MitigationOps.h"

#include "llvm/ADT/TypeSwitch.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"

#include "Mitigation/IR/MitigationDialect.h"

#include "Mitigation/IR/MitigationEnums.cpp.inc"
#define GET_OP_CLASSES
#include "Mitigation/IR/MitigationOps.cpp.inc"

using namespace mlir;
using namespace catalyst::mitigation;

//===----------------------------------------------------------------------===//
// SymbolUserOpInterface
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// ZneOp, CallOpInterface
//===----------------------------------------------------------------------===//

CallInterfaceCallable ZneOp::getCallableForCallee() { return getCalleeAttr(); }

void ZneOp::setCalleeFromCallable(CallInterfaceCallable callee)
{
    (*this)->setAttr("callee", cast<SymbolRefAttr>(callee));
};

Operation::operand_range ZneOp::getArgOperands() { return getOperands(); }

//===----------------------------------------------------------------------===//
// ZneOp, SymbolUserOpInterface
//===----------------------------------------------------------------------===//

LogicalResult ZneOp::verifySymbolUses(SymbolTableCollection &symbolTable)
{
    // Check that the callee attribute refers to a valid function.
    auto callee = this->getCalleeAttr();
    func::FuncOp fn =
        symbolTable.lookupNearestSymbolFrom<func::FuncOp>(this->getOperation(), callee);
    if (!fn) {
        return this->emitOpError("invalid function name specified: ") << callee;
    }
    return success();
}

//===----------------------------------------------------------------------===//
// ZneOp, getArgOperandsMutable
//===----------------------------------------------------------------------===//

MutableOperandRange ZneOp::getArgOperandsMutable() { return getArgsMutable(); }

//===----------------------------------------------------------------------===//
// RemOp, CallOpInterface
//===----------------------------------------------------------------------===//

CallInterfaceCallable RemOp::getCallableForCallee() { return getCalleeAttr(); }

void RemOp::setCalleeFromCallable(CallInterfaceCallable callee)
{
    (*this)->setAttr("callee", cast<SymbolRefAttr>(callee));
};

Operation::operand_range RemOp::getArgOperands() { return getOperands(); }

//===----------------------------------------------------------------------===//
// RemOp, SymbolUserOpInterface
//===----------------------------------------------------------------------===//

LogicalResult RemOp::verifySymbolUses(SymbolTableCollection &symbolTable)
{
    // Check that the callee attribute refers to a valid function.
    auto callee = this->getCalleeAttr();
    func::FuncOp fn =
        symbolTable.lookupNearestSymbolFrom<func::FuncOp>(this->getOperation(), callee);
    if (!fn) {
        return this->emitOpError("invalid function name specified: ") << callee;
    }

    // The lowering calls the callee with the args and replaces the callee_result group with
    // the call's results, so both must match the callee's signature.
    FunctionType fnType = fn.getFunctionType();
    if (this->getArgs().getTypes() != fnType.getInputs()) {
        return this->emitOpError("argument types do not match callee ")
               << callee << ": expected (" << fnType.getInputs() << "), got ("
               << this->getArgs().getTypes() << ")";
    }
    if (this->getCalleeResult().getTypes() != fnType.getResults()) {
        return this->emitOpError("callee_result types do not match the results of callee ")
               << callee << ": expected (" << fnType.getResults() << "), got ("
               << this->getCalleeResult().getTypes() << ")";
    }
    return success();
}

//===----------------------------------------------------------------------===//
// RemOp, verifier
//===----------------------------------------------------------------------===//

LogicalResult RemOp::verify()
{
    // The lowering replaces the op with the callee results, followed by one all-zeros and one
    // all-ones calibration result when runCalibration is true.
    const size_t numZeros = this->getZeros().size();
    const size_t numOnes = this->getOnes().size();
    if (!this->getRunCalibration()) {
        if (numZeros != 0 || numOnes != 0) {
            return this->emitOpError(
                       "expected no zeros and ones results when runCalibration is false, got ")
                   << numZeros << " zeros and " << numOnes << " ones results";
        }
        return success();
    }
    if (numZeros != 1 || numOnes != 1) {
        return this->emitOpError(
                   "expected one zeros and one ones result when runCalibration is true, got ")
               << numZeros << " zeros and " << numOnes << " ones results";
    }
    Type zerosType = this->getZeros().front().getType();
    Type onesType = this->getOnes().front().getType();
    if (zerosType != onesType) {
        return this->emitOpError("expected zeros and ones results of the same type, got ")
               << zerosType << " and " << onesType;
    }
    return success();
}

//===----------------------------------------------------------------------===//
// RemOp, getArgOperandsMutable
//===----------------------------------------------------------------------===//

MutableOperandRange RemOp::getArgOperandsMutable() { return getArgsMutable(); }
