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

#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"

#include "Mitigation/IR/MitigationDialect.h"
#include "Mitigation/IR/MitigationOps.h"

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
    (*this)->setAttr("callee", callee.get<SymbolRefAttr>());
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
