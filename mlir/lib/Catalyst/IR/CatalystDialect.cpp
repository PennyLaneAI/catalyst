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

#include "Catalyst/IR/CatalystDialect.h"
#include "Catalyst/IR/CatalystOps.h"
#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h" // needed for generated type parser
#include "mlir/Interfaces/FunctionImplementation.h"
#include "mlir/Transforms/InliningUtils.h"
#include "llvm/ADT/TypeSwitch.h" // needed for generated type parser

using namespace mlir;
using namespace catalyst;

#include "Catalyst/IR/CatalystOpsDialect.cpp.inc"

//===----------------------------------------------------------------------===//
// Catalyst dialect.
//===----------------------------------------------------------------------===//

void CatalystDialect::initialize()
{
    addTypes<
#define GET_TYPEDEF_LIST
#include "Catalyst/IR/CatalystOpsTypes.cpp.inc"
        >();

    addOperations<
#define GET_OP_LIST
#include "Catalyst/IR/CatalystOps.cpp.inc"
        >();
    declarePromisedInterfaces<bufferization::BufferizableOpInterface, CustomCallOp,
                         PrintOp, CallbackCallOp, CallbackOp>();
}

//===----------------------------------------------------------------------===//
// CallbackOp
//===----------------------------------------------------------------------===//

ParseResult CallbackOp::parse(OpAsmParser &parser, OperationState &result)
{
    auto buildFuncType = [](Builder &builder, ArrayRef<Type> argTypes, ArrayRef<Type> results,
                            function_interface_impl::VariadicFlag,
                            std::string &) { return builder.getFunctionType(argTypes, results); };

    return function_interface_impl::parseFunctionOp(
        parser, result, /*allowVariadic=*/false, getFunctionTypeAttrName(result.name),
        buildFuncType, getArgAttrsAttrName(result.name), getResAttrsAttrName(result.name));
}

void CallbackOp::print(OpAsmPrinter &p)
{
    function_interface_impl::printFunctionOp(p, *this, /*isVariadic=*/false,
                                             getFunctionTypeAttrName(), getArgAttrsAttrName(),
                                             getResAttrsAttrName());
}

//===----------------------------------------------------------------------===//
// CallbackCallOp
//===----------------------------------------------------------------------===//

CallInterfaceCallable CallbackCallOp::getCallableForCallee()
{
    return (*this)->getAttrOfType<SymbolRefAttr>("callee");
}

void CallbackCallOp::setCalleeFromCallable(CallInterfaceCallable callee)
{
    (*this)->setAttr("callee", callee.get<SymbolRefAttr>());
}

Operation::operand_range CallbackCallOp::getArgOperands() { return getInputs(); }

MutableOperandRange CallbackCallOp::getArgOperandsMutable() { return getInputsMutable(); }

//===----------------------------------------------------------------------===//
// Catalyst type definitions.
//===----------------------------------------------------------------------===//

#define GET_TYPEDEF_CLASSES
#include "Catalyst/IR/CatalystOpsTypes.cpp.inc"
