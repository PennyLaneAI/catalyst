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

#include "llvm/ADT/TypeSwitch.h" // needed for generated type parser
#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/DialectImplementation.h" // needed for generated type parser
#include "mlir/Interfaces/FunctionImplementation.h"
#include "mlir/Transforms/InliningUtils.h"

#include "Catalyst/IR/CatalystOps.h"

using namespace mlir;
using namespace catalyst;

#include "Catalyst/IR/CatalystOpsDialect.cpp.inc"

//===----------------------------------------------------------------------===//
// Catalyst dialect.
//===----------------------------------------------------------------------===//

void CatalystDialect::initialize() {
    addTypes<
#define GET_TYPEDEF_LIST
#include "Catalyst/IR/CatalystOpsTypes.cpp.inc"
        >();

    addOperations<
#define GET_OP_LIST
#include "Catalyst/IR/CatalystOps.cpp.inc"
        >();

    declarePromisedInterfaces<bufferization::BufferizableOpInterface, PrintOp, CustomCallOp,
                              CallbackCallOp, CallbackOp, ListPushBlockOp, ListPopBlockOp>();
}

//===----------------------------------------------------------------------===//
// ListPushBlockOp / ListPopBlockOp
//===----------------------------------------------------------------------===//

/// Verify the shaped operand of a block push/pop against the list it is stored in. `name` is used
/// to refer to the operand in diagnostics.
static LogicalResult verifyBlockOperand(Operation *op, llvm::StringRef name, Type blockType,
                                        ArrayListType listType) {
    auto shapedType = cast<ShapedType>(blockType);
    if (shapedType.getElementType() != listType.getElementType()) {
        return op->emitOpError() << "expects the element type of '" << name << "' ("
                                 << shapedType.getElementType()
                                 << ") to match the element type of the list ("
                                 << listType.getElementType() << ")";
    }

    if (isa<RankedTensorType>(blockType)) {
        // A rank-0 tensor holds a single element and is cached with catalyst.list_push/list_pop:
        // it has no rank-1 buffer to collapse to, so bufferization could not lower it.
        if (shapedType.getRank() == 0) {
            return op->emitOpError() << "expects '" << name << "' to have rank at least 1";
        }
        return success();
    }

    // After bufferization the block is the collapsed, contiguous buffer of the tensor. A
    // non-identity layout would either be non-collapsible or alias a strided slice, neither of
    // which the flat list storage can represent.
    auto memrefType = cast<MemRefType>(blockType);
    if (!memrefType.getLayout().isIdentity()) {
        return op->emitOpError() << "expects '" << name << "' to have an identity layout, but got "
                                 << memrefType;
    }
    return success();
}

LogicalResult ListPushBlockOp::verify() {
    return verifyBlockOperand(*this, "elements", getElements().getType(), getList().getType());
}

LogicalResult ListPopBlockOp::verify() {
    Type destinationType = getDestination().getType();
    if (failed(verifyBlockOperand(*this, "destination", destinationType, getList().getType()))) {
        return failure();
    }

    bool hasTensorDestination = isa<RankedTensorType>(destinationType);
    if (hasTensorDestination != static_cast<bool>(getResult())) {
        return emitOpError() << "expects a result if and only if 'destination' is a tensor: a "
                                "tensor destination is returned filled, a memref destination is "
                                "written in place";
    }
    if (getResult() && getResult().getType() != destinationType) {
        return emitOpError() << "expects the result type (" << getResult().getType()
                             << ") to match the type of 'destination' (" << destinationType << ")";
    }
    return success();
}

//===----------------------------------------------------------------------===//
// Catalyst attributes.
//===----------------------------------------------------------------------===//

// Verify a probability value: must be a float attribute in the range [0, 1].
static LogicalResult verifyProbability(Operation *op, llvm::StringRef attrName, Attribute value) {
    auto prob = dyn_cast<FloatAttr>(value);
    if (!prob) {
        return op->emitError() << "'" << attrName << "' must be a float attribute";
    }
    double p = prob.getValueAsDouble();
    if (p < 0.0 || p > 1.0) {
        return op->emitError() << "'" << attrName << "' must be a probability in [0, 1], but got "
                               << p;
    }
    return success();
}

LogicalResult CatalystDialect::verifyOperationAttribute(Operation *op, NamedAttribute attribute) {
    llvm::StringRef name = attribute.getName().strref();

    if (name == EstimatedIterationsAttrName) {
        if (!isa<scf::ForOp, scf::WhileOp>(op)) {
            return op->emitError() << "'" << name << "' is only valid on 'scf.for' or 'scf.while'";
        }
        if (auto intAttr = dyn_cast<IntegerAttr>(attribute.getValue())) {
            if (intAttr.getValue().isNegative()) {
                return op->emitError() << "'" << name << "' must be non-negative, but got "
                                       << intAttr.getValue().getSExtValue();
            }
            return success();
        }
        if (auto floatAttr = dyn_cast<FloatAttr>(attribute.getValue())) {
            if (floatAttr.getValueAsDouble() < 0.0) {
                return op->emitError() << "'" << name << "' must be non-negative, but got "
                                       << floatAttr.getValueAsDouble();
            }
            return success();
        }
        return op->emitError() << "'" << name << "' must be an integer or float attribute";
    }

    if (name == EstimatedProbabilityAttrName) {
        if (!isa<scf::IfOp>(op)) {
            return op->emitError() << "'" << name << "' is only valid on 'scf.if'";
        }
        return verifyProbability(op, name, attribute.getValue());
    }

    if (name == EstimatedProbabilitiesAttrName) {
        auto switchOp = dyn_cast<scf::IndexSwitchOp>(op);
        if (!switchOp) {
            return op->emitError() << "'" << name << "' is only valid on 'scf.index_switch'";
        }

        auto probs = dyn_cast<ArrayAttr>(attribute.getValue());
        if (!probs) {
            return op->emitError() << "'" << name << "' must be an array attribute";
        }

        double sum = 0.0;
        for (Attribute elem : probs) {
            if (failed(verifyProbability(op, name, elem))) {
                return failure();
            }
            sum += cast<FloatAttr>(elem).getValueAsDouble();
        }

        // Allow a small tolerance for floating-point accumulation error.
        if (sum > 1.0 + 1e-10) {
            return op->emitError()
                   << "'" << name << "' entries must sum to at most 1, but got " << sum;
        }

        // There must be exactly one probability per case region.
        size_t numCases = switchOp.getCaseRegions().size();
        if (probs.size() != numCases) {
            return op->emitError() << "'" << name << "' has " << probs.size()
                                   << " entries but the switch has " << numCases << " case(s)";
        }
        return success();
    }

    return success();
}

//===----------------------------------------------------------------------===//
// CallbackOp
//===----------------------------------------------------------------------===//

ParseResult CallbackOp::parse(OpAsmParser &parser, OperationState &result) {
    auto buildFuncType = [](Builder &builder, ArrayRef<Type> argTypes, ArrayRef<Type> results,
                            function_interface_impl::VariadicFlag,
                            std::string &) { return builder.getFunctionType(argTypes, results); };

    return function_interface_impl::parseFunctionOp(
        parser, result, /*allowVariadic=*/false, getFunctionTypeAttrName(result.name),
        buildFuncType, getArgAttrsAttrName(result.name), getResAttrsAttrName(result.name));
}

void CallbackOp::print(OpAsmPrinter &p) {
    function_interface_impl::printFunctionOp(p, *this, /*isVariadic=*/false,
                                             getFunctionTypeAttrName(), getArgAttrsAttrName(),
                                             getResAttrsAttrName());
}

//===----------------------------------------------------------------------===//
// CallbackCallOp
//===----------------------------------------------------------------------===//

CallInterfaceCallable CallbackCallOp::getCallableForCallee() {
    return (*this)->getAttrOfType<SymbolRefAttr>("callee");
}

void CallbackCallOp::setCalleeFromCallable(CallInterfaceCallable callee) {
    (*this)->setAttr("callee", cast<SymbolRefAttr>(callee));
}

Operation::operand_range CallbackCallOp::getArgOperands() { return getInputs(); }

MutableOperandRange CallbackCallOp::getArgOperandsMutable() { return getInputsMutable(); }

//===----------------------------------------------------------------------===//
// LaunchKernelOp
//===----------------------------------------------------------------------===//

CallInterfaceCallable LaunchKernelOp::getCallableForCallee() {
    return (*this)->getAttrOfType<SymbolRefAttr>("callee");
}

void LaunchKernelOp::setCalleeFromCallable(CallInterfaceCallable callee) {
    (*this)->setAttr("callee", cast<SymbolRefAttr>(callee));
}

Operation::operand_range LaunchKernelOp::getArgOperands() { return getInputs(); }

MutableOperandRange LaunchKernelOp::getArgOperandsMutable() { return getInputsMutable(); }

StringAttr LaunchKernelOp::getCalleeModuleName() { return getCallee().getRootReference(); }

StringAttr LaunchKernelOp::getCalleeName() { return getCallee().getLeafReference(); }

//===----------------------------------------------------------------------===//
// Catalyst type definitions.
//===----------------------------------------------------------------------===//

#define GET_TYPEDEF_CLASSES
#include "Catalyst/IR/CatalystOpsTypes.cpp.inc"
