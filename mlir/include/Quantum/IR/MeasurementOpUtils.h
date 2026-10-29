// Copyright 2026 Xanadu Quantum Technologies Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <cstdint>
#include <string>

#include "llvm/ADT/STLExtras.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpImplementation.h"

#include "QRef/IR/QRefTypes.h"
#include "Quantum/IR/QuantumTypes.h"

namespace catalyst::detail {

// The two measurement operations share their input layout and classical type
// contract. Keep their assembly syntax and verification in sync.
inline mlir::LogicalResult verifyMeasurementInputs(mlir::Operation *op, mlir::ValueRange params,
                                                   mlir::ValueRange qubits, mlir::Value qreg,
                                                   mlir::ValueRange indices,
                                                   mlir::ValueRange mres) {
    if (qreg && !qubits.empty()) {
        return op->emitOpError("must use either qubits or registers, but not both");
    }
    if (!qreg && !indices.empty()) {
        return op->emitOpError("index arrays require a register");
    }

    auto isClassical = [](mlir::Type type) {
        bool classical = true;
        type.walk([&](mlir::Type nested) {
            if (mlir::isa<quantum::QubitType, quantum::QuregType, quantum::ObservableType,
                          qref::QubitType, qref::QuregType>(nested)) {
                classical = false;
            }
        });
        return classical;
    };
    for (mlir::Type type : params.getTypes()) {
        if (!isClassical(type)) {
            return op->emitOpError("params must have classical types, got ") << type;
        }
    }
    for (mlir::Type type : mres.getTypes()) {
        if (!isClassical(type)) {
            return op->emitOpError("mres must have classical types, got ") << type;
        }
    }
    return mlir::success();
}

inline void printMeasurement(mlir::OpAsmPrinter &p, mlir::Operation *op, mlir::StringAttr name,
                             mlir::ValueRange params, mlir::ValueRange qubits, mlir::Value qreg,
                             mlir::ValueRange indices, mlir::DictionaryAttr staticData,
                             bool valueSemantics) {
    p << " ";
    p.printAttributeWithoutType(name);
    auto printTypedOperands = [&](mlir::ValueRange operands) {
        llvm::interleaveComma(operands, p,
                              [&](mlir::Value value) { p << value << ": " << value.getType(); });
    };
    p << "(";
    printTypedOperands(params);
    p << ")";
    if (qreg) {
        p << " quregs(" << qreg;
        if (!valueSemantics) {
            p << ": " << qreg.getType();
        }
        p << ") indices(";
        printTypedOperands(indices);
        p << ")";
    } else {
        p << " qubits(" << qubits << ")";
    }
    if (staticData) {
        p << " static_data = " << staticData;
    }
    p.printOptionalAttrDict(
        op->getAttrs(), {"op_name", "static_data", "operandSegmentSizes", "resultSegmentSizes"});
    p << " : ";
    if (op->getNumResults() == 0) {
        p << "()";
    } else {
        p << op->getResultTypes();
    }
}

inline mlir::ParseResult parseMeasurement(mlir::OpAsmParser &parser, mlir::OperationState &result,
                                          bool valueSemantics) {
    using namespace mlir;
    Builder &builder = parser.getBuilder();
    auto *ctx = parser.getContext();
    auto loc = parser.getCurrentLocation();
    std::string name;
    if (parser.parseString(&name)) {
        return failure();
    }
    result.addAttribute("op_name", builder.getStringAttr(name));

    SmallVector<OpAsmParser::UnresolvedOperand> params, qubits, indices;
    SmallVector<Type> paramTypes, indexTypes, resultTypes;
    auto parseTypedOperands = [&](auto &operands, auto &types) {
        return parser.parseCommaSeparatedList(OpAsmParser::Delimiter::Paren, [&]() -> ParseResult {
            OpAsmParser::UnresolvedOperand operand;
            Type type;
            if (parser.parseOperand(operand) || parser.parseColonType(type)) {
                return failure();
            }
            operands.push_back(operand);
            types.push_back(type);
            return success();
        });
    };
    if (parseTypedOperands(params, paramTypes)) {
        return failure();
    }

    SmallVector<OpAsmParser::UnresolvedOperand> qregs;
    Type qregType = quantum::QuregType::get(ctx);
    if (succeeded(parser.parseOptionalKeyword("qubits"))) {
        if (parser.parseOperandList(qubits, OpAsmParser::Delimiter::Paren)) {
            return failure();
        }
    } else {
        OpAsmParser::UnresolvedOperand qreg;
        if (parser.parseKeyword("quregs") || parser.parseLParen() || parser.parseOperand(qreg)) {
            return failure();
        }
        if (!valueSemantics && parser.parseColonType(qregType)) {
            return failure();
        }
        if (parser.parseRParen() || parser.parseKeyword("indices") ||
            parseTypedOperands(indices, indexTypes)) {
            return failure();
        }
        qregs.push_back(qreg);
    }

    if (succeeded(parser.parseOptionalKeyword("static_data"))) {
        DictionaryAttr staticData;
        if (parser.parseEqual() ||
            parser.parseAttribute(staticData, "static_data", result.attributes)) {
            return failure();
        }
    }
    if (parser.parseOptionalAttrDict(result.attributes) || parser.parseColon()) {
        return failure();
    }
    if (succeeded(parser.parseOptionalLParen())) {
        if (failed(parser.parseOptionalRParen()) &&
            (parser.parseTypeList(resultTypes) || parser.parseRParen())) {
            return failure();
        }
    } else if (parser.parseTypeList(resultTypes)) {
        return failure();
    }

    Type qubitType =
        valueSemantics ? Type(quantum::QubitType::get(ctx)) : Type(qref::QubitType::get(ctx));
    if (parser.resolveOperands(params, paramTypes, loc, result.operands) ||
        parser.resolveOperands(qubits, qubitType, result.operands) ||
        parser.resolveOperands(qregs, qregType, result.operands) ||
        parser.resolveOperands(indices, indexTypes, loc, result.operands)) {
        return failure();
    }
    result.addAttribute("operandSegmentSizes",
                        builder.getDenseI32ArrayAttr({static_cast<int32_t>(params.size()),
                                                      static_cast<int32_t>(qubits.size()),
                                                      static_cast<int32_t>(qregs.size()),
                                                      static_cast<int32_t>(indices.size())}));
    result.addTypes(resultTypes);

    if (valueSemantics) {
        // Quantum results form a suffix: classical results, qubits, register.
        unsigned numClassical = 0, numQubits = 0, numQregs = 0;
        for (Type type : resultTypes) {
            if (isa<quantum::QubitType>(type) && numQregs == 0) {
                ++numQubits;
            } else if (isa<quantum::QuregType>(type)) {
                ++numQregs;
            } else if (numQubits == 0 && numQregs == 0) {
                ++numClassical;
            } else {
                return parser.emitError(
                    loc, "expected classical results followed by qubits or a register");
            }
        }
        result.addAttribute("resultSegmentSizes",
                            builder.getDenseI32ArrayAttr({static_cast<int32_t>(numClassical),
                                                          static_cast<int32_t>(numQubits),
                                                          static_cast<int32_t>(numQregs)}));
    }
    return success();
}

} // namespace catalyst::detail
