// Copyright 2025 Xanadu Quantum Technologies Inc.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "llvm/ADT/TypeSwitch.h" // needed for enums
#include "llvm/Support/LogicalResult.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/Region.h"

#include "PBC/IR/PBCOps.h"
#include "Quantum/IR/QuantumDialect.h"

using namespace mlir;
using namespace catalyst::pbc;

//===----------------------------------------------------------------------===//
// PBC op definitions.
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "PBC/IR/PBCOps.cpp.inc"

//===----------------------------------------------------------------------===//
// PBC op canonicalizers/verifiers helper methods.
//===----------------------------------------------------------------------===//

template <typename OpType> LogicalResult canonicalizePPROp(OpType op, PatternRewriter &rewriter)
{
    bool allIdentity = llvm::all_of(op.getPauliProduct(), [](mlir::Attribute attr) {
        auto pauliStr = llvm::cast<mlir::StringAttr>(attr);
        return pauliStr.getValue() == "I";
    });

    if (allIdentity) {
        rewriter.replaceOp(op, op.getInQubits());
        return mlir::success();
    }
    return mlir::failure();
}

template <typename OpType> LogicalResult verifyPPROp(OpType op)
{
    size_t numPauliProduct = op.getPauliProduct().size();

    if (numPauliProduct == 0) {
        return op.emitOpError("Pauli string must be non-empty");
    }

    if (numPauliProduct != op.getInQubits().size()) {
        return op.emitOpError("Number of qubits must match number of pauli operators");
    }
    return mlir::success();
}

//===----------------------------------------------------------------------===//
// PBC op verifiers.
//===----------------------------------------------------------------------===//

LogicalResult PPRotationOp::verify() { return verifyPPROp(*this); }

LogicalResult PPRotationArbitraryOp::verify() { return verifyPPROp(*this); }

LogicalResult PPMeasurementOp::verify()
{
    if (getInQubits().size() != getPauliProduct().size()) {
        return emitOpError("Number of qubits must match number of pauli operators");
    }
    return mlir::success();
}

LogicalResult SelectPPMeasurementOp::verify()
{
    if (getInQubits().size() != getPauliProduct_0().size() ||
        getInQubits().size() != getPauliProduct_1().size()) {
        return emitOpError("Number of qubits must match number of pauli operators");
    }
    return mlir::success();
}

LogicalResult PrepareStateOp::verify()
{
    auto initState = getInitState();
    if (initState == LogicalInitKind::magic || initState == LogicalInitKind::magic_conj) {
        return emitOpError(
            "Magic state cannot be prepared by this operation, use `FabricateOp` instead.");
    }
    return mlir::success();
}

LogicalResult FabricateOp::verify()
{
    auto initState = getInitState();
    if (initState == LogicalInitKind::zero || initState == LogicalInitKind::one ||
        initState == LogicalInitKind::plus || initState == LogicalInitKind::minus) {
        return emitOpError("Logical state should not be fabricated, use `PrepareStateOp` instead.");
    }
    return mlir::success();
}

LogicalResult PPRotationOp::canonicalize(PPRotationOp op, PatternRewriter &rewriter)
{
    return canonicalizePPROp(op, rewriter);
}

LogicalResult PPRotationArbitraryOp::canonicalize(PPRotationArbitraryOp op,
                                                  PatternRewriter &rewriter)
{
    return canonicalizePPROp(op, rewriter);
}

void LayerOp::build(OpBuilder &builder, OperationState &result, ValueRange inValues,
                    ValueRange outValues, BodyBuilderFn bodyBuilder)
{
    OpBuilder::InsertionGuard guard(builder);
    Location loc = result.location;

    // Set the operands of the layer op
    result.addOperands(inValues);

    // Set the result types of the layer op
    result.addTypes(outValues.getTypes());

    // Create the body region of the layer op
    Region *bodyRegion = result.addRegion();
    Block *bodyBlock = builder.createBlock(bodyRegion);
    for (Value v : inValues) {
        bodyBlock->addArgument(v.getType(), v.getLoc());
    }

    builder.setInsertionPointToStart(bodyBlock);
    bodyBuilder(builder, loc, bodyBlock->getArguments(), outValues);
}

ParseResult LayerOp::parse(OpAsmParser &parser, OperationState &result)
{
    auto &builder = parser.getBuilder();

    // Parse the optional initial iteration arguments.
    SmallVector<OpAsmParser::Argument, 4> regionArgs;
    SmallVector<Type, 4> regionTypes;
    SmallVector<OpAsmParser::UnresolvedOperand, 4> operands;

    // Parse assignment list and results type list.
    if (parser.parseAssignmentList(regionArgs, operands) ||
        parser.parseColonTypeList(regionTypes)) {
        return failure();
    }

    if (regionArgs.size() != regionTypes.size()) {
        return parser.emitError(parser.getNameLoc(),
                                "mismatch in number of region-carried values and defined values");
    }

    // Set block argument types, so that they are known when parsing the region.
    for (auto [iterArg, type] : llvm::zip_equal(regionArgs, regionTypes)) {
        iterArg.type = type;
    }

    // Parse the body region
    Region *body = result.addRegion();
    if (parser.parseRegion(*body, regionArgs)) {
        return failure();
    }

    if (!body->hasOneBlock()) {
        return parser.emitError(parser.getNameLoc(), "LayerOp must have exactly one block");
    }

    // Get last operation in the first block and check if it is a pbc.yield op
    auto &block = body->front();
    auto yieldOp = dyn_cast<YieldOp>(block.getTerminator());
    if (!yieldOp) {
        return parser.emitError(parser.getNameLoc(),
                                "LayerOp must have a pbc.yield op as its terminator");
    }

    result.addTypes(yieldOp.getOperandTypes());

    LayerOp::ensureTerminator(*body, builder, result.location);

    // Resolve input operands. This should be done after parsing the region to
    // catch invalid IR where operands were defined inside of the region.
    for (auto argOperandType : llvm::zip_equal(regionArgs, operands, regionTypes)) {
        Type type = std::get<2>(argOperandType);
        std::get<0>(argOperandType).type = type;
        if (parser.resolveOperand(std::get<1>(argOperandType), type, result.operands)) {
            return failure();
        }
    }

    return success();
}

void LayerOp::print(OpAsmPrinter &p)
{
    // Prints the initialization list in the form of
    // (%inner = %outer, %inner2 = %outer2, <...>)
    // where 'inner' values are assumed to be region arguments and 'outer' values
    // are regular SSA values.
    Block::BlockArgListType blocksArgs = getBody()->getArguments();
    ValueRange initializers = getInitArgs();

    p << '(';
    llvm::interleaveComma(llvm::zip(blocksArgs, initializers), p,
                          [&](auto it) { p << std::get<0>(it) << " = " << std::get<1>(it); });
    p << ')';

    // Print type(s) that corresponds to the initialization list
    if (!getInitArgs().empty())
        p << " : " << getInitArgs().getTypes();
    p << ' ';

    // Print the regions
    p.printRegion(getRegion(),
                  /*printEntryBlockArgs=*/false,
                  /*printBlockTerminators=*/!getInitArgs().empty());
}
