// Copyright 2026 Xanadu Quantum Technologies Inc.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "QRef/IR/QRefOps.h"

#include "llvm/ADT/StringSet.h"
#include "llvm/Support/LogicalResult.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/OpImplementation.h"

#include "QRef/IR/QRefDialect.h"
#include "Quantum/IR/QuantumInterfaces.h"

using namespace mlir;
using namespace catalyst::qref;

//===----------------------------------------------------------------------===//
// QRef op definitions.
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "QRef/IR/QRefOps.cpp.inc"

namespace catalyst::qref {

// Utils
static LogicalResult verifyTensorResult(Type ty, int64_t length0, int64_t length1)
{
    ShapedType tensor = cast<ShapedType>(ty);
    if (!tensor.hasStaticShape() || tensor.getShape().size() != 2 ||
        tensor.getShape()[0] != length0 || tensor.getShape()[1] != length1) {
        return failure();
    }

    return success();
}

//===----------------------------------------------------------------------===//
// QRef op canonicalizers.
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

            rewriter.replaceOpWithNewOp<CustomOp>(op, paramsNeg, op.getQubits(), name, false,
                                                  op.getCtrlQubits(), op.getCtrlValues());

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

        rewriter.replaceOpWithNewOp<MultiRZOp>(op, paramNeg, op.getQubits(), nullptr,
                                               op.getCtrlQubits(), op.getCtrlValues());

        return success();
    };
    return failure();
}

LogicalResult PCPhaseOp::canonicalize(PCPhaseOp op, mlir::PatternRewriter &rewriter)
{
    if (op.getAdjoint()) {
        auto paramNeg = mlir::arith::NegFOp::create(rewriter, op.getLoc(), op.getTheta());

        rewriter.replaceOpWithNewOp<PCPhaseOp>(op, paramNeg, op.getDim(), op.getQubits(), nullptr,
                                               op.getCtrlQubits(), op.getCtrlValues());

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

LogicalResult AllocQubitOp::canonicalize(AllocQubitOp allocQb, mlir::PatternRewriter &rewriter)
{
    if (allocQb->use_empty()) {
        rewriter.eraseOp(allocQb);
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

LogicalResult DeallocQubitOp::canonicalize(DeallocQubitOp deallocQb,
                                           mlir::PatternRewriter &rewriter)
{
    if (auto allocQb = dyn_cast_if_present<AllocQubitOp>(deallocQb.getQubit().getDefiningOp())) {
        if (allocQb.getQubit().hasOneUse()) {
            rewriter.eraseOp(deallocQb);
            rewriter.eraseOp(allocQb);
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

OpFoldResult GetOp::fold(FoldAdaptor adaptor)
{
    if (succeeded(foldConstantIndexingOp(*this, adaptor.getIdx()))) {
        return getResult();
    }
    // Returning nullptr tells the caller the op was unchanged.
    return nullptr;
}

//===----------------------------------------------------------------------===//
// QRef op verifiers.
//===----------------------------------------------------------------------===//

static const mlir::StringSet<> validPauliWords = {"X", "Y", "Z", "I"};

LogicalResult AllocOp::verify()
{
    if (!(getNqubits() || getNqubitsAttr().has_value())) {
        return emitOpError() << "expected op to have a non-null allocation size";
    }

    if (getNqubits() && getNqubitsAttr().has_value()) {
        return emitOpError() << "must have a single allocation size";
    }

    QuregType type = getQreg().getType();
    if (auto size = getNqubits()) {
        // Dynamic
        if (!type.isDynamic() || type.getSize().getInt() != mlir::ShapedType::kDynamic) {
            return emitOpError() << "expected result to have dynamic allocation size !qref.qreg<?>";
        }
    }

    else if (auto size = getNqubitsAttr()) {
        // Static
        if (!type.isStatic() || type.getSize().getInt() != size) {
            return emitOpError() << "expected result to have static allocation size !qref.qreg<"
                                 << *size << ">";
        }
    }

    return success();
}

LogicalResult CustomOp::verify()
{
    if (getQubits().size() == 0) {
        return emitOpError("expected op to have at least one qubit");
    }

    return success();
}

LogicalResult OperatorOp::verify()
{
    const bool hasQregMode = static_cast<bool>(getQreg());
    const bool hasQubitMode = !getQubits().empty();

    // At most one mode must be used: explicit qubits or qreg-based addressing.
    if (hasQregMode && hasQubitMode) {
        return emitOpError() << "must use either qubits or registers, but not both";
    }

    if (!getForwardArgs().empty() && !getUID()) {
        return emitOpError() << "forward_args can only be present when UID is provided";
    }

    const bool hasQubitControls = !getCtrlQubits().empty() || !getCtrlValues().empty();
    const bool hasQregControls =
        static_cast<bool>(getArrCtrlIndices()) || static_cast<bool>(getArrCtrlValues());

    if (hasQubitControls && hasQregControls) {
        return emitOpError() << "cannot mix qubit controls (ctrl_qubits/ctrl_values) "
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
        const size_t numTargets = hasQregMode ? getArrQubitIndices().size() : getQubits().size();
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

LogicalResult PauliRotOp::verify()
{
    size_t pauliWordLength = getPauliProduct().size();
    size_t numQubits = getQubits().size();
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
    size_t dim = 1 << getQubits().size();
    if (failed(verifyTensorResult(cast<ShapedType>(getMatrix().getType()), dim, dim))) {
        return emitOpError("The Unitary matrix must be of size 2^(num_qubits) * 2^(num_qubits)");
    }

    return success();
}

LogicalResult AdjointOp::verify()
{
    auto res = this->getRegion().walk(
        [](catalyst::quantum::MeasurementProcess op) { return WalkResult::interrupt(); });

    if (res.wasInterrupted()) {
        return emitOpError("quantum measurements are not allowed in the adjoint regions");
    }

    Block &b = this->getRegion().front();
    if (b.getNumArguments() != 0) {
        return emitOpError("qref.adjoint op must have no arguments on its block");
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
    if (failed(verifyTensorResult(cast<ShapedType>(getMatrix().getType()), dim, dim))) {
        return emitOpError("The Hermitian matrix must be of size 2^(num_qubits) * 2^(num_qubits)");
    }

    return success();
}

//===----------------------------------------------------------------------===//
// Quantum op builders.
//===----------------------------------------------------------------------===//

void OperatorOp::build(OpBuilder &odsBuilder, OperationState &odsState, llvm::StringRef op_name,
                       ValueRange params, ValueRange qubits, ValueRange ctrl_qubits,
                       ValueRange ctrl_values, ValueRange forward_args, bool adjoint,
                       std::optional<int64_t> UID, DictionaryAttr static_data,
                       DictionaryAttr param_map, DictionaryAttr qubit_map)
{
    IntegerAttr uidAttr = UID ? odsBuilder.getI64IntegerAttr(*UID) : IntegerAttr();

    build(odsBuilder, odsState,
          /*op_name=*/op_name,
          /*params=*/params,
          /*forward_args=*/forward_args,
          /*qubits=*/qubits,
          /*ctrl_qubits=*/ctrl_qubits,
          /*ctrl_values=*/ctrl_values,
          /*qreg=*/Value(),
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
                       ValueRange params, Value qreg, ValueRange arr_qubit_indices,
                       Value arr_ctrl_indices, Value arr_ctrl_values, ValueRange forward_args,
                       bool adjoint, std::optional<int64_t> UID, DictionaryAttr static_data,
                       DictionaryAttr param_map, DictionaryAttr qubit_map)
{
    IntegerAttr uidAttr = UID ? odsBuilder.getI64IntegerAttr(*UID) : IntegerAttr();

    build(odsBuilder, odsState,
          /*op_name=*/op_name,
          /*params=*/params,
          /*forward_args=*/forward_args,
          /*qubits=*/ValueRange(),
          /*ctrl_qubits=*/ValueRange(),
          /*ctrl_values=*/ValueRange(),
          /*qreg=*/qreg,
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
    if (!getQreg()) {
        p << " qubits(" << getQubits() << ")";
    }

    // 5. Attribute Dictionary
    SmallVector<StringRef> elidedAttrs = {
        "static_data", "param_map", "qubit_map", "operandSegmentSizes",
        "op_name",     "adjoint",   "UID"};
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
    if (getQreg()) {
        p.printNewline();
        p << "quregs(" << getQreg() << " : " << getQreg().getType() << ") indices(";
        llvm::interleaveComma(
            llvm::zip(getArrQubitIndices(), getArrQubitIndices().getTypes()), p,
            [&](auto pair) { p << std::get<0>(pair) << ": " << std::get<1>(pair); });
        p << ")";
    }

    // 8. Control qubits
    if (!getCtrlQubits().empty()) {
        p.printNewline();
        p << "ctrls(" << getCtrlQubits() << ") ";
        p << "ctrl_vals(" << getCtrlValues() << ")";
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

    SmallVector<OpAsmParser::UnresolvedOperand> qubits;
    SmallVector<OpAsmParser::UnresolvedOperand> forwardArgs;
    SmallVector<Type> forwardArgTypes;
    SmallVector<OpAsmParser::UnresolvedOperand> ctrlQubits;
    SmallVector<OpAsmParser::UnresolvedOperand> ctrlValues;
    std::optional<OpAsmParser::UnresolvedOperand> qreg;
    std::optional<Type> qregType;
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
            qubits.push_back(operand);
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
        Type type;
        if (parser.parseLParen() || parser.parseOperand(operand) || parser.parseColon() ||
            parser.parseType(type) || parser.parseRParen()) {
            return failure();
        }
        qreg = operand;
        qregType = type;

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
        if (qreg) {
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
                ctrlQubits.push_back(operand);
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
                ctrlValues.push_back(operand);
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
    if (parser.resolveOperands(qubits, QubitType::get(ctx), result.operands)) {
        return failure();
    }
    if (parser.resolveOperands(ctrlQubits, QubitType::get(ctx), result.operands)) {
        return failure();
    }
    if (parser.resolveOperands(ctrlValues, builder.getI1Type(), result.operands)) {
        return failure();
    }
    if (qreg && parser.resolveOperand(*qreg, *qregType, result.operands)) {
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

    // 12. Add explicit segment sizes.
    result.addAttribute(
        "operandSegmentSizes",
        builder.getDenseI32ArrayAttr(
            {static_cast<int32_t>(params.size()), static_cast<int32_t>(forwardArgs.size()),
             static_cast<int32_t>(qubits.size()), static_cast<int32_t>(ctrlQubits.size()),
             static_cast<int32_t>(ctrlValues.size()), qreg ? 1 : 0,
             static_cast<int32_t>(arrQubitIndices.size()), arrCtrlIndices ? 1 : 0,
             arrCtrlValues ? 1 : 0}));

    return success();
}

} // namespace catalyst::qref
