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

#include "PBC/Utils/PBCWireIndex.h"

#include <algorithm>

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "stablehlo/dialect/StablehloOps.h"

#include "PBC/Utils/PBCClassicalEvaluation.h"

using namespace mlir;

namespace catalyst::pbc {
namespace {

constexpr int64_t maxElements = 100000;
constexpr unsigned maxCachedValues = 100000;
constexpr unsigned maxEvaluationSteps = 1000000;
using Integers = llvm::SmallVector<int64_t>;
using Elements = llvm::SmallVector<Attribute>;

Type elementType(Type type) {
    if (auto tensor = dyn_cast<RankedTensorType>(type)) {
        return tensor.getElementType();
    }
    return type;
}

std::optional<int64_t> elementCount(Type type) {
    if (isa<IntegerType, IndexType, FloatType>(type)) {
        return 1;
    }
    auto tensor = dyn_cast<RankedTensorType>(type);
    if (!tensor || !tensor.hasStaticShape() ||
        !isa<IntegerType, IndexType, FloatType, ComplexType>(tensor.getElementType()) ||
        tensor.getNumElements() > maxElements) {
        return std::nullopt;
    }
    return tensor.getNumElements();
}

std::optional<Elements> elements(Attribute attr) {
    if (isa_and_nonnull<IntegerAttr, FloatAttr>(attr)) {
        return Elements{attr};
    }
    if (auto dense = dyn_cast_or_null<DenseElementsAttr>(attr)) {
        if (!elementCount(dense.getType())) {
            return std::nullopt;
        }
        Elements result;
        for (Attribute element : dense.getValues<Attribute>()) {
            result.push_back(element);
        }
        return result;
    }
    return std::nullopt;
}

Attribute elementsConstant(Type type, llvm::ArrayRef<Attribute> values) {
    auto count = elementCount(type);
    if (!count || *count != static_cast<int64_t>(values.size())) {
        return {};
    }
    if (auto tensor = dyn_cast<RankedTensorType>(type)) {
        return DenseElementsAttr::get(tensor, values);
    }
    return values.front();
}

std::optional<int64_t> integerValue(const llvm::APInt &value, Type type) {
    auto integer = dyn_cast<IntegerType>(type);
    if (integer && (integer.isUnsigned() || (integer.isSignless() && integer.getWidth() == 1))) {
        if (value.getActiveBits() > 63) {
            return std::nullopt;
        }
        return static_cast<int64_t>(value.getZExtValue());
    }
    if (!value.isSignedIntN(64)) {
        return std::nullopt;
    }
    return value.getSExtValue();
}

std::optional<Integers> integers(Attribute attr) {
    if (auto scalar = dyn_cast_or_null<IntegerAttr>(attr)) {
        if (auto value = integerValue(scalar.getValue(), scalar.getType())) {
            return Integers{*value};
        }
    }
    if (auto tensor = dyn_cast_or_null<DenseIntElementsAttr>(attr)) {
        auto count = elementCount(tensor.getType());
        if (!count) {
            return std::nullopt;
        }
        Integers result;
        result.reserve(*count);
        for (auto value : tensor.getValues<llvm::APInt>()) {
            auto resolved = integerValue(value, tensor.getElementType());
            if (!resolved) {
                return std::nullopt;
            }
            result.push_back(*resolved);
        }
        return result;
    }
    return std::nullopt;
}

bool fits(int64_t value, Type type) {
    auto integer = dyn_cast<IntegerType>(type);
    if (!integer) {
        return isa<IndexType>(type);
    }
    if (integer.isUnsigned() || (integer.isSignless() && integer.getWidth() == 1)) {
        return value >= 0 &&
               llvm::APInt(64, static_cast<uint64_t>(value)).isIntN(integer.getWidth());
    }
    return llvm::APInt(64, static_cast<uint64_t>(value), true).isSignedIntN(integer.getWidth());
}

Attribute constant(Type type, llvm::ArrayRef<int64_t> values) {
    auto count = elementCount(type);
    if (!count || *count != static_cast<int64_t>(values.size())) {
        return {};
    }
    Type element = elementType(type);
    for (int64_t value : values) {
        if (!fits(value, element)) {
            return {};
        }
    }
    if (isa<IntegerType, IndexType>(type)) {
        return IntegerAttr::get(type, values.front());
    }
    unsigned width = isa<IndexType>(element) ? 64 : cast<IntegerType>(element).getWidth();
    llvm::SmallVector<llvm::APInt> elements;
    elements.reserve(values.size());
    for (int64_t value : values) {
        elements.emplace_back(width, static_cast<uint64_t>(value), value < 0);
    }
    return DenseIntElementsAttr::get(cast<RankedTensorType>(type), elements);
}

unsigned bitWidth(Type type) {
    Type element = elementType(type);
    return isa<IndexType>(element) ? 64 : cast<IntegerType>(element).getWidth();
}

Integers coordinates(int64_t flat, llvm::ArrayRef<int64_t> shape) {
    Integers result(shape.size());
    for (size_t dim = shape.size(); dim-- > 0;) {
        result[dim] = flat % shape[dim];
        flat /= shape[dim];
    }
    return result;
}

int64_t flatIndex(llvm::ArrayRef<int64_t> index, llvm::ArrayRef<int64_t> shape) {
    int64_t result = 0;
    for (auto [coordinate, size] : llvm::zip(index, shape)) {
        result = result * size + coordinate;
    }
    return result;
}

std::optional<int64_t> unaryInteger(StringRef name, int64_t input, Type type) {
    llvm::APInt value(bitWidth(type), static_cast<uint64_t>(input), input < 0);
    if (name == "stablehlo.not") {
        value.flipAllBits();
    } else if (name == "stablehlo.negate" || name == "arith.negsi") {
        value = -value;
    } else if (name == "stablehlo.abs") {
        if (value.isMinSignedValue()) {
            return std::nullopt;
        }
        if (value.isNegative()) {
            value = -value;
        }
    } else if (name == "stablehlo.popcnt") {
        return value.popcount();
    } else {
        return std::nullopt;
    }
    return integerValue(value, elementType(type));
}

std::optional<int64_t> binaryInteger(Operation *op, int64_t left, int64_t right) {
    StringRef name = op->getName().getStringRef();
    if (auto flags = op->getAttrOfType<arith::IntegerOverflowFlagsAttr>("overflowFlags");
        flags && flags.getValue() != arith::IntegerOverflowFlags::none) {
        // A violated no-wrap promise produces poison, not a wrapped integer.
        return std::nullopt;
    }
    Type type = elementType(op->getOperand(0).getType());
    unsigned width = bitWidth(type);
    if ((name == "arith.shli" || name == "arith.shrui" || name == "arith.shrsi") &&
        (right < 0 || static_cast<uint64_t>(right) >= width)) {
        return std::nullopt;
    }
    llvm::APInt a(width, static_cast<uint64_t>(left), left < 0);
    llvm::APInt b(width, static_cast<uint64_t>(right), right < 0);
    llvm::APInt result(width, 0);
    if (name == "stablehlo.add" || name == "arith.addi") {
        // StableHLO treats i1 as a boolean: its addition is logical OR.
        result = name == "stablehlo.add" && width == 1 ? a | b : a + b;
    } else if (name == "stablehlo.subtract" || name == "arith.subi") {
        result = a - b;
    } else if (name == "stablehlo.multiply" || name == "arith.muli") {
        result = a * b;
    } else if (name == "stablehlo.and" || name == "arith.andi") {
        result = a & b;
    } else if (name == "stablehlo.or" || name == "arith.ori") {
        result = a | b;
    } else if (name == "stablehlo.xor" || name == "arith.xori") {
        result = a ^ b;
    } else if (name == "stablehlo.shift_left" || name == "arith.shli") {
        result = right < 0 || static_cast<uint64_t>(right) >= width ? llvm::APInt(width, 0)
                                                                    : a.shl(right);
    } else if (name == "stablehlo.shift_right_logical" || name == "arith.shrui") {
        result = right < 0 || static_cast<uint64_t>(right) >= width ? llvm::APInt(width, 0)
                                                                    : a.lshr(right);
    } else if (name == "stablehlo.shift_right_arithmetic" || name == "arith.shrsi") {
        result = right < 0 || static_cast<uint64_t>(right) >= width
                     ? (a.isNegative() ? llvm::APInt::getAllOnes(width) : llvm::APInt(width, 0))
                     : a.ashr(right);
    } else if (name == "stablehlo.maximum" || name == "stablehlo.minimum") {
        auto integer = dyn_cast<IntegerType>(type);
        bool isUnsigned =
            integer && (integer.isUnsigned() || (integer.isSignless() && integer.getWidth() == 1));
        bool smaller = isUnsigned ? a.ult(b) : a.slt(b);
        result = smaller == (name == "stablehlo.minimum") ? a : b;
    } else if (auto comparison = dyn_cast<stablehlo::CompareOp>(op)) {
        auto comparisonType = comparison.getCompareType();
        auto integer = dyn_cast<IntegerType>(type);
        bool inferredUnsigned = integer && (integer.isUnsigned() || integer.getWidth() == 1);
        bool isUnsigned = comparisonType && *comparisonType != stablehlo::ComparisonType::NOTYPE
                              ? *comparisonType == stablehlo::ComparisonType::UNSIGNED
                              : inferredUnsigned;
        bool less = isUnsigned ? a.ult(b) : a.slt(b);
        bool equal = a == b;
        switch (comparison.getComparisonDirection()) {
        case stablehlo::ComparisonDirection::EQ:
            return equal;
        case stablehlo::ComparisonDirection::NE:
            return !equal;
        case stablehlo::ComparisonDirection::LT:
            return less;
        case stablehlo::ComparisonDirection::LE:
            return less || equal;
        case stablehlo::ComparisonDirection::GT:
            return !less && !equal;
        case stablehlo::ComparisonDirection::GE:
            return !less;
        }
    } else if (name == "arith.cmpi") {
        auto predicate = op->getAttrOfType<IntegerAttr>("predicate");
        if (!predicate) {
            return std::nullopt;
        }
        switch (predicate.getInt()) {
        case 0:
            return a == b;
        case 1:
            return a != b;
        case 2:
            return a.slt(b);
        case 3:
            return a.sle(b);
        case 4:
            return a.sgt(b);
        case 5:
            return a.sge(b);
        case 6:
            return a.ult(b);
        case 7:
            return a.ule(b);
        case 8:
            return a.ugt(b);
        case 9:
            return a.uge(b);
        default:
            return std::nullopt;
        }
    } else {
        return std::nullopt;
    }
    return integerValue(result, type);
}

bool orderIndependentReduction(Operation *reducer, int64_t initial,
                               llvm::ArrayRef<int64_t> inputs) {
    StringRef name = reducer->getName().getStringRef();
    if (name != "stablehlo.add" && name != "stablehlo.multiply" && name != "stablehlo.and" &&
        name != "stablehlo.or" && name != "stablehlo.xor" && name != "stablehlo.minimum" &&
        name != "stablehlo.maximum") {
        return false;
    }
    // StableHLO may choose any reduction tree and insert the initial value more
    // than once. These integer operations are associative; require that the
    // initial value is neutral for every concrete input as well.
    return llvm::all_of(inputs, [&](int64_t value) {
        auto combined = binaryInteger(reducer, initial, value);
        return combined && *combined == value;
    });
}

} // namespace

void WireIndexResolver::bind(Value target, Value source, const WireIndexResolver &from) {
    Attribute value = from.resolveConstant(source);
    // Resolving first also handles binding from this resolver itself.
    bindConstant(target, value);
}

void WireIndexResolver::invalidate(Value value) {
    Region *region = value.getParentRegion();
    for (auto it = cache.begin(); it != cache.end();) {
        auto entry = it++;
        if (region->isAncestor(entry->first.getParentRegion())) {
            cache.erase(entry);
        }
    }
}

void WireIndexResolver::bindConstant(Value target, Attribute value) {
    // SSA values outside this region cannot depend directly on its values.
    // Results leaving a region are explicitly rebound by the scheduler.
    invalidate(target);
    bindings[target] = value;
}

void WireIndexResolver::forget(Value value) {
    invalidate(value);
    bindings.erase(value);
}

Attribute WireIndexResolver::resolveConstant(Value value) const {
    PendingValues pending;
    Attribute result = resolve(value, pending);
    if (pending.empty()) {
        return result;
    }
    PendingValues stack{value};
    llvm::DenseSet<Value> active{value};
    unsigned steps = 0;
    while (!stack.empty()) {
        if (++steps > maxEvaluationSteps || cache.size() + active.size() > maxCachedValues) {
            return {};
        }
        Value current = stack.back();
        pending.clear();
        result = evaluate(current, pending);
        if (!result && pending.empty()) {
            result = evaluatePBCClassicalValue(
                current, [&](Value operand) { return resolve(operand, pending); });
        }
        if (!pending.empty()) {
            // Resolve only requested dependencies. In particular, a known
            // select condition must not evaluate its unused branch.
            Value dependency = pending.front();
            if (!active.insert(dependency).second) {
                return {};
            }
            stack.push_back(dependency);
            continue;
        }
        cache[current] = result;
        active.erase(current);
        stack.pop_back();
    }
    return cache.lookup(value);
}

std::optional<int64_t> WireIndexResolver::resolveIndex(Value value) const {
    if (!value) {
        return std::nullopt;
    }
    if (auto tensor = dyn_cast<RankedTensorType>(value.getType());
        tensor && tensor.getRank() != 0) {
        return std::nullopt;
    }
    auto result = integers(resolveConstant(value));
    return result && result->size() == 1 ? std::optional<int64_t>(result->front()) : std::nullopt;
}

Attribute WireIndexResolver::resolve(Value value, PendingValues &pending) const {
    if (!value || !elementCount(value.getType())) {
        return {};
    }
    if (auto it = bindings.find(value); it != bindings.end()) {
        return it->second;
    }
    if (auto it = cache.find(value); it != cache.end()) {
        return it->second;
    }
    if (!value.getDefiningOp() || cache.size() >= maxCachedValues) {
        return {};
    }
    if (!llvm::is_contained(pending, value)) {
        pending.push_back(value);
    }
    return {};
}

Attribute WireIndexResolver::evaluate(Value value, PendingValues &pending) const {
    Operation *op = value.getDefiningOp();
    if (!op) {
        return {};
    }
    llvm::StringRef name = op->getName().getStringRef();
    if (name == "arith.constant" || name == "stablehlo.constant") {
        Attribute attr = op->getAttr("value");
        return elements(attr) ? attr : Attribute{};
    }
    if (name == "tensor.from_elements") {
        Elements values;
        for (Value operand : op->getOperands()) {
            auto item = elements(resolve(operand, pending));
            if (!item || item->size() != 1) {
                return {};
            }
            values.push_back(item->front());
        }
        return elementsConstant(value.getType(), values);
    }
    if (name == "stablehlo.iota") {
        auto type = dyn_cast<RankedTensorType>(value.getType());
        auto axis = op->getAttrOfType<IntegerAttr>("iota_dimension");
        if (!type || !isa<IntegerType, IndexType>(type.getElementType()) || !axis ||
            axis.getInt() < 0 || axis.getInt() >= type.getRank()) {
            return {};
        }
        Integers values;
        for (int64_t i = 0; i < type.getNumElements(); ++i) {
            values.push_back(coordinates(i, type.getShape())[axis.getInt()]);
        }
        return constant(type, values);
    }
    if (name == "stablehlo.concatenate") {
        auto type = dyn_cast<RankedTensorType>(value.getType());
        auto axis = op->getAttrOfType<IntegerAttr>("dimension");
        if (!type || !axis || axis.getInt() < 0 || axis.getInt() >= type.getRank()) {
            return {};
        }
        llvm::SmallVector<Elements> inputs;
        for (Value operand : op->getOperands()) {
            auto input = elements(resolve(operand, pending));
            if (!input) {
                return {};
            }
            inputs.push_back(std::move(*input));
        }
        Elements values;
        for (int64_t i = 0; i < type.getNumElements(); ++i) {
            Integers position = coordinates(i, type.getShape());
            for (auto [j, operand] : llvm::enumerate(op->getOperands())) {
                auto inputType = cast<RankedTensorType>(operand.getType());
                if (position[axis.getInt()] < inputType.getDimSize(axis.getInt())) {
                    values.push_back(inputs[j][flatIndex(position, inputType.getShape())]);
                    break;
                }
                position[axis.getInt()] -= inputType.getDimSize(axis.getInt());
            }
        }
        return elementsConstant(type, values);
    }
    if (name == "stablehlo.select" || name == "arith.select") {
        auto condition = integers(resolve(op->getOperand(0), pending));
        if (!condition) {
            return {};
        }
        if (llvm::all_of(*condition, [](int64_t item) { return item != 0; })) {
            return resolve(op->getOperand(1), pending);
        }
        if (llvm::all_of(*condition, [](int64_t item) { return item == 0; })) {
            return resolve(op->getOperand(2), pending);
        }
        auto yes = elements(resolve(op->getOperand(1), pending));
        auto no = elements(resolve(op->getOperand(2), pending));
        if (!yes || !no || yes->size() != no->size() || yes->size() != condition->size()) {
            return {};
        }
        for (size_t i = 0; i < yes->size(); ++i) {
            (*yes)[i] = (*condition)[i] ? (*yes)[i] : (*no)[i];
        }
        return elementsConstant(value.getType(), *yes);
    }
    if (op->getNumOperands() == 0) {
        return {};
    }
    auto input = elements(resolve(op->getOperand(0), pending));
    if (!input) {
        return {};
    }
    if (name == "tensor.extract") {
        auto tensor = dyn_cast<RankedTensorType>(op->getOperand(0).getType());
        if (!tensor || op->getNumOperands() != tensor.getRank() + 1) {
            return {};
        }
        int64_t index = 0;
        for (int64_t dim = 0; dim < tensor.getRank(); ++dim) {
            auto indices = integers(resolve(op->getOperand(dim + 1), pending));
            if (!indices || indices->size() != 1) {
                return {};
            }
            int64_t coordinate = indices->front();
            if (coordinate < 0 || coordinate >= tensor.getDimSize(dim)) {
                return {};
            }
            index = index * tensor.getDimSize(dim) + coordinate;
        }
        if (index < 0 || index >= static_cast<int64_t>(input->size())) {
            return {};
        }
        return elementsConstant(value.getType(), {(*input)[index]});
    }
    if (name == "stablehlo.slice") {
        auto inputType = cast<RankedTensorType>(op->getOperand(0).getType());
        auto outputType = cast<RankedTensorType>(value.getType());
        auto start = op->getAttrOfType<DenseI64ArrayAttr>("start_indices");
        auto limit = op->getAttrOfType<DenseI64ArrayAttr>("limit_indices");
        auto strides = op->getAttrOfType<DenseI64ArrayAttr>("strides");
        if (!start || !limit || !strides || start.size() != inputType.getRank() ||
            limit.size() != start.size() || strides.size() != start.size()) {
            return {};
        }
        for (int64_t dim = 0; dim < inputType.getRank(); ++dim) {
            if (start[dim] < 0 || limit[dim] < start[dim] ||
                limit[dim] > inputType.getDimSize(dim) || strides[dim] <= 0) {
                return {};
            }
        }
        Elements values;
        for (int64_t i = 0; i < outputType.getNumElements(); ++i) {
            Integers position = coordinates(i, outputType.getShape());
            for (int64_t dim = 0; dim < inputType.getRank(); ++dim) {
                position[dim] = start[dim] + position[dim] * strides[dim];
            }
            values.push_back((*input)[flatIndex(position, inputType.getShape())]);
        }
        return elementsConstant(value.getType(), values);
    }
    if (name == "stablehlo.dynamic_slice") {
        auto inputType = cast<RankedTensorType>(op->getOperand(0).getType());
        auto outputType = cast<RankedTensorType>(value.getType());
        if (op->getNumOperands() != inputType.getRank() + 1) {
            return {};
        }
        Integers starts;
        for (int64_t dim = 0; dim < inputType.getRank(); ++dim) {
            auto start = integers(resolve(op->getOperand(dim + 1), pending));
            if (!start || start->size() != 1) {
                return {};
            }
            starts.push_back(std::clamp(start->front(), int64_t{0},
                                        inputType.getDimSize(dim) - outputType.getDimSize(dim)));
        }
        Elements values;
        for (int64_t i = 0; i < outputType.getNumElements(); ++i) {
            Integers position = coordinates(i, outputType.getShape());
            for (int64_t dim = 0; dim < inputType.getRank(); ++dim) {
                position[dim] += starts[dim];
            }
            values.push_back((*input)[flatIndex(position, inputType.getShape())]);
        }
        return elementsConstant(outputType, values);
    }
    if (name == "stablehlo.broadcast_in_dim") {
        auto inputType = cast<RankedTensorType>(op->getOperand(0).getType());
        auto outputType = cast<RankedTensorType>(value.getType());
        auto dimensions = op->getAttrOfType<DenseI64ArrayAttr>("broadcast_dimensions");
        if (!dimensions || dimensions.size() != inputType.getRank()) {
            return {};
        }
        Elements values;
        for (int64_t i = 0; i < outputType.getNumElements(); ++i) {
            Integers outputPosition = coordinates(i, outputType.getShape());
            Integers inputPosition;
            for (int64_t dim = 0; dim < inputType.getRank(); ++dim) {
                inputPosition.push_back(
                    inputType.getDimSize(dim) == 1 ? 0 : outputPosition[dimensions[dim]]);
            }
            values.push_back((*input)[flatIndex(inputPosition, inputType.getShape())]);
        }
        return elementsConstant(outputType, values);
    }
    if (name == "stablehlo.reduce" && op->getNumOperands() == 2 && op->getNumResults() == 1 &&
        op->getNumRegions() == 1) {
        auto inputType = cast<RankedTensorType>(op->getOperand(0).getType());
        auto outputType = cast<RankedTensorType>(value.getType());
        auto dimensions = op->getAttrOfType<DenseI64ArrayAttr>("dimensions");
        auto inputIntegers = integers(resolve(op->getOperand(0), pending));
        auto initial = integers(resolve(op->getOperand(1), pending));
        Region &body = op->getRegion(0);
        if (!dimensions || !inputIntegers || !initial || initial->size() != 1 ||
            !body.hasOneBlock() || body.front().getOperations().size() != 2) {
            return {};
        }
        Operation &reducer = body.front().front();
        if (reducer.getNumOperands() != 2 || reducer.getNumResults() != 1 ||
            reducer.getOperand(0) != body.front().getArgument(0) ||
            reducer.getOperand(1) != body.front().getArgument(1) ||
            body.front().getTerminator()->getNumOperands() != 1 ||
            body.front().getTerminator()->getOperand(0) != reducer.getResult(0) ||
            !orderIndependentReduction(&reducer, initial->front(), *inputIntegers)) {
            return {};
        }
        Integers values(outputType.getNumElements(), initial->front());
        for (int64_t i = 0; i < inputType.getNumElements(); ++i) {
            Integers inputPosition = coordinates(i, inputType.getShape());
            Integers outputPosition;
            for (int64_t dim = 0; dim < inputType.getRank(); ++dim) {
                if (!llvm::is_contained(dimensions.asArrayRef(), dim)) {
                    outputPosition.push_back(inputPosition[dim]);
                }
            }
            int64_t offset = flatIndex(outputPosition, outputType.getShape());
            auto reduced = binaryInteger(&reducer, values[offset], (*inputIntegers)[i]);
            if (!reduced) {
                return {};
            }
            values[offset] = *reduced;
        }
        return constant(outputType, values);
    }
    if (name == "stablehlo.reshape" || name == "tensor.cast") {
        return elementsConstant(value.getType(), *input);
    }
    auto inputIntegers = integers(resolve(op->getOperand(0), pending));
    if (!inputIntegers || !isa<IntegerType, IndexType>(elementType(value.getType()))) {
        return {};
    }
    if (name == "stablehlo.convert" || name == "arith.index_cast" || name == "arith.index_castui" ||
        name == "arith.extsi" || name == "arith.extui" || name == "arith.trunci") {
        Type sourceType = elementType(op->getOperand(0).getType());
        auto sourceInteger = dyn_cast<IntegerType>(sourceType);
        for (int64_t item : *inputIntegers) {
            if ((name == "arith.index_castui" || name == "arith.extui") && item < 0) {
                return {};
            }
            if ((name == "arith.index_cast" || name == "arith.extsi") && sourceInteger &&
                !llvm::APInt(64, static_cast<uint64_t>(item), true)
                     .isSignedIntN(sourceInteger.getWidth())) {
                return {};
            }
        }
        return constant(value.getType(), *inputIntegers);
    }
    if (op->getNumOperands() == 1) {
        Integers values;
        for (int64_t item : *inputIntegers) {
            auto result = unaryInteger(name, item, op->getOperand(0).getType());
            if (!result) {
                return {};
            }
            values.push_back(*result);
        }
        return constant(value.getType(), values);
    }
    if (op->getNumOperands() == 2) {
        auto right = integers(resolve(op->getOperand(1), pending));
        if (!right || right->size() != inputIntegers->size()) {
            return {};
        }
        Integers values;
        for (auto [leftItem, rightItem] : llvm::zip(*inputIntegers, *right)) {
            auto result = binaryInteger(op, leftItem, rightItem);
            if (!result) {
                return {};
            }
            values.push_back(*result);
        }
        return constant(value.getType(), values);
    }
    return {};
}

} // namespace catalyst::pbc
