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

#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"

using namespace mlir;

namespace catalyst::pbc {
namespace {

constexpr int64_t maxElements = 100000;
constexpr unsigned maxRecursion = 256;
constexpr unsigned maxCachedValues = 100000;
using Integers = llvm::SmallVector<int64_t>;

Type elementType(Type type) {
    if (auto tensor = dyn_cast<RankedTensorType>(type)) {
        return tensor.getElementType();
    }
    return type;
}

std::optional<int64_t> elementCount(Type type) {
    if (isa<IntegerType, IndexType>(type)) {
        return 1;
    }
    auto tensor = dyn_cast<RankedTensorType>(type);
    if (!tensor || !tensor.hasStaticShape() || tensor.getRank() > 1 ||
        !isa<IntegerType, IndexType>(tensor.getElementType()) ||
        tensor.getNumElements() > maxElements) {
        return std::nullopt;
    }
    return tensor.getNumElements();
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

} // namespace

void WireIndexResolver::bind(Value target, Value source, const WireIndexResolver &from) {
    Attribute value = from.resolveConstant(source);
    // Resolving first also handles binding from this resolver itself.
    bindConstant(target, value);
}

void WireIndexResolver::bindConstant(Value target, Attribute value) {
    cache.clear();
    bindings[target] = value;
}

Attribute WireIndexResolver::resolveConstant(Value value) const { return resolve(value, 0); }

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

Attribute WireIndexResolver::resolve(Value value, unsigned depth) const {
    if (!value || depth > maxRecursion || !elementCount(value.getType())) {
        return {};
    }
    if (auto it = bindings.find(value); it != bindings.end()) {
        return it->second;
    }
    if (auto it = cache.find(value); it != cache.end()) {
        return it->second;
    }
    if (cache.size() >= maxCachedValues) {
        return {};
    }
    Attribute result = evaluate(value, depth + 1);
    cache[value] = result;
    return result;
}

Attribute WireIndexResolver::evaluate(Value value, unsigned depth) const {
    Operation *op = value.getDefiningOp();
    if (!op) {
        return {};
    }
    llvm::StringRef name = op->getName().getStringRef();
    if (name == "arith.constant" || name == "stablehlo.constant") {
        auto values = integers(op->getAttr("value"));
        return values ? constant(value.getType(), *values) : Attribute{};
    }
    if (name == "tensor.from_elements") {
        Integers values;
        for (Value operand : op->getOperands()) {
            auto item = integers(resolve(operand, depth));
            if (!item || item->size() != 1) {
                return {};
            }
            values.push_back(item->front());
        }
        return constant(value.getType(), values);
    }
    if (op->getNumOperands() == 0) {
        return {};
    }
    auto input = integers(resolve(op->getOperand(0), depth));
    if (!input) {
        return {};
    }
    if (name == "tensor.extract") {
        auto tensor = dyn_cast<RankedTensorType>(op->getOperand(0).getType());
        if (!tensor || op->getNumOperands() != tensor.getRank() + 1) {
            return {};
        }
        int64_t index = 0;
        if (tensor.getRank() == 1) {
            auto indices = integers(resolve(op->getOperand(1), depth));
            if (!indices || indices->size() != 1) {
                return {};
            }
            index = indices->front();
        }
        if (index < 0 || index >= static_cast<int64_t>(input->size())) {
            return {};
        }
        return constant(value.getType(), {(*input)[index]});
    }
    if (name == "stablehlo.slice") {
        auto start = op->getAttrOfType<DenseI64ArrayAttr>("start_indices");
        auto limit = op->getAttrOfType<DenseI64ArrayAttr>("limit_indices");
        auto strides = op->getAttrOfType<DenseI64ArrayAttr>("strides");
        if (!start || !limit || !strides || start.size() != 1 || limit.size() != 1 ||
            strides.size() != 1 || start[0] < 0 || limit[0] < start[0] ||
            limit[0] > static_cast<int64_t>(input->size()) || strides[0] <= 0) {
            return {};
        }
        Integers values;
        for (int64_t index = start[0]; index < limit[0];) {
            values.push_back((*input)[index]);
            // Avoid overflow for a legal stride larger than the remaining slice.
            if (strides[0] >= limit[0] - index) {
                break;
            }
            index += strides[0];
        }
        return constant(value.getType(), values);
    }
    if (name == "stablehlo.reshape" || name == "tensor.cast" || name == "stablehlo.convert" ||
        name == "arith.index_cast" || name == "arith.index_castui" || name == "arith.extsi" ||
        name == "arith.extui" || name == "arith.trunci") {
        Type sourceType = elementType(op->getOperand(0).getType());
        auto sourceInteger = dyn_cast<IntegerType>(sourceType);
        for (int64_t item : *input) {
            if ((name == "arith.index_castui" || name == "arith.extui") && item < 0) {
                return {};
            }
            if ((name == "arith.index_cast" || name == "arith.extsi") && sourceInteger &&
                !llvm::APInt(64, static_cast<uint64_t>(item), true)
                     .isSignedIntN(sourceInteger.getWidth())) {
                return {};
            }
        }
        return constant(value.getType(), *input);
    }
    return {};
}

} // namespace catalyst::pbc
