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

#include "Quantum/IR/QuantumInterfaces.h"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <utility>

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Types.h"
#include "mlir/Support/LLVM.h"

#include "Quantum/IR/QuantumOps.h"

using namespace mlir;
using namespace catalyst::quantum;

#include "Quantum/IR/QuantumInterfaces.cpp.inc"

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

namespace {

int64_t groupStartIndex(NamedAttribute entry) {
    auto indices = cast<DenseI64ArrayAttr>(entry.getValue()).asArrayRef();
    return indices.empty() ? std::numeric_limits<int64_t>::max() : *llvm::min_element(indices);
}

bool groupPrecedes(NamedAttribute lhs, NamedAttribute rhs) {
    return std::pair(groupStartIndex(lhs), lhs.getName().getValue()) <
           std::pair(groupStartIndex(rhs), rhs.getName().getValue());
}

SmallVector<NamedAttribute> getOrderedGroups(DictionaryAttr groups) {
    SmallVector<NamedAttribute> ordered(groups.begin(), groups.end());
    llvm::sort(ordered, groupPrecedes);
    return ordered;
}

SmallVector<SmallVector<Type>> getOrderedDynamicTypes(DecomposableGate gate, Operation *op) {
    if (auto custom = dyn_cast<CustomOp>(op)) {
        SmallVector<SmallVector<Type>> result;
        result.reserve(custom.getParams().size());
        for (Value param : custom.getParams()) {
            result.push_back({param.getType()});
        }
        return result;
    }

    if (auto generic = dyn_cast<OperatorOp>(op)) {
        SmallVector<NamedAttribute> groups = getOrderedGroups(generic.getParamMap());
        SmallVector<SmallVector<Type>> result;
        result.reserve(groups.size());
        for (NamedAttribute group : groups) {
            SmallVector<Type> types;
            for (int64_t index : cast<DenseI64ArrayAttr>(group.getValue()).asArrayRef()) {
                types.push_back(generic.getParams()[index].getType());
            }
            result.push_back(std::move(types));
        }
        return result;
    }

    // Built-in decomposable operations currently have at most one parameter group. Sorting the
    // fallback by name keeps future implementations deterministic until they expose operand order.
    auto dynamicShape = gate.getDynamicShape();
    SmallVector<StringRef> names;
    names.reserve(dynamicShape.size());
    for (const auto &[name, _] : dynamicShape) {
        names.push_back(name);
    }
    llvm::sort(names);

    SmallVector<SmallVector<Type>> result;
    result.reserve(names.size());
    for (StringRef name : names) {
        result.push_back(dynamicShape.lookup(name));
    }
    return result;
}

SmallVector<size_t> getOrderedWireLengths(DecomposableGate gate, Operation *op) {
    if (auto custom = dyn_cast<CustomOp>(op)) {
        return {custom.getNonCtrlQubitOperands().size()};
    }

    if (auto generic = dyn_cast<OperatorOp>(op)) {
        SmallVector<NamedAttribute> groups = getOrderedGroups(generic.getQubitMap());
        auto wireLengths = gate.getWireLens();
        SmallVector<size_t> result;
        result.reserve(groups.size());
        for (NamedAttribute group : groups) {
            result.push_back(wireLengths.lookup(group.getName()));
        }
        return result;
    }

    auto wireLengths = gate.getWireLens();
    SmallVector<StringRef> names;
    names.reserve(wireLengths.size());
    for (const auto &[name, _] : wireLengths) {
        names.push_back(name);
    }
    llvm::sort(names);

    SmallVector<size_t> result;
    result.reserve(names.size());
    for (StringRef name : names) {
        result.push_back(wireLengths.lookup(name));
    }
    return result;
}

mlir::DictionaryAttr buildGraphOpKeyAttr(DecomposableGate gate, Operation *op) {
    MLIRContext *context = op->getContext();
    Builder builder(context);

    SmallVector<Attribute> dynamicTypes;
    for (const auto &types : getOrderedDynamicTypes(gate, op)) {
        SmallVector<Attribute> typeNames;
        typeNames.reserve(types.size());
        for (Type type : types) {
            typeNames.push_back(TypeAttr::get(type));
        }
        dynamicTypes.push_back(builder.getArrayAttr(typeNames));
    }

    SmallVector<Attribute> wireLengths;
    for (size_t length : getOrderedWireLengths(gate, op)) {
        wireLengths.push_back(builder.getI64IntegerAttr(length));
    }

    size_t numControls = 0;
    if (auto quantumGate = mlir::dyn_cast<QuantumGate>(op)) {
        numControls = quantumGate.getCtrlQubitOperands().size();
    }

    NamedAttrList fields;
    fields.append("op", builder.getStringAttr(gate.getOperatorName()));
    NamedAttrList traits;
    if (op->hasAttr("adjoint")) {
        traits.append("adj", builder.getBoolAttr(true));
    }
    if (!dynamicTypes.empty()) {
        fields.append("params", builder.getArrayAttr(dynamicTypes));
    }
    if (numControls) {
        traits.append("controls", builder.getI64IntegerAttr(numControls));
    }
    DictionaryAttr staticData = gate.getStaticData();
    if (!staticData.empty()) {
        fields.append("static", staticData);
    }
    if (!traits.empty()) {
        fields.append("traits", traits.getDictionary(context));
    }

    if (std::string uid = gate.getExtraData(); !uid.empty()) {
        uint64_t value = 0;
        bool invalid = llvm::StringRef(uid).getAsInteger(10, value);
        assert(!invalid && "DecomposableGate extra data must be an integer UID");
        fields.append("uid", builder.getI64IntegerAttr(value));
    }
    if (!wireLengths.empty()) {
        fields.append("wires", builder.getArrayAttr(wireLengths));
    }

    return fields.getDictionary(context);
}

} // namespace

//===----------------------------------------------------------------------===//
// Quantum interface definitions.
//===----------------------------------------------------------------------===//

namespace catalyst {
namespace quantum {

std::string defaultGetGraphOpId(Operation *op) {
    DecomposableGate gate = cast<DecomposableGate>(op);
    std::string out;
    llvm::raw_string_ostream stream(out);
    buildGraphOpKeyAttr(gate, op).print(stream);
    return out;
}

} // namespace quantum
} // namespace catalyst
