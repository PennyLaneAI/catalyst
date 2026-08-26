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

#include "QRef/IR/QRefInterfaces.h"

#include <cstddef>
#include <cstdint>
#include <string>

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/Types.h"
#include "mlir/Support/LLVM.h"

using namespace mlir;
using namespace catalyst::qref;

#include "QRef/IR/QRefInterfaces.cpp.inc"

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

namespace {

void printAttr(mlir::Attribute attr, llvm::raw_string_ostream &ss) {
    llvm::TypeSwitch<mlir::Attribute, void>(attr)
        .Case<mlir::DictionaryAttr>([&](mlir::DictionaryAttr dict) {
            ss << "{";
            for (auto [i, entry] : llvm::enumerate(dict)) {
                if (i > 0) {
                    ss << ",";
                }

                ss << entry.getName().str() << ":";
                printAttr(entry.getValue(), ss);
            }
            ss << "}";
        })
        .Case<mlir::ArrayAttr>([&](mlir::ArrayAttr arr) {
            ss << "[";
            for (auto [i, attr] : llvm::enumerate(arr)) {
                if (i > 0) {
                    ss << ",";
                }
                printAttr(attr, ss);
            }
            ss << "]";
        })
        .Case<mlir::StringAttr>([&](mlir::StringAttr attr) { ss << attr.str(); })
        .Case<mlir::IntegerAttr>([&](mlir::IntegerAttr attr) { ss << attr.getInt(); })
        .Case<mlir::FloatAttr>([&](mlir::FloatAttr attr) { ss << attr.getValueAsDouble(); })
        .Default([&](mlir::Attribute attr) { attr.print(ss); });
}

void printShapedType(ArrayRef<int64_t> shape, int64_t dim, Type elementType,
                        llvm::raw_string_ostream &ss) {
    // Rank-0 tensors (e.g. tensor<f64>) have an empty shape; print the
    // element type directly instead of indexing into the empty ArrayRef.
    if (shape.empty()) {
        ss << elementType;
        return;
    }

    int64_t length = shape[dim];
    auto printList = [&](auto printItem) {
        ss << "[";
        for (int64_t i = 0; i < length; i++) {
            printItem();
            if (i != length - 1) {
                ss << ",";
            }
        }
        ss << "]";
    };

    if (static_cast<int64_t>(shape.size()) == dim + 1) {
        printList([&]() { ss << elementType; });
    } else {
        printList([&]() { printShapedType(shape, dim + 1, elementType, ss); });
    }
}

void printType(mlir::Type type, llvm::raw_string_ostream &ss) {
    llvm::TypeSwitch<mlir::Type, void>(type)
        .Case<mlir::ShapedType>([&](mlir::ShapedType shapedType) {
            printShapedType(shapedType.getShape(), 0, shapedType.getElementType(), ss);
        })
        .Default([&](mlir::Type other) { other.print(ss); });
}

template <typename T, typename PrintFunc>
void printSortedMap(const llvm::StringMap<T> &map, llvm::raw_string_ostream &ss,
                    PrintFunc printValue) {
    llvm::SmallVector<llvm::StringRef> keys;
    for (const llvm::StringRef key : map.keys()) {
        keys.push_back(key);
    }
    llvm::sort(keys);

    ss << "{";
    for (auto [i, key] : llvm::enumerate(keys)) {
        if (i > 0) {
            ss << ",";
        }
        ss << key << ":";
        printValue(map.lookup(key), ss);
    }
    ss << "}";
}

void printDynamicShape(const llvm::StringMap<llvm::SmallVector<mlir::Type>> &map,
                        llvm::raw_string_ostream &ss) {
    printSortedMap(map, ss, [](const auto &types, llvm::raw_string_ostream &stream) {
        stream << "[";
        for (auto [j, type] : llvm::enumerate(types)) {
            if (j > 0) {
                stream << ",";
            }
            printType(type, stream);
        }
        stream << "]";
    });
}

void printWireLens(const llvm::StringMap<size_t> &map, llvm::raw_string_ostream &ss) {
    printSortedMap(map, ss, [](size_t len, llvm::raw_string_ostream &stream) { stream << len; });
}
    
} // namespace

//===----------------------------------------------------------------------===//
// QRef interface definitions.
//===----------------------------------------------------------------------===//

namespace catalyst {
namespace qref {

std::string defaultGetGraphOpId(Operation *op) {
    std::string out;
    llvm::raw_string_ostream ss(out);

    DecomposableGate gate = cast<DecomposableGate>(op);

    // Fold the adjoint modifier into the operator name so that `Op` and `Adjoint(Op)` are
    // distinct in the graphOpId. The modifier wraps only the name; the param/wire/static/uid
    // groups follow it, i.e. `Adjoint(Rot){...}{wires...}`, not `Adjoint(Rot{...}{wires...})`.
    std::string name = gate.getOperatorName();
    if (op->hasAttr("adjoint")) {
        name = "Adjoint(" + name + ")";
    }

    ss << name;
    printDynamicShape(gate.getDynamicShape(), ss);
    printWireLens(gate.getWireLens(), ss);
    printAttr(gate.getStaticData(), ss);
    if (gate.getExtraData() != "") {
        ss << '[' << gate.getExtraData() << ']';
    }
    ss.flush();

    return out;
}

} // namespace qref
} // namespace catalyst
