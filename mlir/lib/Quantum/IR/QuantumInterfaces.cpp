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

#include <cstdint>
#include <string>

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Support/LLVM.h"

using namespace mlir;
using namespace catalyst::quantum;

#include "Quantum/IR/QuantumInterfaces.cpp.inc"

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

namespace {
template <typename T> void printIterable(T iterable, llvm::raw_string_ostream &ss)
{
    ss << "[";
    llvm::interleave(iterable, ss, ",");
    ss << "]";
}

void printAttr(mlir::Attribute attr, llvm::raw_string_ostream &ss)
{
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
                     llvm::raw_string_ostream &ss)
{
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
    }
    else {
        printList([&]() { printShapedType(shape, dim + 1, elementType, ss); });
    }
}

void printType(mlir::Type type, llvm::raw_string_ostream &ss)
{
    llvm::TypeSwitch<mlir::Type, void>(type)
        .Case<mlir::ShapedType>([&](mlir::ShapedType shapedType) {
            printShapedType(shapedType.getShape(), 0, shapedType.getElementType(), ss);
        })
        .Default([&](mlir::Type other) { other.print(ss); });
}

void printTypeRange(mlir::TypeRange typerange, llvm::raw_string_ostream &ss)
{
    ss << "[";
    for (auto [i, type] : llvm::enumerate(typerange)) {
        if (i > 0) {
            ss << ",";
        }
        printType(type, ss);
    }
    ss << "]";
}

} // namespace

//===----------------------------------------------------------------------===//
// Quantum interface definitions.
//===----------------------------------------------------------------------===//

namespace catalyst {
namespace quantum {

std::string defaultGetGraphOpId(Operation *op)
{
    std::string out;
    llvm::raw_string_ostream ss(out);

    DecomposableGate gate = cast<DecomposableGate>(op);

    ss << gate.getOperatorName();
    printTypeRange(gate.getDynamicShape(), ss);
    printIterable(gate.getWireLens(), ss);
    printAttr(gate.getStaticData(), ss);
    if (gate.getExtraData() != "") {
        ss << '[' << gate.getExtraData() << ']';
    }
    return out;
}

} // namespace quantum
} // namespace catalyst
