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
#include <string>

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Types.h"

using namespace mlir;
using namespace catalyst::qref;

//===----------------------------------------------------------------------===//
// QRef interface definitions.
//===----------------------------------------------------------------------===//

#include "QRef/IR/QRefInterfaces.cpp.inc"

namespace {

void printAttr(Attribute attr, llvm::raw_string_ostream &stream) {
    llvm::TypeSwitch<Attribute, void>(attr)
        .Case<DictionaryAttr>([&](DictionaryAttr dict) {
            stream << "{";
            for (auto [index, entry] : llvm::enumerate(dict)) {
                if (index > 0) {
                    stream << ",";
                }
                stream << entry.getName().str() << ":";
                printAttr(entry.getValue(), stream);
            }
            stream << "}";
        })
        .Case<ArrayAttr>([&](ArrayAttr array) {
            stream << "[";
            for (auto [index, value] : llvm::enumerate(array)) {
                if (index > 0) {
                    stream << ",";
                }
                printAttr(value, stream);
            }
            stream << "]";
        })
        .Case<StringAttr>([&](StringAttr value) { stream << value.str(); })
        .Case<IntegerAttr>([&](IntegerAttr value) { stream << value.getInt(); })
        .Case<FloatAttr>([&](FloatAttr value) { stream << value.getValueAsDouble(); })
        .Default([&](Attribute value) { value.print(stream); });
}

template <typename T, typename PrintValue>
void printSortedMap(const llvm::StringMap<T> &map, llvm::raw_string_ostream &stream,
                    PrintValue printValue) {
    llvm::SmallVector<llvm::StringRef> keys(map.keys());
    llvm::sort(keys);

    stream << "{";
    for (auto [index, key] : llvm::enumerate(keys)) {
        if (index > 0) {
            stream << ",";
        }
        stream << key << ":";
        printValue(map.lookup(key), stream);
    }
    stream << "}";
}

} // namespace

std::string catalyst::qref::defaultGetGraphOpId(Operation *op) {
    DecomposableGate gate = cast<DecomposableGate>(op);
    std::string name = gate.getOperatorName();
    if (gate.getAdjointFlag()) {
        name = "Adjoint(" + name + ")";
    }

    size_t numControls = gate.getCtrlQubitOperands().size();
    if (numControls == 1) {
        name = "C(" + name + ")";
    } else if (numControls > 1) {
        name = std::to_string(numControls) + "C(" + name + ")";
    }

    std::string result;
    llvm::raw_string_ostream stream(result);
    stream << name;
    printSortedMap(gate.getDynamicShape(), stream,
                   [](const llvm::SmallVector<Type> &types, llvm::raw_string_ostream &out) {
                       out << "[";
                       for (auto [index, type] : llvm::enumerate(types)) {
                           if (index > 0) {
                               out << ",";
                           }
                           out << type;
                       }
                       out << "]";
                   });
    printSortedMap(gate.getWireLens(), stream,
                   [](size_t count, llvm::raw_string_ostream &out) { out << count; });
    printAttr(gate.getStaticData(), stream);
    if (!gate.getExtraData().empty()) {
        stream << '[' << gate.getExtraData() << ']';
    }
    stream.flush();
    return result;
}
