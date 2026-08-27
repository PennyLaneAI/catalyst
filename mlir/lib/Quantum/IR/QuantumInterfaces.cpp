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
using namespace catalyst::quantum;

#include "Quantum/IR/QuantumInterfaces.cpp.inc"

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
            stream << type;
        }
        stream << "]";
    });
}

void printWireLens(const llvm::StringMap<size_t> &map, llvm::raw_string_ostream &ss) {
    printSortedMap(map, ss, [](size_t len, llvm::raw_string_ostream &stream) { stream << len; });
}

} // namespace

//===----------------------------------------------------------------------===//
// Quantum interface definitions.
//===----------------------------------------------------------------------===//

namespace catalyst {
namespace quantum {

// Wrap an operator's base name with its op-level modifiers (name-wrap), so that `Op`,
// `Adjoint(Op)`, `C(Op)`, and nested combinations like `C(Adjoint(Op))` are distinct graph
// identities. Modifiers are applied innermost-first in a canonical order (adjoint innermost,
// control outermost) so a nested op has a single spelling.
// An op that is both controlled and adjointed could otherwise be formed as
// `C(Adjoint(Op))` or `Adjoint(C(Op))` which will produce two ids for one operator,
// which would split the decomposition graph into two nodes and prevent rules from matching.
// Always emitting the control-outermost form keeps such ops on a single node. The
// caller appends the param/wire/static/uid groups. Extend this helper to support future op-level
// modifiers, preserving the canonical order (mirror the Python side in decomposition_rules.py).
static std::string wrapModifiers(std::string name, Operation *op) {
    if (op->hasAttr("adjoint")) {
        name = "Adjoint(" + name + ")";
    }
    if (auto gate = mlir::dyn_cast<QuantumGate>(op)) {
        size_t numCtrl = gate.getCtrlQubitOperands().size();
        if (numCtrl == 1) {
            name = "C(" + name + ")";
        } else if (numCtrl > 1) {
            name = std::to_string(numCtrl) + "C(" + name + ")";
        }
    }
    return name;
}

std::string defaultGetGraphOpId(Operation *op) {
    std::string out;
    llvm::raw_string_ostream ss(out);

    DecomposableGate gate = cast<DecomposableGate>(op);

    // Fold op-level modifiers into the operator name so that `Op`, `Adjoint(Op)`, `C(Op)`, and
    // nested combinations are distinct in the graphOpId. Modifiers wrap only the name; the
    // param/wire/static/uid groups follow, e.g. `C(Adjoint(Rot)){...}{wires...}`.
    ss << wrapModifiers(gate.getOperatorName(), op);
    printDynamicShape(gate.getDynamicShape(), ss);
    printWireLens(gate.getWireLens(), ss);
    printAttr(gate.getStaticData(), ss);
    if (gate.getExtraData() != "") {
        ss << '[' << gate.getExtraData() << ']';
    }
    ss.flush();

    return out;
}

} // namespace quantum
} // namespace catalyst
