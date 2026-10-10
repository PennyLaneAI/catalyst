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

#pragma once

#include <memory>
#include <string>
#include <tuple>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/IR/Operation.h"
#include "mlir/Support/LogicalResult.h"

#include "DGTypes.hpp"

namespace DecompGraph::Solver {
class DecompositionGraph;
} // namespace DecompGraph::Solver

namespace catalyst {
namespace quantum {

namespace GraphDecompositionPrep {

/**
 * @brief Options shared by passes that build a decomposition graph.
 *
 * Necessary options for the preparation of a decomposition graph.
 */
struct PrepOptions {
    llvm::ArrayRef<std::string> targetGateSetOption;
    llvm::ArrayRef<std::string> fixedDecompsOption;
    llvm::ArrayRef<std::string> altDecompsOption;
    llvm::StringRef bytecodeRulesFile;
    llvm::StringRef libQPDPath;
    llvm::StringRef libpythonPath;
};

/// Callback to lower `quantum.ctrl` / `quantum.adjoint` regions to op-level modifiers.
using LowerModifiersFn = mlir::function_ref<mlir::LogicalResult(mlir::Operation *)>;

/// Extract `{...}` starting at `rest.front()`, handling nesting. Advances `rest` past it.
inline bool consumeBraceGroup(llvm::StringRef &rest, llvm::StringRef &content) {
    if (!rest.consume_front("{")) {
        return false;
    }
    int depth = 1;
    size_t i = 0;
    for (; i < rest.size(); ++i) {
        if (rest[i] == '{') {
            ++depth;
        } else if (rest[i] == '}') {
            --depth;
            if (depth == 0) {
                content = rest.take_front(i);
                rest = rest.drop_front(i + 1);
                return true;
            }
        }
    }
    return false;
}

/// Count top-level comma-separated entries in a `{params}` body (respects `[...]`).
inline int countParamEntries(llvm::StringRef content) {
    if (content.empty()) {
        return 0;
    }
    int bracketDepth = 0;
    int count = 1;
    for (char c : content) {
        if (c == '[') {
            ++bracketDepth;
        } else if (c == ']') {
            --bracketDepth;
        } else if (c == ',' && bracketDepth == 0) {
            ++count;
        }
    }
    return count;
}

/// Parse `{wires:N}` (or `{}` → 0). Returns -1 if the group is absent/unparseable.
inline int parseWireLen(llvm::StringRef content) {
    if (content.empty()) {
        return 0;
    }
    constexpr llvm::StringLiteral key = "wires:";
    auto pos = content.find(key);
    if (pos == llvm::StringRef::npos) {
        return -1;
    }
    llvm::StringRef num = content.drop_front(pos + key.size()).take_until([](char c) {
        return c == ',';
    });
    int w = -1;
    if (num.getAsInteger(10, w)) {
        return -1;
    }
    return w;
}

/**
 * @brief Parse a graphOpId string into an OperatorNode.
 *
 * The graphOpId format is "<name>{params}{wires}{static}[uid]", where <name> already
 * carries any name-wrapped op-level modifiers produced by `defaultGetGraphOpId`, e.g.
 * "C(Adjoint(RX)){0:[f64]}{wires:1}{}". Also accepts the legacy "Name(w,p)" form.
 *
 * For the graphOpId form, `numParams` is the number of entries in `{params}` and
 * `numWires` is taken from `{wires:N}` (or 0 when that group is empty).
 */
inline DecompGraph::Core::OperatorNode parseOperator(llvm::StringRef raw) {
    DecompGraph::Core::OperatorNode node;

    // Base op: either the graphOpId "Name{...}..." form or the legacy "Name(w,p)" form.
    if (raw.contains('{')) {
        node.id = raw.str();
        node.name = raw.take_until([](char c) { return c == '{'; }).str();
        raw = raw.drop_front(node.name.size());

        llvm::StringRef paramsBody;
        llvm::StringRef wiresBody;
        if (consumeBraceGroup(raw, paramsBody)) {
            node.numParams = countParamEntries(paramsBody);
        }
        if (consumeBraceGroup(raw, wiresBody)) {
            int w = parseWireLen(wiresBody);
            if (w >= 0) {
                node.numWires = w;
            }
        }
        // Remaining `{static}[uid]` is ignored for numWires/numParams.
        return node;
    }

    auto openIdx = raw.find('(');
    if (openIdx == llvm::StringRef::npos) {
        node.name = raw.trim().str();
        return node;
    }
    node.name = raw.take_front(openIdx).trim().str();
    raw = raw.drop_front(openIdx); // leftover: "(w,p)" or "(w)"

    // Parse "(w,p)" (new) or "(w)" (legacy) suffix.
    if (raw.consume_front("(") && raw.consume_back(")")) {
        llvm::StringRef wStr, pStr;
        std::tie(wStr, pStr) = raw.split(',');
        int w = -1, p = -1;
        if (!wStr.getAsInteger(10, w)) {
            node.numWires = w;
        }
        if (!pStr.empty() && !pStr.getAsInteger(10, p)) {
            node.numParams = p;
        }
        // If pStr is empty we were given the legacy "(w)" format; leave
        // numParams at the wildcard default so old bytecode keeps working.
    }

    return node;
}

/**
 * @brief Prepare a decomposition graph from a module.
 *
 * Invokes `lowerModifiers` to strip `quantum.ctrl` / `quantum.adjoint` regions
 * if needed, then gathers rules for the target gateset and builds a
 * DecompositionGraph.
 *
 * Returns nullptr on failure.
 */
std::unique_ptr<DecompGraph::Solver::DecompositionGraph>
prepareGraph(mlir::Operation *op, const PrepOptions &opts, LowerModifiersFn lowerModifiers);

} // namespace GraphDecompositionPrep
} // namespace quantum
} // namespace catalyst
