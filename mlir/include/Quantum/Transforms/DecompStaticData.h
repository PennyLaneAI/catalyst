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

/**
 * @file DecompStaticData.h
 *
 * @brief helpers for operators specialized by static data.
 */
#pragma once

#include <algorithm>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Operation.h"

#include "Quantum/IR/QuantumOps.h"

namespace catalyst {
namespace quantum {

/// Stringify a single static-data attribute value. String values are used verbatim;
/// anything else falls back to its generic printed form so the representation stays
/// stable and comparable.
inline std::string stringifyStaticDataValue(mlir::Attribute value)
{
    if (auto strAttr = mlir::dyn_cast<mlir::StringAttr>(value)) {
        return strAttr.getValue().str();
    }
    std::string out;
    llvm::raw_string_ostream os(out);
    value.print(os);
    return os.str();
}

/// Convert a `static_data` dictionary in Operators into the `staticNamedArgs` map
/// used by the graph solver.
inline std::unordered_map<std::string, std::string> staticDataToMap(mlir::DictionaryAttr staticData)
{
    std::unordered_map<std::string, std::string> result;
    if (!staticData) {
        return result;
    }
    for (mlir::NamedAttribute entry : staticData) {
        result[entry.getName().str()] = stringifyStaticDataValue(entry.getValue());
    }
    return result;
}

/// Return the static data specializing a concrete op as a DictionaryAttr.
inline mlir::DictionaryAttr staticDataOf(mlir::Operation *op)
{
    if (auto operatorOp = mlir::dyn_cast<OperatorOp>(op)) {
        return operatorOp.getStaticDataAttr();
    }
    if (auto pauliRot = mlir::dyn_cast<PauliRotOp>(op)) {
        mlir::MLIRContext *ctx = op->getContext();
        return mlir::DictionaryAttr::get(
            ctx, {mlir::NamedAttribute(mlir::StringAttr::get(ctx, "pauli_word"),
                                       mlir::StringAttr::get(ctx, pauliRot.getPauliWord()))});
    }
    return mlir::DictionaryAttr();
}

/// Build a decomposition-registry key from a base operator name and its static data.
inline std::string makeDecompRegistryKey(llvm::StringRef baseName, mlir::DictionaryAttr staticData)
{
    if (!staticData || staticData.empty()) {
        return baseName.str();
    }

    std::vector<std::pair<std::string, std::string>> entries;
    entries.reserve(staticData.size());
    for (mlir::NamedAttribute entry : staticData) {
        entries.emplace_back(entry.getName().str(), stringifyStaticDataValue(entry.getValue()));
    }
    std::sort(entries.begin(), entries.end());

    std::string key = baseName.str() + "#";
    for (std::size_t i = 0; i < entries.size(); ++i) {
        if (i != 0) {
            key += ";";
        }
        key += entries[i].first + "=" + entries[i].second;
    }
    return key;
}

} // namespace quantum
} // namespace catalyst
