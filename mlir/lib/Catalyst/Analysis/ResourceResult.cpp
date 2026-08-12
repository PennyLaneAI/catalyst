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

#include "Catalyst/Analysis/ResourceResult.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <utility>

#include "llvm/ADT/Hashing.h"
#include "llvm/Support/ErrorHandling.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"

#include "Catalyst/Analysis/ResourceResultExtension.h"

#include <llvm/ADT/DenseMap.h>
#include <llvm/Support/JSON.h>

using namespace mlir;

using namespace llvm;

namespace catalyst {

using MergeMethod = ResourceResult::MergeMethod;

/// Helper: select merge function based on method enum.
template <typename T> static T applyMerge(T a, T b, MergeMethod method) {
    switch (method) {
    case MergeMethod::Max:
        return std::max(a, b);
    case MergeMethod::Min:
        return std::min(a, b);
    case MergeMethod::Sum:
        return a + b;
    }
    llvm_unreachable("unknown ResourceResult::MergeMethod");
}

// Merge a flat StringMap with a single operator[] per key.
template <typename Map> static void mergeStringMap(Map &dst, const Map &src, MergeMethod method) {
    for (const auto &entry : src) {
        auto &slot = dst[entry.getKey()];
        slot = applyMerge(slot, entry.getValue(), method);
    }
}

void ResourceResult::mergeWith(const ResourceResult &other, MergeMethod method) {
    for (const auto &opEntry : other.operations) {
        auto &innerDst = operations[opEntry.getKey()];
        for (const auto &sizeEntry : opEntry.getValue()) {
            auto &slot = innerDst[sizeEntry.first];
            slot = applyMerge(slot, sizeEntry.second, method);
        }
    }

    mergeStringMap(measurements, other.measurements, method);
    mergeStringMap(detailedOperations, other.detailedOperations, method);
    mergeStringMap(classicalInstructions, other.classicalInstructions, method);
    mergeStringMap(functionCalls, other.functionCalls, method);

    // varFunctionCalls hold identifiers for unknown dynamic counts. If the
    // same key appears twice, the merge result represents a new unknown value
    // such as Sum/Max/Min(lhs, rhs), not either input identifier.
    for (const auto &entry : other.varFunctionCalls) {
        auto [it, inserted] = varFunctionCalls.try_emplace(entry.getKey(), entry.getValue());
        if (!inserted) {
            it->second = static_cast<size_t>(hash_combine(
                entry.getKey(), it->second, entry.getValue(), static_cast<int>(method)));
        }
    }

    numAllocQubits = applyMerge(numAllocQubits, other.numAllocQubits, method);

    hasBranches = hasBranches || other.hasBranches;
    hasDynLoop = hasDynLoop || other.hasDynLoop;
    collectDetailedOperations = collectDetailedOperations || other.collectDetailedOperations;

    for (auto [ext, otherExt] : llvm::zip(extensions, other.extensions)) {
        assert(ext->name() == otherExt->name() && "extension names must match");
        ext->mergeWith(*otherExt, method);
    }
}

void ResourceResult::multiplyBy(double scalar) {
    for (auto &opEntry : operations) {
        for (auto &sizeEntry : opEntry.getValue()) {
            sizeEntry.second *= scalar;
        }
    }

    for (auto &entry : measurements) {
        entry.getValue() *= scalar;
    }

    for (auto &entry : detailedOperations) {
        entry.getValue() *= scalar;
    }

    for (auto &entry : classicalInstructions) {
        entry.getValue() *= scalar;
    }

    for (auto &entry : functionCalls) {
        entry.getValue() *= scalar;
    }

    numAllocQubits *= scalar;

    for (auto &m : extensions) {
        m->multiplyBy(scalar);
    }
}

// Emit a count as a JSON number. Counts are tracked as doubles to support probabilistic
// (fractional) count values, but the JSON output always reports the nearest integer.
//
// JSON Schema (per function, keyed by name at the root):
//   metadata: { qnode, auto_qubit_management?, has_branches, device_name? }
//   num_qubits: { alloc, arg, total }
//   classical_instructions: { "dialect.op": count, ... }
//   quantum_operations: { "<wires>": { "op_name": count, ... }, ... }  // optional
//   function_calls: { static: { "fxn": count, ... }, dynamic: { "fxn": id, ... } }
//   measurement_processes: { "meas_type": count, ... }
//   extended_fields: { "<extension>": { ... }, ... }  // e.g. pbc_depth: { any_commuting_depth,
//   qubit_disjoint_depth }
static llvm::json::Value countToJson(double count) {
    return llvm::json::Value(static_cast<int64_t>(std::llround(count)));
}

llvm::json::Object ResourceResult::toJson() const {
    llvm::json::Object funcObj;

    /// Metadata
    llvm::json::Object metaDataObject;
    metaDataObject["device_name"] = deviceName;
    metaDataObject["qnode"] = isQnode;
    metaDataObject["has_branches"] = hasBranches;
    metaDataObject["auto_qubit_management"] = autoQubitManagement;
    funcObj["metadata"] = std::move(metaDataObject);

    /// Qubit Allocations
    llvm::json::Object numQubitObject;
    numQubitObject["alloc"] = countToJson(numAllocQubits);
    numQubitObject["arg"] = numArgQubits;
    numQubitObject["total"] = countToJson(numQubits());
    funcObj["num_qubits"] = std::move(numQubitObject);

    // Classical Operations
    llvm::json::Object classicalInstructionsObject;
    for (const auto &entry : classicalInstructions) {
        classicalInstructionsObject[entry.getKey()] = countToJson(entry.getValue());
    }
    funcObj["classical_instructions"] = std::move(classicalInstructionsObject);

    /// Quantum Operations
    llvm::DenseMap<int, llvm::DenseMap<StringRef, double>> quantumOperationCounts;
    for (const auto &opEntry : operations) {
        StringRef opName = opEntry.getKey();
        for (const auto &sizeEntry : opEntry.getValue()) {
            const auto &[nQubits, nParams] = sizeEntry.first;
            double count = sizeEntry.second;
            quantumOperationCounts[nQubits][opName] += count;
        }
    }
    llvm::json::Object quantumOperationObject;
    for (const auto &[nQubits, opCounts] : quantumOperationCounts) {
        auto [it, _] =
            quantumOperationObject.try_emplace(std::to_string(nQubits), llvm::json::Object{});
        for (const auto &[opName, count] : opCounts) {
            (*it->getSecond().getAsObject())[opName] = countToJson(count);
        }
    }
    funcObj["quantum_operations"] = std::move(quantumOperationObject);

    /// Function Calls
    llvm::json::Object functionObject;
    llvm::json::Object staticFunctionObject;
    for (const auto &entry : functionCalls) {
        staticFunctionObject[entry.getKey()] = countToJson(entry.getValue());
    }
    llvm::json::Object dynamicFunctionObject;
    for (const auto &entry : varFunctionCalls) {
        dynamicFunctionObject[entry.getKey()] = llvm::formatv("{0:x16}", entry.getValue()).str();
    }
    functionObject["static"] = std::move(staticFunctionObject);
    functionObject["dynamic"] = std::move(dynamicFunctionObject);
    funcObj["function_calls"] = std::move(functionObject);

    /// Measurement processes
    llvm::json::Object measurementObject;
    for (const auto &entry : measurements) {
        measurementObject[entry.getKey()] = countToJson(entry.getValue());
    }
    funcObj["measurement_processes"] = std::move(measurementObject);

    // Extention fields
    // Emit registered extensions under their own keys (e.g. "depth").
    // Stage 4 will nest these under "extended_fields".
    json::Object extendedFieldObject;
    for (const auto &ext : extensions) {
        auto json = ext->toJson();
        if (json != nullptr) {
            extendedFieldObject[ext->name()] = json;
        }
    }
    funcObj["extended_fields"] = std::move(extendedFieldObject);

    // Detail operations
    llvm::json::Object detailedOperationsObject;
    for (const auto &entry : detailedOperations) {
        detailedOperationsObject[entry.getKey()] = countToJson(entry.getValue());
    }
    if (collectDetailedOperations) {
        funcObj["quantum_operations_detailed"] = std::move(detailedOperationsObject);
    }

    return funcObj;
}

/**
 * @brief Build a DictionaryAttr from a ResourceResult for annotating functions.
 *
 * The structure of the DictionaryAttr will mirror the JSON output,
 * but with MLIR attributes.
 *
 * Note that this is a simplified version of the ResourceResult
 * only including operations, measurements, and num_qubits,
 * but it can be extended to include more fields such as
 * classical instructions and function calls as needed
 * for the decomposition framework.
 *
 * @param ctx MLIRContext for creating attributes
 * @param result The ResourceResult to convert into attributes
 * @return DictionaryAttr representing the resource counts
 *
 */
DictionaryAttr buildResourceDict(MLIRContext *ctx, const ResourceResult &result) {

    assert(result.collectDetailedOperations && "detailedOperations should be collected");

    SmallVector<NamedAttribute> entries;
    // operations
    SmallVector<NamedAttribute> opsEntries;
    for (const auto &opEntry : result.detailedOperations) {
        llvm::StringRef opName = opEntry.getKey();
        int64_t count = static_cast<int64_t>(std::llround(opEntry.getValue()));
        opsEntries.push_back(NamedAttribute(StringAttr::get(ctx, opName),
                                            IntegerAttr::get(IntegerType::get(ctx, 64), count)));
    }
    entries.push_back(
        NamedAttribute(StringAttr::get(ctx, "operations"), DictionaryAttr::get(ctx, opsEntries)));

    // measurements
    SmallVector<NamedAttribute> measEntries;
    for (const auto &entry : result.measurements) {
        int64_t count = static_cast<int64_t>(std::llround(entry.getValue()));
        measEntries.push_back(NamedAttribute(StringAttr::get(ctx, entry.getKey()),
                                             IntegerAttr::get(IntegerType::get(ctx, 64), count)));
    }
    entries.push_back(NamedAttribute(StringAttr::get(ctx, "measurements"),
                                     DictionaryAttr::get(ctx, measEntries)));

    // scalars
    entries.push_back(
        NamedAttribute(StringAttr::get(ctx, "num_qubits"),
                       IntegerAttr::get(IntegerType::get(ctx, 64),
                                        static_cast<int64_t>(std::llround(result.numQubits())))));
    entries.push_back(
        NamedAttribute(StringAttr::get(ctx, "num_arg_qubits"),
                       IntegerAttr::get(IntegerType::get(ctx, 64), result.numArgQubits)));
    entries.push_back(NamedAttribute(
        StringAttr::get(ctx, "num_alloc_qubits"),
        IntegerAttr::get(IntegerType::get(ctx, 64),
                         static_cast<int64_t>(std::llround(result.numAllocQubits)))));

    return DictionaryAttr::get(ctx, entries);
}

} // namespace catalyst
