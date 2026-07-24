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

#include "llvm/ADT/Hashing.h"
#include "llvm/Support/ErrorHandling.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"

using namespace mlir;

using namespace llvm;

namespace catalyst {

using MergeMethod = ResourceResult::MergeMethod;

/// Helper: select merge function based on method enum.
template <typename T> static T applyMerge(T a, T b, MergeMethod method)
{
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
template <typename Map> static void mergeStringMap(Map &dst, const Map &src, MergeMethod method)
{
    for (const auto &entry : src) {
        auto &slot = dst[entry.getKey()];
        slot = applyMerge(slot, entry.getValue(), method);
    }
}

void ResourceResult::mergeWith(const ResourceResult &other, MergeMethod method)
{
    for (const auto &opEntry : other.operations) {
        auto &innerDst = operations[opEntry.getKey()];
        for (const auto &sizeEntry : opEntry.getValue()) {
            auto &slot = innerDst[sizeEntry.first];
            slot = applyMerge(slot, sizeEntry.second, method);
        }
    }

    mergeStringMap(measurements, other.measurements, method);
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
    numArgQubits = applyMerge(numArgQubits, other.numArgQubits, method);

    hasBranches = hasBranches || other.hasBranches;
    hasDynLoop = hasDynLoop || other.hasDynLoop;
}

void ResourceResult::multiplyByScalar(double scalar)
{
    for (auto &opEntry : operations) {
        for (auto &sizeEntry : opEntry.getValue()) {
            sizeEntry.second *= scalar;
        }
    }

    for (auto &entry : measurements) {
        entry.getValue() *= scalar;
    }

    for (auto &entry : classicalInstructions) {
        entry.getValue() *= scalar;
    }

    for (auto &entry : functionCalls) {
        entry.getValue() *= scalar;
    }

    numAllocQubits *= scalar;
    // TODO: does it make sense to scale the arg qubit number?
    // numArgQubits *= scalar;
}

// Emit a count as a JSON number. Counts are tracked as doubles to support probabilistic
// (fractional) count values, although generally they are whole numbers. To keep the output
// clean only print in floating-point format when the value is non-integral.
static llvm::json::Value countToJson(double count)
{
    double rounded = std::nearbyint(count);
    if (count == rounded && std::abs(count) < 9.007199254740992e15 /* 2^53 */) {
        return llvm::json::Value(static_cast<int64_t>(rounded));
    }
    return llvm::json::Value(count);
}

llvm::json::Object ResourceResult::toJson() const
{
    llvm::json::Object funcObj;

    llvm::json::Object opsObj;
    for (const auto &opEntry : operations) {
        StringRef opName = opEntry.getKey();
        for (const auto &sizeEntry : opEntry.getValue()) {
            const auto &[nQubits, nParams] = sizeEntry.first;
            double count = sizeEntry.second;
            std::string key = opName.str() + "(" + std::to_string(nQubits) + ")";
            opsObj[key] = countToJson(count);
        }
    }
    funcObj["operations"] = std::move(opsObj);

    llvm::json::Object measObj;
    for (const auto &entry : measurements) {
        measObj[entry.getKey()] = countToJson(entry.getValue());
    }
    funcObj["measurements"] = std::move(measObj);

    llvm::json::Object classObj;
    for (const auto &entry : classicalInstructions) {
        classObj[entry.getKey()] = countToJson(entry.getValue());
    }
    funcObj["classical_instructions"] = std::move(classObj);

    llvm::json::Object fcObj;
    for (const auto &entry : functionCalls) {
        fcObj[entry.getKey()] = countToJson(entry.getValue());
    }
    funcObj["function_calls"] = std::move(fcObj);

    llvm::json::Object vfcObj;
    for (const auto &entry : varFunctionCalls) {
        vfcObj[entry.getKey()] = llvm::formatv("{0:x16}", entry.getValue()).str();
    }
    funcObj["var_function_calls"] = std::move(vfcObj);

    funcObj["num_qubits"] = countToJson(numQubits());
    funcObj["num_alloc_qubits"] = countToJson(numAllocQubits);
    funcObj["num_arg_qubits"] = numArgQubits;
    funcObj["device_name"] = deviceName;
    funcObj["qnode"] = isQnode;
    funcObj["has_branches"] = hasBranches;
    if (autoQubitManagement.has_value()) {
        funcObj["auto_qubit_management"] = *autoQubitManagement;
    }
    llvm::json::Object depthObj;
    if (pbcDepth) {
        depthObj["any_commuting_depth"] = pbcDepth->first;
        depthObj["qubit_disjoint_depth"] = pbcDepth->second;
    }
    funcObj["depth"] = std::move(depthObj);

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
DictionaryAttr buildResourceDict(MLIRContext *ctx, const ResourceResult &result)
{
    // TODO: maintain int counts here for now, but this whole function is deprecated
    SmallVector<NamedAttribute> entries;

    // operations
    SmallVector<NamedAttribute> opsEntries;
    for (const auto &opEntry : result.operations) {
        llvm::StringRef opName = opEntry.getKey();
        for (const auto &sizeEntry : opEntry.getValue()) {
            auto &[nQubits, nParams] = sizeEntry.first;
            int64_t count = static_cast<int64_t>(std::llround(sizeEntry.second));
            std::string key =
                (opName + "(" + std::to_string(nQubits) + "," + std::to_string(nParams) + ")")
                    .str();
            opsEntries.push_back(NamedAttribute(
                StringAttr::get(ctx, key), IntegerAttr::get(IntegerType::get(ctx, 64), count)));
        }
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
