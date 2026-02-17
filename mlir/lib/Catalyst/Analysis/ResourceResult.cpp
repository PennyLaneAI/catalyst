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
#include <sstream>

#include "llvm/Support/JSON.h"

using namespace llvm;

namespace catalyst {

/// Helper: select merge function based on method enum.
static int64_t applyMerge(int64_t a, int64_t b, MergeMethod method)
{
    switch (method) {
    case MergeMethod::Max:
        return std::max(a, b);
    case MergeMethod::Min:
        return std::min(a, b);
    case MergeMethod::Sum:
        return a + b;
    }
    return a + b; // unreachable, silences warning
}

void ResourceResult::mergeWith(const ResourceResult &other, MergeMethod method)
{
    for (const auto &opEntry : other.operations) {
        StringRef opName = opEntry.getKey();
        for (const auto &sizeEntry : opEntry.getValue()) {
            int nQubits = sizeEntry.first;
            int64_t count = sizeEntry.second;
            operations[opName][nQubits] = applyMerge(operations[opName][nQubits], count, method);
        }
    }

    for (const auto &entry : other.measurements) {
        measurements[entry.getKey()] =
            applyMerge(measurements[entry.getKey()], entry.getValue(), method);
    }

    for (const auto &entry : other.classicalInstructions) {
        classicalInstructions[entry.getKey()] =
            applyMerge(classicalInstructions[entry.getKey()], entry.getValue(), method);
    }

    for (const auto &entry : other.functionCalls) {
        functionCalls[entry.getKey()] =
            applyMerge(functionCalls[entry.getKey()], entry.getValue(), method);
    }

    for (const auto &entry : other.unresolvedFunctionCalls) {
        unresolvedFunctionCalls[entry.getKey()] =
            applyMerge(unresolvedFunctionCalls[entry.getKey()], entry.getValue(), method);
    }

    if (deviceName.empty() && !other.deviceName.empty()) {
        deviceName = other.deviceName;
    }
    numAllocQubits = applyMerge(numAllocQubits, other.numAllocQubits, method);
    numArgQubits = applyMerge(numArgQubits, other.numArgQubits, method);
}

void ResourceResult::multiplyByScalar(int64_t scalar)
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

    for (auto &entry : unresolvedFunctionCalls) {
        entry.getValue() *= scalar;
    }

    numAllocQubits *= scalar;
    numArgQubits *= scalar;
}

std::string ResourceResult::toJson(int indent) const
{
    // Build JSON object using llvm::json
    llvm::json::Object root;

    // Operations: flatten to "OpName(nqubits)" -> count
    llvm::json::Object opsObj;
    for (const auto &opEntry : operations) {
        StringRef opName = opEntry.getKey();
        for (const auto &sizeEntry : opEntry.getValue()) {
            int nQubits = sizeEntry.first;
            int64_t count = sizeEntry.second;
            std::string key = (opName + "(" + std::to_string(nQubits) + ")").str();
            opsObj[key] = count;
        }
    }
    root["operations"] = std::move(opsObj);

    // Measurements
    llvm::json::Object measObj;
    for (const auto &entry : measurements) {
        measObj[entry.getKey()] = entry.getValue();
    }
    root["measurements"] = std::move(measObj);

    // Classical instructions
    llvm::json::Object classObj;
    for (const auto &entry : classicalInstructions) {
        classObj[entry.getKey()] = entry.getValue();
    }
    root["classical_instructions"] = std::move(classObj);

    // Function calls
    llvm::json::Object fcObj;
    for (const auto &entry : functionCalls) {
        fcObj[entry.getKey()] = entry.getValue();
    }
    root["function_calls"] = std::move(fcObj);

    root["num_qubits"] = numQubits();
    root["num_alloc_qubits"] = numAllocQubits;
    root["num_arg_qubits"] = numArgQubits;
    root["device_name"] = deviceName;

    llvm::json::Value jsonValue(std::move(root));
    std::string result;
    llvm::raw_string_ostream os(result);
    os << formatv("{0:2}", jsonValue);
    os.flush();
    return result;
}

} // namespace catalyst
