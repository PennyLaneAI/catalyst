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

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringMap.h"

namespace catalyst {

// method for merging two ResourceResult values
enum class MergeMethod { Sum, Max, Min };

// ResourceResult holds the resource counts for a single function.
// It mirrors the Python-side ResourcesResult from specs_collector.py.
struct ResourceResult {
    // quantum, pbc, mbqc operations
    llvm::StringMap<llvm::DenseMap<int, int64_t>> operations;

    llvm::StringMap<int64_t> measurements;

    llvm::StringMap<int64_t> classicalInstructions;

    llvm::StringMap<int64_t> functionCalls;

    // unresolved function calls (to be inlined later)
    llvm::StringMap<int64_t> unresolvedFunctionCalls;

    // qubits from quantum.alloc / quantum.alloc_qubit ops
    int64_t numAllocQubits = 0;

    // qubits from !quantum.bit function arguments (entry function only)
    int64_t numArgQubits = 0;

    // total qubits (allocated + argument)
    int64_t numQubits() const { return numAllocQubits + numArgQubits; }

    // from quantum.device op
    std::string deviceName;

    // merge another ResourceResult into this one
    void mergeWith(const ResourceResult &other, MergeMethod method = MergeMethod::Sum);

    // multiply all counts by a scalar
    void multiplyByScalar(int64_t scalar);

    std::string toJson(int indent = 4) const;
};

} // namespace catalyst
