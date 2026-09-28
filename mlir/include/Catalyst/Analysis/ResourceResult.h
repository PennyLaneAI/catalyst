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

#include <optional>
#include <utility>

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/JSON.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/MLIRContext.h"

#include "Catalyst/Analysis/ResourceResultExtension.h"

#include <llvm/ADT/StringRef.h>

namespace catalyst {

enum class MergeMethod { Sum, Max, Min };

// ResourceResult holds the resource counts for a single function.
// It mirrors the Python-side ResourcesResult from specs_collector.py.
struct ResourceResult {
    // method for merging two ResourceResult values
    using MergeMethod = catalyst::MergeMethod;

    // quantum, qref, pbc, mbqc operations are stored
    // as a map from operation name to a map of
    // name -> ((numWires, numParams) -> count)
    llvm::StringMap<llvm::DenseMap<std::pair<int, int>, double>> operations;

    llvm::StringMap<double> detailedOperations;

    llvm::StringMap<double> measurements;

    llvm::StringMap<double> classicalInstructions;

    llvm::StringMap<double> functionCalls;

    // `dyn_for_loop_<N>` -> stable hash id for that loop op (not a trip count).
    // Ignored by `multiplyBy`; `mergeWith` mints a fresh id on key conflicts.
    llvm::StringMap<uint64_t> varFunctionCalls;

    // qubits from qref/quantum alloc/alloc_qubit ops
    double numAllocQubits = 0;

    // qubits from !quantum.bit, qref.bit and qref.reg<{static}> function arguments (entry function
    // only)
    int64_t numArgQubits = 0;

    // total qubits (allocated + argument)
    double numQubits() const { return numAllocQubits + numArgQubits; }

    // from quantum.device op
    std::string deviceName;

    // whether this function carries the `quantum.node` attribute
    bool isQnode = false;

    // whether the function contains conditional control flow (scf.if / scf.index_switch)
    bool hasBranches = false;

    // whether any loop has a trip count that could not be statically resolved
    bool hasDynLoop = false;

    // If this is true, detailedOperations is included during serialization, otherwise it is
    // omitted.
    bool collectDetailedOperations = false;

    // Set when quantum.device is present: true if {auto_qubit_management} is
    // active (register grows dynamically on quantum.extract/qref.get), false if not.
    // nullopt means no quantum.device in this function.
    std::optional<bool> autoQubitManagement;

    // merge another ResourceResult into this one
    void mergeWith(const ResourceResult &other, MergeMethod method = MergeMethod::Sum);

    // multiply all counts by a scalar, which may be be fractional to account for probabilistic
    // counting sometimes employed in branches for example
    void multiplyBy(double scalar);

    // Serialize this function's resources into a JSON object.
    llvm::json::Object toJson() const;

    // It is used to store additional resource information
    // that is not covered by the core resource result
    llvm::SmallVector<std::unique_ptr<ResourceResultExtension>> extensions;

    ResourceResult() = default;
    ResourceResult(ResourceResult &&) = default;
    ResourceResult &operator=(ResourceResult &&) = default;
};

mlir::DictionaryAttr buildResourceDict(mlir::MLIRContext *ctx, const ResourceResult &result);

} // namespace catalyst
