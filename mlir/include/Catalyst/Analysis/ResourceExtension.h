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

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/JSON.h"
#include "mlir/IR/Operation.h" // Operation, Region

namespace catalyst {

enum class MergeMethod { Sum, Max, Min }; // hoisted out of the struct

/// Value object for an optional resource metric. Lives in ResourceResult.
class ResourceExtension {
  public:
    virtual ~ResourceExtension() = default;
    virtual llvm::StringRef name() const = 0;
    virtual llvm::json::Value toJson() const = 0;

    virtual void mergeWith(const ResourceExtension &other, MergeMethod mergeMethod) {}
    virtual void multiplyBy(int64_t factor) {}
};

/// Owned by ResourceAnalysis for the duration of a run; writes into per-result data.
class ResourceExtensionAnalysis {
  public:
    virtual ~ResourceExtensionAnalysis() = default;
    virtual llvm::StringRef name() const = 0; // must be match with ResourceExtension::name()

    /// Mint an empty data object for a new ResourceResult.
    virtual std::unique_ptr<ResourceExtension> makeEmpty() const = 0;

    /// Per-op hook (e.g. uid_map). Walk through each individual operation.
    virtual void collect(mlir::Operation *op, ResourceExtension &ext, bool isAdjoint) {}

    /// Whole-region hook (e.g. depth). Walk through the whole region.
    virtual void analyze(mlir::Region &region, ResourceExtension &ext, bool isAdjoint) {}
};

} // namespace catalyst
