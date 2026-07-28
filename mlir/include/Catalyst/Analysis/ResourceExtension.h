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
#include <type_traits>

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/JSON.h"
#include "mlir/IR/Operation.h" // Operation, Region

namespace catalyst {

// Define in ResourceResult.h.
enum class MergeMethod;

// Value object for an optional resource metric. Lives in ResourceResult.
class ResourceExtension {
  public:
    virtual ~ResourceExtension() = default;
    virtual llvm::StringRef name() const = 0;
    virtual llvm::json::Value toJson() const = 0;

    virtual void mergeWith(const ResourceExtension &other, MergeMethod mergeMethod) {}
    virtual void multiplyBy(int64_t factor) {}
};

template <typename Ext> class ResourceExtensionAnalysisOf;

// Owned by ResourceAnalysis for the duration of a run; writes into per-result data.
// Do not inherit directly, use ResourceExtensionAnalysisOf<Ext>.
class ResourceExtensionAnalysis {
    template <typename Ext> friend class ResourceExtensionAnalysisOf;

    ResourceExtensionAnalysis() = default;

  public:
    virtual ~ResourceExtensionAnalysis() = default;
    ResourceExtensionAnalysis(const ResourceExtensionAnalysis &) = delete;
    ResourceExtensionAnalysis &operator=(const ResourceExtensionAnalysis &) = delete;

    virtual llvm::StringRef name() const = 0; // must be matched to ResourceExtension::name()

    // Mint an empty data object for a new ResourceResult.
    virtual std::unique_ptr<ResourceExtension> makeEmpty() const = 0;

    // Walk through each individual operation.
    virtual void collect(mlir::Operation *op, ResourceExtension &ext, bool isAdjoint) {}

    // Walk through the each region.
    virtual void analyze(mlir::Region &region, ResourceExtension &ext, bool isAdjoint) {}
};

// Subclasses override the typed collect / analyze overloads.
template <typename Ext> class ResourceExtensionAnalysisOf : public ResourceExtensionAnalysis {
  public:
    static_assert(std::is_base_of_v<ResourceExtension, Ext>,
                  "Ext must derive from ResourceExtension");

    std::unique_ptr<ResourceExtension> makeEmpty() const final { return std::make_unique<Ext>(); }

    void collect(mlir::Operation *op, ResourceExtension &ext, bool isAdjoint) final
    {
        collect(op, static_cast<Ext &>(ext), isAdjoint);
    }

    void analyze(mlir::Region &region, ResourceExtension &ext, bool isAdjoint) final
    {
        analyze(region, static_cast<Ext &>(ext), isAdjoint);
    }

  protected:
    virtual void collect(mlir::Operation * /*op*/, Ext & /*ext*/, bool /*isAdjoint*/) {}
    virtual void analyze(mlir::Region & /*region*/, Ext & /*ext*/, bool /*isAdjoint*/) {}
};

} // namespace catalyst
