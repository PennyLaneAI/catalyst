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
class ResourceResultExtension {
  public:
    virtual ~ResourceResultExtension() = default;
    virtual llvm::StringRef name() const = 0;
    virtual llvm::json::Value toJson() const = 0;

    // Override both if your extension accumulates state in `collect`.
    // Skip them if it recomputes per region in `analyze` (e.g. PBCDepthExtension).
    virtual void mergeWith(const ResourceResultExtension &other, MergeMethod mergeMethod) {}

    // `factor` may be fractional (e.g. probabilistic branch weighting).
    virtual void multiplyBy(double factor) {}
};

template <typename Ext> class ResourceAnalysisExtensionOf;

// Owned by ResourceAnalysis for the duration of a run; writes into per-result data.
// Do not inherit directly, use ResourceAnalysisExtensionOf<Ext>.
class ResourceAnalysisExtension {
    template <typename Ext> friend class ResourceAnalysisExtensionOf;

    ResourceAnalysisExtension() = default;

  public:
    virtual ~ResourceAnalysisExtension() = default;
    ResourceAnalysisExtension(const ResourceAnalysisExtension &) = delete;
    ResourceAnalysisExtension &operator=(const ResourceAnalysisExtension &) = delete;

    virtual llvm::StringRef name() const = 0; // must be matched to ResourceResultExtension::name()

    // Mint an empty data object for a new ResourceResult.
    virtual std::unique_ptr<ResourceResultExtension> makeEmpty() const = 0;

    // Walk through each individual operation.
    virtual void collect(mlir::Operation *op, ResourceResultExtension &ext, bool isAdjoint) {}

    // Walk through the each region.
    virtual void analyze(mlir::Region &region, ResourceResultExtension &ext, bool isAdjoint) {}
};

// Subclasses override the typed collect / analyze overloads.
template <typename Ext> class ResourceAnalysisExtensionOf : public ResourceAnalysisExtension {
  public:
    static_assert(std::is_base_of_v<ResourceResultExtension, Ext>,
                  "Ext must derive from ResourceResultExtension");

    std::unique_ptr<ResourceResultExtension> makeEmpty() const final {
        return std::make_unique<Ext>();
    }

    void collect(mlir::Operation *op, ResourceResultExtension &ext, bool isAdjoint) final {
        collect(op, static_cast<Ext &>(ext), isAdjoint);
    }

    void analyze(mlir::Region &region, ResourceResultExtension &ext, bool isAdjoint) final {
        analyze(region, static_cast<Ext &>(ext), isAdjoint);
    }

  protected:
    virtual void collect(mlir::Operation * /*op*/, Ext & /*ext*/, bool /*isAdjoint*/) {}
    virtual void analyze(mlir::Region & /*region*/, Ext & /*ext*/, bool /*isAdjoint*/) {}
};

} // namespace catalyst
