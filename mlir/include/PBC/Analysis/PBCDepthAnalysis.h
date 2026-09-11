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
#include <optional>
#include <utility>

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/JSON.h"

#include "Catalyst/Analysis/ResourceResultExtension.h"

namespace catalyst {
namespace pbc {

// PBC depth data stored in ResourceResult::extensions.
class PBCDepthExtension : public ResourceResultExtension {
  public:
    llvm::StringRef name() const override { return "pbc_depth"; }

    llvm::json::Value toJson() const override;

    void mergeWith(const ResourceResultExtension &other, MergeMethod mergeMethod) override;
    void multiplyBy(double factor) override;

    void setDepth(std::optional<std::pair<int64_t, int64_t>> value) { depth = std::move(value); }

  private:
    // (any_commuting_depth, qubit_disjoint_depth), or nullopt if unavailable.
    std::optional<std::pair<int64_t, int64_t>> depth;
};

// Computes PBC depth for a region into a PBCDepthExtension.
class PBCDepthAnalysis : public ResourceAnalysisExtensionOf<PBCDepthExtension> {
  public:
    llvm::StringRef name() const override { return "pbc_depth"; }

  protected:
    void analyze(mlir::Region &region, PBCDepthExtension &ext, bool isAdjoint) override;
};

// Registers PBCDepthAnalysis into ResourceAnalysisRegistry
void registerPBCResourceAnalysisExtensions();

} // namespace pbc
} // namespace catalyst
