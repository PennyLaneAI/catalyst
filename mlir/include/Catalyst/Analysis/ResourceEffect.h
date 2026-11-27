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

#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <utility>

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/IR/Location.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"

#include "Catalyst/Analysis/ResourceResult.h"

namespace catalyst {

/// An operation as `ResourceResult::operations` counts it.
struct OpKey {
    std::string name;
    int numWires;
    int numParams;
};

/// The qubits that the replacement of an operation uses. `alloc` is added to the allocations for
/// every replaced operation. `peak` raises the peak of the function once, since each replacement
/// releases its qubits before the next operation.
struct WireEffect {
    double alloc = 0;
    double peak = 0;
};

/// What replaces an operation. An empty `ops` removes it.
struct Substitution {
    llvm::SmallVector<std::pair<OpKey, double>> ops;
    WireEffect wires = {};
};

/// How a compilation pass changes the resources of a function, declared next to the pass so that
/// the resources after it can be estimated without running it.
class ResourceEffect {
  public:
    virtual ~ResourceEffect() = default;

    /// Check that the effect is declared for the options of `pass`, or emit an error at `loc`.
    virtual mlir::LogicalResult verify(const mlir::Pass &pass, mlir::Location loc) const = 0;

    /// Update the resources of one function for `pass`, configured with its options.
    virtual void apply(ResourceResult &function, const mlir::Pass &pass) const = 0;
};

/// Returns what replaces an operation, or std::nullopt to keep it.
using SubstitutionRule =
    std::function<std::optional<Substitution>(const OpKey &op, const mlir::Pass &pass)>;
using TransferFunction = std::function<void(ResourceResult &function, const mlir::Pass &pass)>;
using OptionsVerifier = std::function<mlir::LogicalResult(const mlir::Pass &, mlir::Location)>;

/// The effect of a pass that changes nothing the analysis counts.
std::unique_ptr<ResourceEffect> makeIdentityEffect();

/// The effect of a pass that replaces each operation independently of the others. `rule` is called
/// once per distinct operation of each function.
std::unique_ptr<ResourceEffect> makeSubstitutionEffect(SubstitutionRule rule,
                                                       OptionsVerifier verifier = nullptr);

/// The effect of a pass that needs a whole function's resources, but not their order.
std::unique_ptr<ResourceEffect> makeTransferEffect(TransferFunction transfer,
                                                   OptionsVerifier verifier = nullptr);

/// The resource effects of compilation passes, by pass argument (e.g. "to-ppr").
class ResourceEffectRegistry {
  public:
    static ResourceEffectRegistry &get();

    void add(llvm::StringRef passArgument, std::unique_ptr<ResourceEffect> effect);

    /// Returns nullptr if no effect is registered for `passArgument`.
    const ResourceEffect *lookup(llvm::StringRef passArgument) const;

  private:
    llvm::StringMap<std::unique_ptr<ResourceEffect>> effects;
};

/// Register the resource effects of the upstream MLIR passes that Catalyst pipelines use.
void registerUpstreamResourceEffects();

} // namespace catalyst
