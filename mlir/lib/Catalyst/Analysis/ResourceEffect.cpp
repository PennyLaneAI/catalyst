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

#include "Catalyst/Analysis/ResourceEffect.h"

#include <algorithm>

#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/WithColor.h"

using namespace mlir;

namespace catalyst {

namespace {

class EffectWithVerifier : public ResourceEffect {
  public:
    explicit EffectWithVerifier(OptionsVerifier verifier) : verifier(std::move(verifier)) {}

    LogicalResult verify(const Pass &pass, Location loc) const override {
        return verifier ? verifier(pass, loc) : success();
    }

  private:
    OptionsVerifier verifier;
};

class IdentityEffect : public EffectWithVerifier {
  public:
    IdentityEffect() : EffectWithVerifier(nullptr) {}

    void apply(ResourceResult &, const Pass &) const override {}
};

class SubstitutionEffect : public EffectWithVerifier {
  public:
    SubstitutionEffect(SubstitutionRule rule, OptionsVerifier verifier)
        : EffectWithVerifier(std::move(verifier)), rule(std::move(rule)) {}

    void apply(ResourceResult &function, const Pass &pass) const override {
        ResourceResult::OperationCounts operations;
        double extraPeak = 0;
        for (const auto &opEntry : function.operations) {
            for (const auto &[size, count] : opEntry.getValue()) {
                std::optional<Substitution> substitution =
                    rule({opEntry.getKey().str(), size.first, size.second}, pass);
                if (!substitution) {
                    operations[opEntry.getKey()][size] += count;
                    continue;
                }
                for (const auto &[op, opCount] : substitution->ops) {
                    operations[op.name][{op.numWires, op.numParams}] += opCount * count;
                }
                function.numAllocQubits += substitution->wires.alloc * count;
                if (count != 0) {
                    extraPeak = std::max(extraPeak, substitution->wires.peak);
                }
            }
        }
        function.operations = std::move(operations);
        function.peakQubits += extraPeak;
    }

  private:
    SubstitutionRule rule;
};

class TransferEffect : public EffectWithVerifier {
  public:
    TransferEffect(TransferFunction transfer, OptionsVerifier verifier)
        : EffectWithVerifier(std::move(verifier)), transfer(std::move(transfer)) {}

    void apply(ResourceResult &function, const Pass &pass) const override {
        transfer(function, pass);
    }

  private:
    TransferFunction transfer;
};

} // namespace

std::unique_ptr<ResourceEffect> makeIdentityEffect() { return std::make_unique<IdentityEffect>(); }

std::unique_ptr<ResourceEffect> makeSubstitutionEffect(SubstitutionRule rule,
                                                       OptionsVerifier verifier) {
    return std::make_unique<SubstitutionEffect>(std::move(rule), std::move(verifier));
}

std::unique_ptr<ResourceEffect> makeTransferEffect(TransferFunction transfer,
                                                   OptionsVerifier verifier) {
    return std::make_unique<TransferEffect>(std::move(transfer), std::move(verifier));
}

static llvm::ManagedStatic<ResourceEffectRegistry> registry;

ResourceEffectRegistry &ResourceEffectRegistry::get() { return *registry; }

void ResourceEffectRegistry::add(llvm::StringRef passArgument,
                                 std::unique_ptr<ResourceEffect> effect) {
    if (!effects.try_emplace(passArgument, std::move(effect)).second) {
        llvm::WithColor::warning() << "A resource effect for the pass '" << passArgument
                                   << "' is already registered; ignoring duplicate\n";
    }
}

const ResourceEffect *ResourceEffectRegistry::lookup(llvm::StringRef passArgument) const {
    auto it = effects.find(passArgument);
    return it == effects.end() ? nullptr : it->second.get();
}

void registerUpstreamResourceEffects() {
    [[maybe_unused]] static const bool once = [] {
        ResourceEffectRegistry::get().add("symbol-dce", makeIdentityEffect());
        return true;
    }();
}

} // namespace catalyst
