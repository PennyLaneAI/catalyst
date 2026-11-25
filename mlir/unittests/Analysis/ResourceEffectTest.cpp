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

#include "gtest/gtest.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;
using namespace catalyst;

namespace {

struct TestPass : public PassWrapper<TestPass, OperationPass<>> {
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestPass)

    void runOnOperation() override {}

    int multiplicity = 2;
};

ResourceResult makeFunction() {
    ResourceResult function;
    function.operations["A"][{1, 0}] = 3;
    function.operations["C"][{2, 0}] = 1;
    function.numAllocQubits = 5;
    function.peakQubits = 5;
    return function;
}

// A -> B x multiplicity on one new qubit; everything else is kept.
std::optional<Substitution> replaceA(const OpKey &op, const Pass &pass) {
    if (op.name != "A") {
        return std::nullopt;
    }
    int multiplicity = static_cast<const TestPass &>(pass).multiplicity;
    return Substitution{{{{"B", op.numWires, -1}, static_cast<double>(multiplicity)}},
                        {/*alloc=*/1, /*peak=*/1}};
}

TEST(ResourceEffect, SubstitutionReplacesMatchingOperations) {
    TestPass pass;
    ResourceResult function = makeFunction();
    makeSubstitutionEffect(replaceA)->apply(function, pass);

    EXPECT_FALSE(function.operations.contains("A"));
    EXPECT_EQ((function.operations["B"][{1, -1}]), 6);
    EXPECT_EQ((function.operations["C"][{2, 0}]), 1);
    EXPECT_EQ(function.numAllocQubits, 8);
    EXPECT_EQ(function.peakQubits, 6);
}

TEST(ResourceEffect, SubstitutionMergesWithExistingOperations) {
    TestPass pass;
    ResourceResult function = makeFunction();
    function.operations["B"][{1, -1}] = 4;
    makeSubstitutionEffect(replaceA)->apply(function, pass);

    EXPECT_EQ((function.operations["B"][{1, -1}]), 10);
}

TEST(ResourceEffect, SubstitutionCanRemoveOperations) {
    TestPass pass;
    ResourceResult function = makeFunction();
    auto removeC = [](const OpKey &op, const Pass &) -> std::optional<Substitution> {
        return op.name == "C" ? std::optional<Substitution>(Substitution{}) : std::nullopt;
    };
    makeSubstitutionEffect(removeC)->apply(function, pass);

    EXPECT_FALSE(function.operations.contains("C"));
    EXPECT_EQ((function.operations["A"][{1, 0}]), 3);
    EXPECT_EQ(function.numAllocQubits, 5);
    EXPECT_EQ(function.peakQubits, 5);
}

TEST(ResourceEffect, SubstitutionOfAbsentOperationKeepsPeak) {
    TestPass pass;
    ResourceResult function = makeFunction();
    function.operations["A"][{1, 0}] = 0;
    makeSubstitutionEffect(replaceA)->apply(function, pass);

    EXPECT_EQ(function.peakQubits, 5);
}

TEST(ResourceEffect, IdentityKeepsResources) {
    TestPass pass;
    ResourceResult function = makeFunction();
    makeIdentityEffect()->apply(function, pass);

    EXPECT_EQ((function.operations["A"][{1, 0}]), 3);
    EXPECT_EQ((function.operations["C"][{2, 0}]), 1);
    EXPECT_EQ(function.numAllocQubits, 5);
}

TEST(ResourceEffect, TransferUpdatesFunction) {
    TestPass pass;
    ResourceResult function = makeFunction();
    auto dropA = [](ResourceResult &resources, const Pass &) { resources.operations.erase("A"); };
    makeTransferEffect(dropA)->apply(function, pass);

    EXPECT_FALSE(function.operations.contains("A"));
    EXPECT_TRUE(function.operations.contains("C"));
}

TEST(ResourceEffect, VerifyUsesOptionsVerifier) {
    MLIRContext context;
    TestPass pass;
    Location loc = UnknownLoc::get(&context);
    EXPECT_TRUE(succeeded(makeIdentityEffect()->verify(pass, loc)));

    auto rejectAll = [](const Pass &, Location) { return failure(); };
    EXPECT_TRUE(failed(makeSubstitutionEffect(replaceA, rejectAll)->verify(pass, loc)));
    EXPECT_TRUE(failed(
        makeTransferEffect([](ResourceResult &, const Pass &) {}, rejectAll)->verify(pass, loc)));
}

TEST(ResourceEffectRegistry, LookupByPassArgument) {
    ResourceEffectRegistry &registry = ResourceEffectRegistry::get();
    EXPECT_EQ(registry.lookup("test-unregistered-pass"), nullptr);

    auto effect = makeIdentityEffect();
    const ResourceEffect *registered = effect.get();
    registry.add("test-registered-pass", std::move(effect));
    EXPECT_EQ(registry.lookup("test-registered-pass"), registered);

    registry.add("test-registered-pass", makeIdentityEffect());
    EXPECT_EQ(registry.lookup("test-registered-pass"), registered);
}

} // namespace
