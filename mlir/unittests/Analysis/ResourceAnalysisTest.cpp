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

#include "Catalyst/Analysis/ResourceAnalysis.h"

#include <memory>

#include "gtest/gtest.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Support/TypeID.h"

using namespace mlir;
using namespace catalyst;

namespace {
struct SharedAnalysis {
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SharedAnalysis)

    explicit SharedAnalysis(Operation *) {}
    int initializedExtensions = 0;
};

struct TestResult : ResourceResultExtension {
    llvm::StringRef name() const override { return "test"; }
    llvm::json::Value toJson() const override { return nullptr; }
};

struct TestExtension : ResourceAnalysisExtensionOf<TestResult> {
    explicit TestExtension(llvm::SmallVector<int> &observations) : observations(observations) {}

    llvm::StringRef name() const override { return "test"; }

    void initialize(AnalysisManager analysisManager) override {
        sharedAnalysis = &analysisManager.getAnalysis<SharedAnalysis>();
        sharedAnalysis->initializedExtensions = 10;
    }

    void analyze(Region &, TestResult &, bool) override {
        observations.push_back(sharedAnalysis ? sharedAnalysis->initializedExtensions : 0);
    }

    llvm::SmallVector<int> &observations;
    SharedAnalysis *sharedAnalysis = nullptr;
};

TEST(ResourceAnalysis, ExtensionsShareAnalysisBeforeTraversal) {
    MLIRContext context;
    context.loadDialect<func::FuncDialect>();
    auto module = parseSourceString<ModuleOp>("func.func @test() { return }", &context);
    ASSERT_TRUE(module);

    llvm::SmallVector<int> observations;
    auto provider = [&] { return std::make_unique<TestExtension>(observations); };
    ResourceAnalysis analysis(*module, {provider});

    EXPECT_EQ(observations, (llvm::SmallVector<int>{10}));
}
} // namespace
