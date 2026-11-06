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
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Support/TypeID.h"

#include "Quantum/IR/QuantumDialect.h"

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

TEST(ResourceAnalysis, SingleFunctionControlledRecursionStopsExpansion) {
    MLIRContext context;
    context.loadDialect<arith::ArithDialect, func::FuncDialect, quantum::QuantumDialect>();
    auto module = parseSourceString<ModuleOp>(R"mlir(
        func.func @recursive(%q: !quantum.bit, %c: !quantum.bit)
            -> (!quantum.bit, !quantum.bit) {
            %true = arith.constant true
            %input = quantum.custom "PauliX"() %q : !quantum.bit
            %oc, %oq = quantum.ctrl(%c) ctrlvals(%true) (%input)
                : !quantum.bit -> !quantum.bit {
            ^bb0(%a: !quantum.bit):
                %r:2 = func.call @recursive(%a, %c)
                    : (!quantum.bit, !quantum.bit) -> (!quantum.bit, !quantum.bit)
                quantum.yield %r#0 : !quantum.bit
            }
            return %oq, %oc : !quantum.bit, !quantum.bit
        }
    )mlir",
                                              &context);
    ASSERT_TRUE(module);
    auto function = module->lookupSymbol<func::FuncOp>("recursive");

    unsigned warnings = 0;
    ScopedDiagnosticHandler handler(&context, [&](Diagnostic &diagnostic) {
        EXPECT_EQ(diagnostic.getSeverity(), DiagnosticSeverity::Warning);
        EXPECT_NE(diagnostic.str().find("Recursive calls are not flattened"), std::string::npos);
        ++warnings;
        return success();
    });

    for (bool detailed : {false, true}) {
        ResourceAnalysis analysis(function, {}, detailed);
        EXPECT_EQ(analysis.getResults().size(), 1u);
        const auto *result = analysis.getResult("recursive");
        ASSERT_NE(result, nullptr);
        EXPECT_EQ(result->modifierContexts.size(), 1u);
        EXPECT_EQ(result->functionCalls.lookup("recursive"), 1);

        const auto *flat = analysis.getFlattenedResource("recursive");
        ASSERT_NE(flat, nullptr);
        EXPECT_EQ(flat->functionCalls.lookup("recursive"), 1);
        EXPECT_EQ(flat->operations.lookup("PauliX").lookup({1, 0}), 1);
    }
    EXPECT_EQ(warnings, 2u);
}

TEST(ResourceAnalysis, MutualRecursionKeepsExternalModifiersAndNonrecursiveCalls) {
    MLIRContext context;
    context.loadDialect<arith::ArithDialect, func::FuncDialect, scf::SCFDialect,
                        quantum::QuantumDialect>();
    auto module = parseSourceString<ModuleOp>(R"mlir(
        func.func @entry(%q: !quantum.bit, %c: !quantum.bit) -> !quantum.bit
            attributes {llvm.emit_c_interface} {
            %true = arith.constant true
            %oc, %oq = quantum.ctrl(%c) ctrlvals(%true) (%q)
                : !quantum.bit -> !quantum.bit {
            ^bb0(%a: !quantum.bit):
                %out = func.call @first(%a) : (!quantum.bit) -> !quantum.bit
                quantum.yield %out : !quantum.bit
            }
            return %oq : !quantum.bit
        }
        func.func private @first(%q: !quantum.bit) -> !quantum.bit {
            %input = quantum.custom "PauliX"() %q : !quantum.bit
            %lb = arith.constant 0 : index
            %ub = arith.constant 1 : index
            %step = arith.constant 1 : index
            %loop = scf.for %i = %lb to %ub step %step
                iter_args(%a = %input) -> (!quantum.bit) {
                %out = func.call @second(%a) : (!quantum.bit) -> !quantum.bit
                scf.yield %out : !quantum.bit
            }
            %out = func.call @helper(%loop) : (!quantum.bit) -> !quantum.bit
            return %out : !quantum.bit
        }
        func.func private @second(%q: !quantum.bit) -> !quantum.bit {
            %input = quantum.custom "PauliY"() %q : !quantum.bit
            %out = func.call @first(%input) : (!quantum.bit) -> !quantum.bit
            return %out : !quantum.bit
        }
        func.func private @helper(%q: !quantum.bit) -> !quantum.bit {
            %out = quantum.custom "Hadamard"() %q : !quantum.bit
            return %out : !quantum.bit
        }
    )mlir",
                                              &context);
    ASSERT_TRUE(module);

    unsigned warnings = 0;
    ScopedDiagnosticHandler handler(&context, [&](Diagnostic &diagnostic) {
        EXPECT_EQ(diagnostic.getSeverity(), DiagnosticSeverity::Warning);
        EXPECT_NE(diagnostic.str().find("Recursive calls are not flattened"), std::string::npos);
        ++warnings;
        return success();
    });

    for (bool detailed : {false, true}) {
        ResourceAnalysis analysis(*module, {}, detailed);
        EXPECT_EQ(analysis.getResult("first")->modifierContexts.size(), 2u);
        EXPECT_EQ(analysis.getResult("second")->modifierContexts.size(), 1u);
        EXPECT_EQ(analysis.getResult("helper")->modifierContexts.size(), 2u);

        // Query both recursive functions before the entry to check that cached
        // partial results do not depend on which function was flattened first.
        const auto *second = analysis.getFlattenedResource("second");
        ASSERT_NE(second, nullptr);
        EXPECT_EQ(second->operations.size(), 1u);
        EXPECT_EQ(second->operations.lookup("PauliY").lookup({1, 0}), 1);
        const auto *first = analysis.getFlattenedResource("first");
        ASSERT_NE(first, nullptr);
        EXPECT_EQ(first->operations.size(), 2u);
        EXPECT_EQ(first->operations.lookup("PauliX").lookup({1, 0}), 1);
        EXPECT_EQ(first->operations.lookup("Hadamard").lookup({1, 0}), 1);
        EXPECT_EQ(first->functionCalls.lookup("second"), 1);

        const auto *flat = analysis.getFlattenedResource("entry");
        ASSERT_NE(flat, nullptr);
        EXPECT_EQ(flat->operations.size(), 2u);
        EXPECT_EQ(flat->operations.lookup("C(PauliX)").lookup({2, 0}), 1);
        EXPECT_EQ(flat->operations.lookup("C(Hadamard)").lookup({2, 0}), 1);
        EXPECT_EQ(flat->functionCalls.lookup("second"), 1);
        if (detailed) {
            EXPECT_EQ(flat->detailedOperations.size(), 2u);
            EXPECT_EQ(flat->observedGraphOpIds.size(), 2u);
        }
    }
    EXPECT_EQ(warnings, 2u);
}
} // namespace
