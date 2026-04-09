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

#include "gtest/gtest.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Parser/Parser.h"

#include "Quantum/IR/QuantumOps.h"
#include "Quantum/Utils/QubitTracing.h"

using namespace mlir;
using namespace catalyst::quantum;

namespace {

void registerDialects(DialectRegistry &registry)
{
    registry.insert<arith::ArithDialect, func::FuncDialect, scf::SCFDialect,
                    catalyst::quantum::QuantumDialect>();
}

TEST(TraceQubitTests, directExtract)
{
    std::string moduleStr = R"mlir(
module {
  func.func @f() {
    %reg = quantum.alloc(1) : !quantum.reg
    %q = quantum.extract %reg[0] : !quantum.reg -> !quantum.bit
    %obs = quantum.namedobs %q[PauliZ] : !quantum.obs
    return
  }
}
)mlir";

    DialectRegistry registry;
    registerDialects(registry);
    MLIRContext context(registry);
    ParserConfig config(&context, /*verifyAfterParse=*/false);
    OwningOpRef<ModuleOp> mod = parseSourceString<ModuleOp>(moduleStr, config);
    ASSERT_TRUE(mod);

    func::FuncOp f = *(*mod).getOps<func::FuncOp>().begin();
    auto namedObs = *f.getOps<NamedObsOp>().begin();
    auto extract = *f.getOps<ExtractOp>().begin();

    Value traced = traceQubit(namedObs.getQubit());
    ASSERT_TRUE(traced);
    EXPECT_EQ(traced, extract.getResult());
}

TEST(TraceQubitTests, throughSingleQubitGate)
{
    std::string moduleStr = R"mlir(
module {
  func.func @f() {
    %reg = quantum.alloc(1) : !quantum.reg
    %q = quantum.extract %reg[0] : !quantum.reg -> !quantum.bit
    %qh = quantum.custom "Hadamard"() %q : !quantum.bit
    %obs = quantum.namedobs %qh[PauliZ] : !quantum.obs
    return
  }
}
)mlir";

    DialectRegistry registry;
    registerDialects(registry);
    MLIRContext context(registry);
    ParserConfig config(&context, /*verifyAfterParse=*/false);
    OwningOpRef<ModuleOp> mod = parseSourceString<ModuleOp>(moduleStr, config);
    ASSERT_TRUE(mod);

    func::FuncOp f = *(*mod).getOps<func::FuncOp>().begin();
    auto namedObs = *f.getOps<NamedObsOp>().begin();
    auto extract = *f.getOps<ExtractOp>().begin();

    Value traced = traceQubit(namedObs.getQubit());
    ASSERT_TRUE(traced);
    EXPECT_EQ(traced, extract.getResult());
}

TEST(TraceQubitTests, scfForIterArg)
{
    std::string moduleStr = R"mlir(
module {
  func.func @f() {
    %reg = quantum.alloc(1) : !quantum.reg
    %q = quantum.extract %reg[0] : !quantum.reg -> !quantum.bit
    %lb = arith.constant 0 : index
    %ub = arith.constant 1 : index
    %step = arith.constant 1 : index
    %_ = scf.for %iv = %lb to %ub step %step iter_args(%qb = %q) -> (!quantum.bit) {
      %obs = quantum.namedobs %qb[PauliZ] : !quantum.obs
      scf.yield %qb : !quantum.bit
    }
    return
  }
}
)mlir";

    DialectRegistry registry;
    registerDialects(registry);
    MLIRContext context(registry);
    ParserConfig config(&context, /*verifyAfterParse=*/false);
    OwningOpRef<ModuleOp> mod = parseSourceString<ModuleOp>(moduleStr, config);
    ASSERT_TRUE(mod);

    func::FuncOp f = *(*mod).getOps<func::FuncOp>().begin();
    NamedObsOp namedObs;
    f.walk([&](NamedObsOp op) {
        namedObs = op;
        return WalkResult::interrupt();
    });
    ASSERT_TRUE(namedObs);
    auto extract = *f.getOps<ExtractOp>().begin();

    Value traced = traceQubit(namedObs.getQubit());
    ASSERT_TRUE(traced);
    EXPECT_EQ(traced, extract.getResult());
}

TEST(TraceQubitTests, scfIfMergedResult)
{
    std::string moduleStr = R"mlir(
module {
  func.func @f(%cond: i1) {
    %reg = quantum.alloc(1) : !quantum.reg
    %q = quantum.extract %reg[0] : !quantum.reg -> !quantum.bit
    %r = scf.if %cond -> (!quantum.bit) {
      scf.yield %q : !quantum.bit
    } else {
      scf.yield %q : !quantum.bit
    }
    %obs = quantum.namedobs %r[PauliZ] : !quantum.obs
    return
  }
}
)mlir";

    DialectRegistry registry;
    registerDialects(registry);
    MLIRContext context(registry);
    ParserConfig config(&context, /*verifyAfterParse=*/false);
    OwningOpRef<ModuleOp> mod = parseSourceString<ModuleOp>(moduleStr, config);
    ASSERT_TRUE(mod);

    func::FuncOp f = *(*mod).getOps<func::FuncOp>().begin();
    auto namedObs = *f.getOps<NamedObsOp>().begin();
    auto extract = *f.getOps<ExtractOp>().begin();

    Value traced = traceQubit(namedObs.getQubit());
    ASSERT_TRUE(traced);
    EXPECT_EQ(traced, extract.getResult());
}

TEST(TraceQubitTests, visitorInvokedAlongPath)
{
    std::string moduleStr = R"mlir(
module {
  func.func @f() {
    %reg = quantum.alloc(1) : !quantum.reg
    %q = quantum.extract %reg[0] : !quantum.reg -> !quantum.bit
    %qh = quantum.custom "Hadamard"() %q : !quantum.bit
    %obs = quantum.namedobs %qh[PauliZ] : !quantum.obs
    return
  }
}
)mlir";

    DialectRegistry registry;
    registerDialects(registry);
    MLIRContext context(registry);
    ParserConfig config(&context, /*verifyAfterParse=*/false);
    OwningOpRef<ModuleOp> mod = parseSourceString<ModuleOp>(moduleStr, config);
    ASSERT_TRUE(mod);

    func::FuncOp f = *(*mod).getOps<func::FuncOp>().begin();
    auto namedObs = *f.getOps<NamedObsOp>().begin();

    int visitCount = 0;
    Value traced = traceQubit(namedObs.getQubit(), [&](Value) { ++visitCount; });
    ASSERT_TRUE(traced);
    EXPECT_GE(visitCount, 2);
}

} // namespace
