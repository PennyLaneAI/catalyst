// Copyright 2024 Xanadu Quantum Technologies Inc.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "Quantum/IR/QuantumOps.h"
#include "mlir/AsmParser/AsmParser.h"
#include "mlir/AsmParser/AsmParserState.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "llvm/Support/SourceMgr.h"

#include "gtest/gtest.h"

using namespace mlir;

namespace {
TEST(MLIRParser, ParseQuantumIR)
{
    std::string moduleStr = R"mlir(
func.func @test_alloc_dealloc_no_fold() -> !quantum.bit {
  %r = quantum.alloc(3) : !quantum.reg
  %q = quantum.extract %r[0] : !quantum.reg -> !quantum.bit
  quantum.dealloc %r : !quantum.reg
  return %q : !quantum.bit
}
  )mlir";

    DialectRegistry registry;
    registry.insert<func::FuncDialect, catalyst::quantum::QuantumDialect>();
    MLIRContext context(registry);

    ParserConfig config(&context, /*verifyAfterParse=*/false);

    OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(moduleStr, config);
    ASSERT_TRUE(module);
}
} // namespace
